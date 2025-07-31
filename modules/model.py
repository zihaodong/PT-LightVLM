from transformers import T5ForConditionalGeneration
from timm import create_model
from torch import nn
import torch
from modules.utils import print_trainable_parameters
from modules.unireplknet import *
import math

device = torch.device('cuda')


class Gate(nn.Module):

    def __init__(self, in_channels, num_experts=4):
        super(Gate, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_experts)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Expert(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1, stride=2):
        super(Expert, self).__init__()

        if type(out_channels) in (list, tuple):
            assert len(out_channels) == num_blocks
        else:
            out_channels = [out_channels]

        self.layers = nn.ModuleList()
        for i in range(num_blocks):
            self.layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels[i], kernel_size=3, stride=stride, padding=1,
                                   output_padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels[i], out_channels[i], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels[i]),
                nn.ReLU()
            ))
            in_channels = out_channels[i]

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x


class CMoE(nn.Module):

    def __init__(self, in_channels, out_channels, num_experts=4, num_per_expert=1, top_k=4):
        super().__init__()
        self.gate = Gate(in_channels, num_experts)
        self.experts = nn.ModuleList()
        self.top_k = top_k

        for i in range(num_experts):
            self.experts.append(
                Expert(in_channels, out_channels, num_blocks=num_per_expert))

    def forward(self, x):
        batch_size = x.size(0)

        weights = self.gate(x)

        expert_outputs = torch.stack([expert(x) for expert in self.experts])

        topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=1)

        topk_weights = torch.softmax(topk_weights, dim=1).view(batch_size, self.top_k, 1, 1, 1)

        batch_idx = torch.arange(batch_size, device=x.device)[:, None]

        selected_outputs = expert_outputs[topk_indices, batch_idx]

        outputs = (selected_outputs * topk_weights).sum(dim=1)

        return outputs


class Projection(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(Projection, self).__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class MiTGM(nn.Module):
    def __init__(self, in_channels, num_blocks, text_dim, num_view):
        super(MiTGM, self).__init__()
        self.mlps = nn.ModuleList()
        self.cross_list = nn.ModuleList()
        self.learnable_tokens = nn.ParameterList()
        self.fcs = nn.ModuleList()

        for i, (in_channel, num_block) in enumerate(zip(in_channels, num_blocks)):
            learnable_token = nn.Parameter(torch.Tensor(num_view, text_dim))
            nn.init.kaiming_uniform_(learnable_token, a=math.sqrt(5))
            self.learnable_tokens.append(learnable_token)
            self.cross_list.append(CrossAttention(dim=text_dim))
            self.mlps.append(nn.Sequential(
                nn.Linear(text_dim, in_channel),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(in_channel, in_channel),
            ))
            self.fcs.append(nn.ModuleList())
            for k in range(num_block):
                self.fcs[i].append(nn.Sequential(
                    nn.Linear(in_channel, in_channel),
                ))

        self.initialize_fcs_to_zero()

    def initialize_fcs_to_zero(self):
        for i in range(len(self.fcs)):
            for k in range(len(self.fcs[i])):
                for param in self.fcs[i][k].parameters():
                    nn.init.constant_(param, 0)

    def forward(self, layer_outputs, text_key_padding_mask):

        bs = layer_outputs[0].shape[0]

        assert len(layer_outputs) == len(self.cross_list)

        weights = []

        for i, (cross, mlp, learnable_token) in enumerate(zip(self.cross_list, self.mlps, self.learnable_tokens)):
            learnable_token_expanded = learnable_token.unsqueeze(0).expand(bs, -1, -1)
            layer_outputs[i] = cross(learnable_token_expanded, layer_outputs[i], layer_outputs[i],
                                     text_key_padding_mask)
            layer_outputs[i] = mlp(layer_outputs[i])
            weights.append([])
            for k, fc in enumerate(self.fcs[i]):
                weights[i].append(fc(layer_outputs[i]))

        return weights


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.q_ln = nn.LayerNorm(dim)
        self.k_ln = nn.LayerNorm(dim)

    def forward(self, query, key, value, attention_mask=None):
        ln_query = self.q_ln(query)
        ln_key = self.k_ln(key)
        attn_output, _ = self.cross_attention(
            query=ln_query,
            key=ln_key,
            value=value,
            key_padding_mask=attention_mask
        )
        output = self.proj(attn_output) + query
        return output


class PTLightVLM(nn.Module):

    def __init__(self,
                 num_view=1,
                 moe_out_channels=None,
                 text_guide=False,
                 freeze_img_encoder=True,
                 num_experts=4,
                 encoder_output=None,
                 top_k=4):

        super().__init__()

        self.text_guide = text_guide
        self.model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        hidden_size = self.model.config.d_model

        self.layer_outputs = []
        self.encoder_output = [2, 3, 4, 5] if len(encoder_output) == 0 else encoder_output

        print('Trainable Parameters for LM model:')
        print_trainable_parameters(self.model)

        self.img_model = create_model(
            'unireplknet_a',
            pretrained=False,
            in_1k_pretrained=True,
            num_classes=1000,
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            head_init_scale=1.0,
        )

        if self.text_guide:
            for i, layer in enumerate(self.model.encoder.block):
                layer.register_forward_hook(self.create_hook(i))

            in_channels = [40, 80, 160, 320]
            num_blocks = [2, 2, 6, 2]
            self.mitgm = MiTGM(in_channels, num_blocks, text_dim=hidden_size, num_view=num_view)

            for stage_idx in range(0, 4):
                for block in self.img_model.stages[stage_idx]:
                    block.film = True

        self.modal_embeddings = nn.Embedding(2, hidden_size)
        self.modal_embeddings.weight.data.normal_(mean=0.0, std=0.02)

        self.moe_out_channels = moe_out_channels if moe_out_channels else [128, 64, 16]
        self.moe = CMoE(320, self.moe_out_channels, num_experts=num_experts,
                        num_per_expert=len(self.moe_out_channels), top_k=top_k)

        num_inputs = 7 * (2 ** len(self.moe_out_channels))
        self.projection = Projection(num_inputs * num_inputs, num_inputs * num_inputs // 2, hidden_size)

        self.cross_attn = CrossAttention(dim=hidden_size)

        if freeze_img_encoder:
            for param in self.img_model.parameters():
                param.requires_grad = False

    def encode_text(self, text_enc, attention_mask):
        self.layer_outputs = []
        encoder_outputs = self.model.encoder(
            input_ids=text_enc,
            attention_mask=attention_mask,
            return_dict=True
        )
        layer_outputs = None
        if self.text_guide:
            layer_outputs = [self.layer_outputs[i] for i in self.encoder_output]
        self.layer_outputs = []
        return encoder_outputs, layer_outputs

    def create_hook(self, layer_index):
        def hook(module, inputs, outputs):
            self.layer_outputs.append(outputs[0])

        return hook

    def encode_image(self, imgs, layer_outputs):

        is_multi_view = len(imgs.shape) > 4

        if is_multi_view:
            bs, num_view, C, H, W = imgs.shape
            imgs = imgs.view(-1, C, H, W)
            feature_maps = self.img_model(imgs, layer_outputs)
            feature_maps = self.moe(feature_maps)

            C, H, W = feature_maps.shape[-3:]
            feature_maps = feature_maps.view(bs, num_view, C, H, W)
            image_embeddings = torch.flatten(feature_maps, start_dim=3)

            image_embeddings = self.projection(image_embeddings)
            bz, num_imgs, c, dim = image_embeddings.shape[:]
            image_embeddings = image_embeddings.view(bz, num_view * c, dim)
        else:
            feature_maps = self.img_model(imgs, layer_outputs)

            feature_maps = self.moe(feature_maps)

            image_embeddings = torch.flatten(feature_maps, start_dim=2)

            image_embeddings = self.projection(image_embeddings)

        return image_embeddings

    def _forward(self, text_enc, attention_mask, images):

        text_key_padding_mask = ~attention_mask.bool()

        encoder_outputs, layer_outputs = self.encode_text(text_enc, attention_mask)

        text_encoding = encoder_outputs.last_hidden_state

        layer_outputs = self.mitgm(layer_outputs, text_key_padding_mask) if self.text_guide else None

        image_encoding = self.encode_image(images, layer_outputs)

        image_encoding = self.cross_attn(image_encoding, text_encoding, text_encoding,
                                         text_key_padding_mask)

        image_encoding = image_encoding + self.modal_embeddings(
            torch.ones((1, image_encoding.shape[1]), dtype=torch.int, device=device))
        text_encoding = text_encoding + self.modal_embeddings(
            torch.zeros((1, text_encoding.shape[1]), dtype=torch.int,
                        device=device))

        combined_encoding = torch.cat([image_encoding, text_encoding], dim=1)
        encoder_outputs.last_hidden_state = combined_encoding

        combined_attention_mask = torch.cat([
            torch.ones(text_enc.shape[0], image_encoding.shape[1], dtype=torch.long, device=device),
            attention_mask
        ], dim=1)

        return encoder_outputs, combined_attention_mask

    def forward(self, text_enc, attention_mask, images, labels=None):

        encoder_outputs, combined_attention_mask = self._forward(text_enc, attention_mask, images)

        return self.model(encoder_outputs=encoder_outputs, attention_mask=combined_attention_mask, labels=labels)

    def generate(self, text_enc, attention_mask, images):

        encoder_outputs, combined_attention_mask = self._forward(text_enc, attention_mask, images)

        decoder_input_ids = torch.ones((text_enc.shape[0], 1), dtype=torch.long,
                                       device=device) * self.model.config.decoder_start_token_id

        output_ids = self.model.generate(attention_mask=combined_attention_mask,
                                         decoder_input_ids=decoder_input_ids,
                                         encoder_outputs=encoder_outputs, max_length=512)

        return output_ids


def get_model(config):
    return PTLightVLM(
        config.num_view,
        config.moe_out_channels,
        config.text_guide,
        config.freeze_img_encoder,
        config.num_experts,
        config.encoder_output,
        config.top_k
    )
