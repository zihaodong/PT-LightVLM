import os
from transformers import T5Tokenizer
from torchvision import transforms
import json

from torch.utils.data import DataLoader
import torch
import argparse
from dataset.drivelm_dataset import DriveLMDataset, get_data_root
from modules.utils import print_trainable_parameters
from modules.model import get_model
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import yaml

device = torch.device('cuda')

img_sz = (224, 224)

data_root = get_data_root()

def setup_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed


def save_model(model, model_name):
    path = os.path.join('multi_frame_results', config.run_name, model_name + '.pth')
    torch.save(model, path)


def val_model(dloader, val_model):
    val_model.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, (inputs, attention_mask, imgs, labels) in tqdm(enumerate(dloader), total=len(dloader)):
            inputs, attention_mask, imgs, labels = inputs.to(device), attention_mask.to(device), imgs.to(
                device), labels.to(device)
            outputs = val_model(inputs, attention_mask, imgs, labels)
            val_loss += outputs.loss.item()

    return val_loss / len(dloader)


def save_stats(train_loss, val_loss, epochs, lr):
    stats_dict = {
        'losses': losses,
        'val losses': val_losses,
        'min train loss': train_loss,
        'min val loss': val_loss,
        'epochs': epochs,
        'learning rate': lr,
        'LM': 'T5-Small',
    }

    with open(os.path.join('multi_frame_results', config.run_name, 'stats.json'), 'w') as f:
        json.dump(stats_dict, f)


def custom_train(train_loss, val_loss, epochs, learning_rate):
    print('training device:', device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config.weight_decay)

    if config.lr_scheduler == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)
    elif config.lr_scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.final_lr
        )
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)

    for epoch in range(epochs, config.epochs):
        print('-------------------- EPOCH ' + str(epoch) + ' ---------------------')
        model.train()
        epoch_loss = 0
        for step, (inputs, attention_mask, imgs, labels) in tqdm(enumerate(train_dataloader),
                                                                 total=len(train_dataloader)):

            inputs, attention_mask, imgs, labels = inputs.to(device), attention_mask.to(device), imgs.to(
                device), labels.to(device)

            outputs = model(inputs, attention_mask, imgs, labels)
            loss = outputs.loss
            epoch_loss += loss.item()

            if step % config.checkpoint_frequency == 0:
                print()
                print('Loss: ' + str(loss.item()))

                hidden_states = outputs.logits
                outputs = torch.argmax(hidden_states, dim=-1)

                text_outputs = [processor.decode(output.to('cpu'), skip_special_tokens=True) for output in outputs]
                text_questions = [processor.decode(q.to('cpu'), skip_special_tokens=True) for q in inputs]
                text_labels = [processor.decode(a.to('cpu'), skip_special_tokens=True) for a in labels]
                print()
                print('Questions:')
                print(text_questions)
                print()
                print('Generated Answers:')
                print(text_outputs)
                print()
                print('Ground Truth Answers:')
                print(text_labels)

            # Back-propogate
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_train_loss = epoch_loss / len(train_dataloader)
        losses.append(epoch_train_loss)

        epoch_val_loss = val_model(val_dataloader, model)
        val_losses.append(epoch_val_loss)

        if not val_loss or epoch_val_loss < val_loss:
            val_loss = epoch_val_loss
            save_model(model.state_dict(), 'best_model')
        if not train_loss or train_loss > epoch_train_loss:
            train_loss = epoch_train_loss

        scheduler.step()

        print('Training Loss: ' + str(epoch_train_loss))
        print('Validation Loss: ' + str(epoch_val_loss))
        print('---------------------------------------------')

        save_model(model.state_dict(), 'last_model')
        epochs += 1
        save_stats(train_loss, val_loss, epochs, scheduler.get_last_lr()[0])

    return train_loss, val_loss


def save_experiment(statistics):
    """
    Saves the experiment multi_frame_results to a csv
    :param config: The hyperparameters used
    :param statistics: The accuracies for the training, validation, and test sets
    """
    trial_dict = {
        'Model name': [config.run_name],
        'Learning rate': [config.learning_rate],
        'Weight decay': [config.weight_decay],
        'Batch size': [config.batch_size],
        'Epochs': [config.epochs],
        'Min Training Loss': [statistics[0]],
        'Min Validation Loss': [statistics[1]],
        'Min Testing Loss': [statistics[2]],
    }

    trial_dict = pd.DataFrame(trial_dict)
    trial_dict.to_csv(os.path.join('multi_frame_results', config.run_name, 'multi_frame_results.csv'), index=False, header=True)


def get_dataloader(data_dir):
    train_dset = DriveLMDataset(
        input_file=os.path.join(data_dir, 'multi_frame_train.json'),
        tokenizer=processor,
        transform=transforms.Compose([
            transforms.Resize(img_sz),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ])
    )
    val_dset = DriveLMDataset(
        input_file=os.path.join(data_dir, 'multi_frame_val.json'),
        tokenizer=processor,
        transform=transforms.Compose([
            transforms.Resize(img_sz),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ])
    )
    test_dset = DriveLMDataset(
        input_file=os.path.join(data_dir, 'multi_frame_test.json'),
        tokenizer=processor,
        transform=transforms.Compose([
            transforms.Resize(img_sz),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ])
    )

    train_dataloader = DataLoader(train_dset, shuffle=True, batch_size=config.batch_size,
                                  num_workers=config.num_workers, collate_fn=train_dset.collate_fn)
    val_dataloader = DataLoader(val_dset, shuffle=False, batch_size=config.batch_size,
                                num_workers=config.num_workers, collate_fn=train_dset.collate_fn)
    test_dataloader = DataLoader(test_dset, shuffle=False, batch_size=config.batch_size,
                                 num_workers=config.num_workers, collate_fn=train_dset.collate_fn)
    return train_dataloader, val_dataloader, test_dataloader


def params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-yaml', default='config/stage1.yaml', type=str)
    parser.add_argument('--run-name', default='stage1', type=str)

    args = parser.parse_args()

    with open(args.config_yaml, 'r') as f:
        yml_arg = yaml.load(f, Loader=yaml.FullLoader)

    parser.set_defaults(**yml_arg)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    config = params()

    config.seed = setup_seed(config.seed)
    print('config:', config)

    losses = []
    val_losses = []

    min_train_loss = None
    min_val_loss = None

    best_model = None
    epochs_ran = 0

    model = get_model(config)

    processor = T5Tokenizer.from_pretrained('google-t5/t5-small')
    processor.add_tokens('<')

    if config.model_path:
        model.load_state_dict(torch.load(config.model_path, map_location=device), strict=False)
    model.to(device)

    print('Trainable Parameters for full model')
    print_trainable_parameters(model)

    train_dataloader, val_dataloader, test_dataloader = get_dataloader(os.path.join(data_root, 'data', 'multi_frame'))
    checkpoint_path = os.path.join('multi_frame_results', config.run_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    print(f'All model checkpoints and training stats will be saved in {checkpoint_path}')

    lr = config.learning_rate

    min_train_loss, min_val_loss = custom_train(min_train_loss, min_val_loss, epochs_ran, lr)

    model = get_model(config)
    model.load_state_dict(torch.load(os.path.join('multi_frame_results', config.run_name, 'best_model.pth')))
    model.to(device)

    test_loss = val_model(test_dataloader, model)
    statistics = [min_train_loss, min_val_loss, test_loss]
    save_experiment(statistics)

    print('training over........')