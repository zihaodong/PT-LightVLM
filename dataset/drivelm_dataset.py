from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import json
import torch

device = torch.device('cuda')

ROOT = '/data/drivelm'


def get_data_root():
    return ROOT


class DriveLMDataset(Dataset):

    def __init__(self, input_file, tokenizer, transform=None):
        with open(input_file) as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qa, img_path = self.data[idx]
        img_path = list(img_path.values())

        q_text, a_text = qa['Q'], qa['A']
        q_text = f"Question: {q_text} Answer:"

        imgs = [self.transform(read_image(os.path.join(ROOT, p)).float()) for p in img_path]
        imgs = torch.stack(imgs, dim=0)

        return q_text, imgs, a_text, sorted(list(img_path))


    def collate_fn(self, batch):
        q_texts, imgs, a_texts, _ = zip(*batch)
        imgs = torch.stack(list(imgs), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt")
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids

        return encodings.input_ids, encodings.attention_mask, imgs, labels

    def test_collate_fn(self, batch):
        q_texts, imgs, a_texts, img_path = zip(*batch)
        imgs = torch.stack(list(imgs), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt")
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids

        return list(q_texts), encodings.input_ids, encodings.attention_mask, imgs, labels, img_path
