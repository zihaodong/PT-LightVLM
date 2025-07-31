from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import json
import torch

device = torch.device('cuda')

ROOT = '/data/codalm'


def get_data_root():
    return ROOT


class CODALMDataset(Dataset):

    def __init__(self, data_dir, tokenizer, transform=None):
        self.data = [{}]
        self.data.pop()

        with open(os.path.join(data_dir, 'general_perception.jsonl')) as f:
            for line in f:
                self.data.append(json.loads(line))

        with open(os.path.join(data_dir, 'region_perception.jsonl')) as f:
            for line in f:
                self.data.append(json.loads(line))

        with open(os.path.join(data_dir, 'driving_suggestion.jsonl')) as f:
            for line in f:
                self.data.append(json.loads(line))

        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        q_text, a_text, img_path = self.data[idx]['question'], self.data[idx]['answer'], self.data[idx]['image']
        q_text = f"Question: {q_text} Answer:"
        imgs = self.transform(read_image(os.path.join(ROOT, img_path)).float())

        return q_text, imgs, a_text, img_path

    def collate_fn(self, batch):
        q_texts, imgs, a_texts, _ = zip(*batch)

        imgs = torch.stack(imgs, dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt")
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids

        return encodings.input_ids, encodings.attention_mask, imgs, labels


class CODALMDatasetForTest(Dataset):

    def __init__(self, data_dir, tokenizer, task_type='general_perception', transform=None):
        self.data = [{}]
        self.data.pop()
        with open(os.path.join(data_dir, f'{task_type}.jsonl')) as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q_text, a_text, img_path, q_id = self.data[idx]['question'], self.data[idx]['answer'], self.data[idx]['image'], \
                                         self.data[idx]['question_id']

        input_text = f"Question: {q_text} Answer:"

        imgs = self.transform(read_image(os.path.join(ROOT, img_path)).float())

        return input_text, q_text, imgs, a_text, img_path, q_id

    def collate_fn(self, batch):
        input_text, q_texts, imgs, a_texts, img_path, q_id = zip(*batch)

        imgs = torch.stack(imgs, dim=0)

        encodings = self.tokenizer(input_text, padding=True, return_tensors="pt")
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids

        return list(q_texts), encodings.input_ids, encodings.attention_mask, imgs, labels, img_path, q_id