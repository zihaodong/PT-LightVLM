import argparse
import pdb
import os
import torch
import os
from dataset.codalm_dataset import CODALMDatasetForTest, get_data_root
from modules.model import get_model
from tqdm import tqdm as progress_bar
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import yaml

data_root = get_data_root()
device = torch.device('cuda')
img_sz = (224, 224)


def val_model(dloader, save_path):
    model.eval()
    results = []

    with torch.no_grad():
        for idx, (q_texts, encodings, attention_mask, imgs, labels, img_paths, q_ids) in progress_bar(
                enumerate(dloader),
                total=len(
                    dloader)):

            encodings, attention_mask, imgs = encodings.to(device), attention_mask.to(device), imgs.to(device)
            outputs = model.generate(encodings, attention_mask, imgs)

            text_outputs = [processor.decode(output, skip_special_tokens=True) for output in outputs]

            if idx % 50 == 0:
                print(q_texts)
                print(text_outputs)

            for q_text, img_path, q_id, answer in zip(q_texts, img_paths, q_ids, text_outputs):
                results.append({
                    "question_id": q_id,
                    "image": img_path,
                    "question": q_text,
                    "answer": answer
                })

    with open(save_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"Results saved to {save_path}")


def params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', default='', type=str)
    parser.add_argument('--model-path', default='', type=str)
    parser.add_argument('--config-yaml', default='config/stage2.yaml', type=str)
    args = parser.parse_args()

    with open(args.config_yaml, 'r', encoding='utf-8') as f:
        yml_arg = yaml.load(f, Loader=yaml.FullLoader)

    parser.set_defaults(**yml_arg)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = params()

    model = get_model(config)

    processor = T5Tokenizer.from_pretrained('google-t5/t5-small')

    model.load_state_dict(torch.load(config.model_path, map_location=device))
    model.to(device)

    os.makedirs(config.save_dir, exist_ok=True)

    task_types = ['general_perception', 'region_perception', 'driving_suggestion']

    for task_type in task_types:
        print(f"Processing {task_type} task...")

        test_dset = CODALMDatasetForTest(
            data_dir=os.path.join(data_root, 'CODA-LM', 'Mini', 'vqa_anno'),
            tokenizer=processor,
            transform=transforms.Compose([
                transforms.Resize(img_sz),
                transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
            ]),
            task_type=task_type
        )
        test_dloader = DataLoader(test_dset, shuffle=False, batch_size=4, drop_last=False,
                                  collate_fn=test_dset.collate_fn, num_workers=config.num_workers)

        save_path = os.path.join(config.save_dir, f'{task_type}_answer.jsonl')

        val_model(test_dloader, save_path)
