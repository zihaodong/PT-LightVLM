import argparse
import pdb
import os
import torch
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import os
from dataset.drivelm_dataset import DriveLMDataset, get_data_root
from modules.model import get_model
from tqdm import tqdm as progress_bar
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import pandas as pd
import yaml

data_root = get_data_root()
device = torch.device('cuda')
img_sz = (224, 224)


def val_model(dloader):
    model.eval()
    ids_answered = set()
    test_data = []

    with torch.no_grad():
        for idx, (q_texts, encodings, attention_mask, imgs, labels, img_paths) in progress_bar(enumerate(dloader),
                                                                                               total=len(dloader)):

            encodings, attention_mask, imgs, labels = encodings.to(device), attention_mask.to(device), imgs.to(
                device), labels.to(device)
            outputs = model.generate(encodings, attention_mask, imgs)

            # Get the text output
            text_outputs = [processor.decode(output, skip_special_tokens=True) for output in outputs]

            if idx % 100 == 0:
                print(q_texts)
                print(text_outputs)

            for image_path, q_text, text_output in zip(img_paths, q_texts, text_outputs):

                img_key = image_path[0]

                # Skip duplicate questions
                if image_id_dict[img_key + ' ' + q_text][0] in ids_answered:
                    continue
                if len(text_output) > config.max_len:
                    continue

                ids_answered.add(image_id_dict[img_key + ' ' + q_text][0])
                test_data.append({'image_id': image_id_dict[img_key + ' ' + q_text][0], 'caption': text_output})

    # Save test output to file
    with open(os.path.join(config.save_dir, 'predictions.json'), 'w') as f:
        json.dump(test_data, f)


def save_experiment():
    """
    Saves the experiment results to a csv
    :param config: The hyperparameters used
    :param statistics: The accuracies for the training, validation, and test sets
    """

    trial_dict = {}

    # Add metrics to dictionary
    for metric, score in coco_eval.eval.items():
        trial_dict[metric] = [score]

    trial_dict = pd.DataFrame(trial_dict)
    trial_dict.to_csv(
        os.path.join(config.save_dir, 'metrics.csv'),
        index=False, header=True)


def params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-len', default=512, type=int, help='Max length for generating sequence')
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
    image_id = 'image_id.json'
    dataset_name = 'multi_frame_test.json'
    label_name = 'multi_frame_test_coco.json'

    # Load processors and models
    model = get_model(config)
    model.load_state_dict(torch.load(config.model_path, map_location=device))
    model.to(device)

    os.makedirs(config.save_dir, exist_ok=True)

    processor = T5Tokenizer.from_pretrained('google-t5/t5-small')

    processor.add_tokens('<')

    test_dset = DriveLMDataset(
        input_file=os.path.join(data_root, 'data', 'multi_frame',
                                dataset_name),
        tokenizer=processor,
        transform=transforms.Compose([
            transforms.Resize(img_sz),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ])
    )
    test_dloader = DataLoader(test_dset, shuffle=False, batch_size=config.batch_size, drop_last=False,
                              collate_fn=test_dset.test_collate_fn, num_workers=config.num_workers)

    with open(os.path.join(data_root, 'data', 'multi_frame', image_id)) as f:
        image_id_dict = json.load(f)

    val_model(test_dloader)

    annotation_file = os.path.join(data_root, 'data', 'multi_frame', label_name)
    results_file = os.path.join(config.save_dir, 'predictions.json')

    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    coco_eval = COCOEvalCap(coco, coco_result)

    coco_eval.params['image_id'] = coco_result.getImgIds()

    coco_eval.evaluate()

    save_experiment()
