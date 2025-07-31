# Progressive Text-Guided Lightweight Vision-Language Models with Decoupling-and-Aggregating for Autonomous Driving



## Datasets

Please follow [EM-VLM4AD](https://github.com/akshaygopalkr/EM-VLM4AD) and [CODA-LM](https://github.com/DLUT-LYZ/CODA-LM) to prepare all datasets.



## Training

- For DriveLM dataset, you can run `python train_drivelm.py`.
- For CODA-LM dataset, you can run `python train_codalm.py`.



## Evaluation

- For DriveLM dataset, you can run `python test_drivelm.py ` to generate BLEU-4, CIDEr, METEOR, and ROUGE_L metrics.
- For CODA-LM dataset, please run `python test_codalm.py` and follow  [CODA-LM](https://github.com/DLUT-LYZ/CODA-LM) instructions to generate Text-Score



## Acknowledge

Our code are based on [EM-VLM4AD](https://github.com/akshaygopalkr/EM-VLM4AD) and [UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet) repository. We thank the authors for releasing their code.