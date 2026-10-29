# deep-learning-project

This project trains a CNN in PyTorch to classify German traffic signs (GTSRB, 43 classes) from raw
pixels. A baseline CNN is trained first, then a second model is trained with one change,
the addition of training-time data augmentation, so the two runs can be compared.

**Dataset:** [GTSRB (German Traffic Sign Recognition Benchmark)](https://benchmark.ini.rub.de/gtsrb_dataset.html)
(Stallkamp et al., 2012). The data is downloaded from the [`tanganke/gtsrb`](https://huggingface.co/datasets/tanganke/gtsrb)
Hugging Face mirror: 26,640 training images and 12,630 test images, same labels and split
structure as the original benchmark.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare Data

```bash
from datasets import load_dataset

train_set = load_dataset("tanganke/gtsrb", split="train")
test_set = load_dataset("tanganke/gtsrb", split="test")
```


## Running the notebook

```bash
source .venv/bin/activate       
jupyter notebook deep_learning.ipynb
```

Run all cells top to bottom to reproduce data loading, baseline training, the augmentation
comparison, and evaluation.

## Results

| Model | Final Validation Accuracy | Test Accuracy | Test Macro-F1 |
|---|---|---|---|
| Baseline | 98.35% | 81.96% | 0.751 |
| Baseline + Augmentation | 97.70% | 81.11% | 0.752 |

The two models are tied on the held-out test set. A notable finding is the gap
between validation accuracy (~98%) and test accuracy (~82%) for both models, possibly caused by GTSRB's
training images containing near-duplicate video tracks per sign instance.

## Report

`module_summary.pdf` is the written Deep Learning Systems Analysis Report (built from
`module_summary.html` via WeasyPrint), covering architecture decisions, the controlled comparison,
results interpretation, limitations, ethical considerations, and references.
