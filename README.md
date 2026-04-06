# meme-decoder

## Overview

- **Model:** Qwen/Qwen2.5-VL-3B-Instruct  
- **Dataset:** [MemeCap](https://github.com/eujhwang/meme-cap)
  
- **Input Settings:**
  1. meme + title  
  2. meme + img captions  
  3. meme + title + img captions  
  4. meme + title + img captions + OCR text  
  5. meme + title + img captions + OCR text + rationale  

## Project Structure

```text
project_root/
├── scripts/
│   ├── evaluate.py
│   ├── make_folds.py
│   ├── run_ocr.py
│   └── train.py
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── loss_utils.py
│   ├── metrics_utils.py
│   ├── model_utils.py
│   ├── ocr_utils.py
│   ├── path_utils.py
│   ├── prompt_utils.py
│   └── train_utils.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── checkpoints/
│   └── metrics/
│
├── .env.example
├── .gitignore
├── LICENSE
├── pipeline_colab.ipynb
├── README.md
└── requirements.txt
```

## Workflow

### Run OCR preprocessing:
```bash
python -m scripts.run_ocr --input memes-trainval.json --output memes-trainval.ocr.json --gpu
python -m scripts.run_ocr --input memes-test.json --output memes-test.ocr.json --gpu
```

### Split train/val set into 5 folds:
```bash
python -m scripts.make_folds --input memes-trainval.ocr.json --k 5 --seed 42 --prefix memes --save-manifest
```

### Train the model:
```bash
python -m scripts.train --train-folds 1 2 3 4 --val-fold 5 --setting meme_title_imgcap_ocr
```

### Evaluate the model:
```bash
python -m scripts.evaluate --model-path outputs/checkpoints/<run_name> --checkpoint-name best --eval-json memes-test.ocr.json --setting meme_title_imgcap_ocr
```