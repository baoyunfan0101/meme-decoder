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

### One-command ablation runner

The recommended entrypoint is `scripts.run_pipeline`. It supports the five input settings and three training
strategies used in the project report.

Input settings:

| ID | Setting |
|---|---|
| 1 | meme + title |
| 2 | meme + image captions |
| 3 | meme + title + image captions |
| 4 | meme + title + image captions + OCR text |
| 5 | meme + title + image captions + OCR text + rationale |

Training strategies:

| Name | Behavior |
|---|---|
| `zero-shot` | no training; evaluate the base Qwen2.5-VL model |
| `projector-only` | freeze the model and train projector / merger / visual-to-text bridge parameters |
| `projector-lora` | train projector parameters plus LoRA adapters on LLM layers |

Example: zero-shot evaluation for setting 4:

```bash
python -m scripts.run_pipeline --mode eval --strategy zero-shot --input-setting 4 --eval-json memes-test.ocr.json
```

Example: train projector-only for setting 4 with 5-fold split:

```bash
python -m scripts.run_pipeline --mode train --strategy projector-only --input-setting 4 --epochs 2 --train-folds 1 2 3 4 --val-fold 5
```

Example: train projector + LoRA for setting 5:

```bash
python -m scripts.run_pipeline --mode train --strategy projector-lora --input-setting 5 --epochs 2 --grad-accum-steps 8 --lora-r 16
```

Example: evaluate a trained checkpoint:

```bash
python -m scripts.run_pipeline --mode eval --strategy projector-only --input-setting 4 --model-path outputs/checkpoints/<run_name> --checkpoint-name best --eval-json memes-test.ocr.json
```

If the data is stored in Google Drive / Colab, either set `RAW_DIR`, `PROCESSED_DIR`, and `OUTPUT_DIR` in an
environment file or pass them directly:

```bash
python -m scripts.download_data --output data
python -m scripts.download_images --processed-dir data/processed --raw-dir data/raw
python -m scripts.run_pipeline --mode train --strategy projector-lora --input-setting 4 --processed-dir data/processed --raw-dir data/raw
```

The expected data layout is:

```text
data/
  raw/
    memes-trainval.json
    memes-test.json
    # Images are not included initially; they are downloaded from each record's url.
    # After running scripts.download_images, cached images are stored here using img_fname.
    memes_bpet7l.png
    ...
  processed/
    memes-trainval.ocr.json
    memes-test.ocr.json
```

If fold files are not present, `scripts.run_pipeline` automatically creates them from
`data/processed/memes-trainval.ocr.json` before training.

Training and evaluation use the OCR-enriched files under `processed/`. The original JSON files under `raw/`
are needed for rerunning OCR and as the image cache location. If images are not cached yet, run
`scripts.download_images` first or pass `--allow-download` to `scripts.run_pipeline`.

### Download images from URLs:
```bash
python -m scripts.download_images --processed-dir data/processed --raw-dir data/raw
```

Alternatively, allow on-demand downloads during training/evaluation:
```bash
python -m scripts.run_pipeline --mode train --strategy projector-lora --input-setting 4 --processed-dir data/processed --raw-dir data/raw --allow-download
```

### Run OCR preprocessing:
```bash
python -m scripts.run_ocr --input memes-trainval.json --output memes-trainval.ocr.json --gpu
python -m scripts.run_ocr --input memes-test.json --output memes-test.ocr.json --gpu
```

If images are not cached locally yet:

```bash
python -m scripts.run_ocr --input memes-trainval.json --output memes-trainval.ocr.json --gpu --allow-download
python -m scripts.run_ocr --input memes-test.json --output memes-test.ocr.json --gpu --allow-download
```

### Split train/val set into 5 folds:
```bash
python -m scripts.make_folds --input memes-trainval.ocr.json --k 5 --seed 42 --prefix memes --suffix .ocr.json --save-manifest
```

### Train the model:
```bash
python -m scripts.train --train-folds 1 2 3 4 --val-fold 5 --setting meme_title_imgcap_ocr
```

### Evaluate the model:
```bash
python -m scripts.evaluate --model-path outputs/checkpoints/<run_name> --checkpoint-name best --eval-json memes-test.ocr.json --setting meme_title_imgcap_ocr
```
