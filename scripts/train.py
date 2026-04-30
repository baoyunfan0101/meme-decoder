# scripts/train.py

from __future__ import annotations
from typing import Dict, Optional, List, Any

import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, PROCESSED_DIR, OUTPUT_DIR
from src.loss_utils import LossConfig, compute_training_loss
from src.model_utils import (
    DEFAULT_MODEL_NAME,
    forward_step,
    generate_one,
    load_processor_and_model,
    move_batch_to_device,
    collate_fn,
    get_parameter_summary,
    get_model_device,
)
from src.dataset import MemeCaptionDataset
from src.metrics_utils import compute_generation_metrics
from src.prompt_utils import resolve_setting_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-json", type=str, default=None)
    parser.add_argument("--val-json", type=str, default=None)

    parser.add_argument("--train-folds", type=int, nargs="*", default=None)
    parser.add_argument("--val-fold", type=int, default=None)
    parser.add_argument("--fold-prefix", type=str, default="memes")
    parser.add_argument("--fold-suffix", type=str, default=".ocr.json")

    parser.add_argument("--setting", type=str, default=None)
    parser.add_argument("--input-setting", type=str, default=None, help="Input setting alias: 1, 2, 3, 4, or 5.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--strategy", type=str, default="projector-only", choices=["projector-only", "projector-lora"])
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "label_smoothing", "bert", "ce+bert"])
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--bert-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)

    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-name", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--eval-max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--allow-download", action="store_true")

    parser.add_argument("--selection-metric", type=str, default="bert_f1", choices=["bleu4", "rougeL", "bert_f1"])
    parser.add_argument("--greater-is-better", action="store_true", default=True)

    return parser.parse_args()


def validate_data_args(args: argparse.Namespace) -> None:
    if args.setting is None and args.input_setting is None:
        raise ValueError("Provide either --setting or --input-setting.")
    args.setting = resolve_setting_name(args.input_setting if args.input_setting is not None else args.setting)

    using_json_mode = args.train_json is not None
    using_fold_mode = args.train_folds is not None and len(args.train_folds) > 0

    if using_json_mode and using_fold_mode:
        raise ValueError("Use either --train-json/--val-json or --train-folds/--val-fold, not both.")

    if not using_json_mode and not using_fold_mode:
        raise ValueError("You must provide either --train-json or --train-folds.")

    if using_fold_mode:
        if args.val_fold is None:
            raise ValueError("When using --train-folds, you must also provide --val-fold.")
        if args.val_fold in args.train_folds:
            raise ValueError("--val-fold must not be included in --train-folds.")

    if using_json_mode and args.val_json is None:
        print("Warning: --val-json is not provided. Validation will be skipped.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_processed_path(path: str | None) -> Optional[str]:
    if path is None:
        return None

    p = Path(path)
    if p.is_absolute():
        return str(p)

    processed_candidate = Path(PROCESSED_DIR) / path
    if processed_candidate.exists():
        return str(processed_candidate)

    project_candidate = Path(PROJECT_ROOT) / path
    return str(project_candidate)


def resolve_save_dir(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return OUTPUT_DIR / path


def build_fold_path(fold_idx: int, fold_prefix: str, fold_suffix: str) -> str:
    return str(Path(PROCESSED_DIR) / f"{fold_prefix}-fold{fold_idx}{fold_suffix}")


def build_train_and_val_paths(args: argparse.Namespace) -> tuple[List[str], Optional[List[str]]]:
    if args.train_json is not None:
        train_paths = [resolve_processed_path(args.train_json)]
        val_paths = [resolve_processed_path(args.val_json)] if args.val_json is not None else None
        return train_paths, val_paths

    train_paths = [
        build_fold_path(
            fold_idx=fold_idx,
            fold_prefix=args.fold_prefix,
            fold_suffix=args.fold_suffix,
        )
        for fold_idx in args.train_folds
    ]

    val_paths = [
        build_fold_path(
            fold_idx=args.val_fold,
            fold_prefix=args.fold_prefix,
            fold_suffix=args.fold_suffix,
        )
    ]

    return train_paths, val_paths


def make_run_name(args: argparse.Namespace) -> str:
    if args.save_name:
        return args.save_name

    model_stub = args.model_name.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if args.train_folds is not None and len(args.train_folds) > 0:
        train_fold_str = "-".join(str(x) for x in args.train_folds)
        split_stub = f"trainF{train_fold_str}_valF{args.val_fold}"
    else:
        split_stub = "jsonsplit"

    return f"{args.setting}_{split_stub}_{model_stub}_{args.loss}_{timestamp}"


def build_optimizer(model, lr: float, weight_decay: float) -> AdamW:
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    if not trainable_params:
        raise ValueError("No trainable parameters found. Check freezing / unfreezing logic in src/model_utils.py.")

    return AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


def build_dataloader_from_paths(
    json_paths: List[str],
    setting_name: str,
    processor,
    batch_size: int,
    shuffle: bool,
    allow_download: bool,
    max_samples: Optional[int] = None,
):
    dataset = MemeCaptionDataset(
        json_path=json_paths,
        setting_name=setting_name,
        allow_download=allow_download,
        max_samples=max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(processor, batch),
    )

    return dataset, loader


def evaluate_dataset(
    processor,
    model,
    dataset,
    max_new_tokens: int = 64,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()

    total = len(dataset)
    if max_samples is not None:
        total = min(total, max_samples)

    predictions: List[str] = []
    references: List[str] = []
    post_ids: List[str] = []

    for idx in tqdm(range(total), desc="Validation", leave=False):
        item = dataset[idx]
        pred = generate_one(
            processor=processor,
            model=model,
            image_path=item["image_path"],
            prompt=item["prompt"],
            max_new_tokens=max_new_tokens,
        )
        predictions.append(pred)
        references.append(item["target"])
        post_ids.append(item["post_id"])

    metrics = compute_generation_metrics(
        predictions=predictions,
        references=references,
        bert_lang="en",
    )

    return {
        "num_samples": total,
        "post_ids": post_ids,
        "predictions": predictions,
        "references": references,
        "metrics": metrics,
    }


def save_run_metadata(
    save_dir: Path,
    config_payload: Dict[str, object],
    history: List[Dict[str, object]],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    with (save_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    with (save_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def save_epoch_artifacts(
    epoch_dir: Path,
    processor,
    model,
    epoch_payload: Dict[str, object],
    val_payload: Optional[Dict[str, object]] = None,
) -> None:
    epoch_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(epoch_dir)
    processor.save_pretrained(epoch_dir)

    with (epoch_dir / "epoch_info.json").open("w", encoding="utf-8") as f:
        json.dump(epoch_payload, f, ensure_ascii=False, indent=2)

    if val_payload is not None:
        with (epoch_dir / "val_results.json").open("w", encoding="utf-8") as f:
            json.dump(val_payload, f, ensure_ascii=False, indent=2)


def save_best_artifacts(
    best_dir: Path,
    processor,
    model,
    best_payload: Dict[str, object],
    val_payload: Optional[Dict[str, object]] = None,
) -> None:
    best_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(best_dir)
    processor.save_pretrained(best_dir)

    with (best_dir / "best_info.json").open("w", encoding="utf-8") as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)

    if val_payload is not None:
        with (best_dir / "val_results.json").open("w", encoding="utf-8") as f:
            json.dump(val_payload, f, ensure_ascii=False, indent=2)


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    loss_config: LossConfig,
    device,
    grad_accum_steps: int,
    max_grad_norm: float,
    log_every: int,
    epoch_index: int,
) -> Dict[str, float]:
    model.train()

    running_loss = 0.0
    running_steps = 0

    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(train_loader, desc=f"Train epoch {epoch_index + 1}", leave=True)

    for step, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)
        outputs = forward_step(model, batch)

        loss, loss_info = compute_training_loss(
            logits=outputs.logits,
            labels=batch["labels"],
            loss_config=loss_config,
        )

        loss_for_backward = loss / grad_accum_steps
        loss_for_backward.backward()

        if step % grad_accum_steps == 0:
            trainable_params = [param for param in model.parameters() if param.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss.detach().item())
        running_steps += 1

        if step % log_every == 0 or step == 1:
            progress.set_postfix({"loss": f"{running_loss / running_steps:.4f}"})

    if running_steps % grad_accum_steps != 0:
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {
        "train_loss": running_loss / max(running_steps, 1),
    }


def is_better_metric(current: float, best: Optional[float], greater_is_better: bool = True) -> bool:
    if best is None:
        return True
    if greater_is_better:
        return current > best
    return current < best


def main() -> None:
    args = parse_args()
    validate_data_args(args)
    set_seed(args.seed)

    train_paths, val_paths = build_train_and_val_paths(args)

    save_root = resolve_save_dir(args.save_dir)
    run_name = make_run_name(args)
    save_dir = save_root / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    processor, model = load_processor_and_model(
        model_name=args.model_name,
        training_strategy=args.strategy,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        load_in_4bit=args.load_in_4bit,
    )
    device = get_model_device(model)
    parameter_summary = get_parameter_summary(model)

    train_dataset, train_loader = build_dataloader_from_paths(
        json_paths=train_paths,
        setting_name=args.setting,
        processor=processor,
        batch_size=args.batch_size,
        shuffle=True,
        allow_download=args.allow_download,
        max_samples=args.train_max_samples,
    )

    val_dataset = None
    if val_paths is not None:
        val_dataset, _ = build_dataloader_from_paths(
            json_paths=val_paths,
            setting_name=args.setting,
            processor=processor,
            batch_size=1,
            shuffle=False,
            allow_download=args.allow_download,
        )

    optimizer = build_optimizer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_config = LossConfig(
        name=args.loss,
        label_smoothing=args.label_smoothing,
        bert_weight=args.bert_weight,
        ce_weight=args.ce_weight,
    )

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"Train files: {train_paths}")
    if val_paths is not None:
        print(f"Val files: {val_paths}")
    print(f"Setting: {args.setting}")
    print(f"Model: {args.model_name}")
    print(f"Strategy: {args.strategy}")
    print(f"Max pixels: {args.max_pixels}")
    print(f"Load in 4bit: {args.load_in_4bit}")
    print(f"Loss: {args.loss}")
    print(f"Selection metric: {args.selection_metric}")
    print(f"Train size: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Val size: {len(val_dataset)}")
    print(f"Save dir: {save_dir}")
    print(f"Trainable params: {parameter_summary['trainable_params']:,} / {parameter_summary['total_params']:,} ({parameter_summary['trainable_ratio']:.6f})")

    history: List[Dict[str, object]] = []
    best_metric_value: Optional[float] = None
    best_epoch: Optional[int] = None

    config_payload = {
        "train_paths": train_paths,
        "val_paths": val_paths,
        "setting": args.setting,
        "model_name": args.model_name,
        "strategy": args.strategy,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "load_in_4bit": args.load_in_4bit,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_accum_steps": args.grad_accum_steps,
        "train_max_samples": args.train_max_samples,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "loss": args.loss,
        "label_smoothing": args.label_smoothing,
        "bert_weight": args.bert_weight,
        "ce_weight": args.ce_weight,
        "train_folds": args.train_folds,
        "val_fold": args.val_fold,
        "fold_prefix": args.fold_prefix,
        "fold_suffix": args.fold_suffix,
        "selection_metric": args.selection_metric,
        "max_new_tokens": args.max_new_tokens,
        "trainable_params": parameter_summary["trainable_params"],
        "total_params": parameter_summary["total_params"],
        "trainable_ratio": parameter_summary["trainable_ratio"],
        "trainable_parameter_names": parameter_summary["trainable_parameter_names"],
    }

    for epoch in range(args.epochs):
        epoch_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_config=loss_config,
            device=device,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            log_every=args.log_every,
            epoch_index=epoch,
        )

        print(f"Epoch {epoch + 1}/{args.epochs} - train_loss: {epoch_stats['train_loss']:.6f}")

        val_payload = None
        current_metric_value = None

        if val_dataset is not None:
            val_payload = evaluate_dataset(
                processor=processor,
                model=model,
                dataset=val_dataset,
                max_new_tokens=args.max_new_tokens,
                max_samples=args.eval_max_samples,
            )

            val_metrics = val_payload["metrics"]
            current_metric_value = float(val_metrics[args.selection_metric])

            print("Validation metrics:")
            print(f"  BLEU-4 : {val_metrics['bleu4']:.6f}")
            print(f"  ROUGE-L: {val_metrics['rougeL']:.6f}")
            print(f"  BERT-F1: {val_metrics['bert_f1']:.6f}")

            if val_payload["num_samples"] > 0:
                print("Validation preview:")
                print(f"  post_id: {val_payload['post_ids'][0]}")
                print(f"  pred   : {val_payload['predictions'][0]}")
                print(f"  ref    : {val_payload['references'][0]}")

        epoch_record: Dict[str, object] = {
            "epoch": epoch + 1,
            **epoch_stats,
        }

        if val_payload is not None:
            epoch_record["val_metrics"] = val_payload["metrics"]

        history.append(epoch_record)

        epoch_dir = save_dir / f"epoch_{epoch + 1}"
        epoch_payload = {
            "epoch": epoch + 1,
            "train_stats": epoch_stats,
            "selection_metric": args.selection_metric,
            "selection_metric_value": current_metric_value,
        }

        save_epoch_artifacts(
            epoch_dir=epoch_dir,
            processor=processor,
            model=model,
            epoch_payload=epoch_payload,
            val_payload=val_payload,
        )

        if val_payload is not None and current_metric_value is not None:
            if is_better_metric(current_metric_value, best_metric_value, greater_is_better=args.greater_is_better):
                best_metric_value = current_metric_value
                best_epoch = epoch + 1

                best_payload = {
                    "best_epoch": best_epoch,
                    "selection_metric": args.selection_metric,
                    "selection_metric_value": best_metric_value,
                }

                save_best_artifacts(
                    best_dir=save_dir / "best",
                    processor=processor,
                    model=model,
                    best_payload=best_payload,
                    val_payload=val_payload,
                )

                print(f"Updated best model at epoch {best_epoch} with {args.selection_metric}={best_metric_value:.6f}")

        save_run_metadata(
            save_dir=save_dir,
            config_payload=config_payload,
            history=history,
        )

    print("Training finished.")
    if best_epoch is not None:
        print(f"Best epoch: {best_epoch}")
        print(f"Best {args.selection_metric}: {best_metric_value:.6f}")


if __name__ == "__main__":
    main()
