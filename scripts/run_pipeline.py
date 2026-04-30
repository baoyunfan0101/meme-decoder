from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.prompt_utils import resolve_setting_name

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command entrypoint for MemeCap input and training-strategy ablations."
    )

    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--input-setting", type=str, required=True, help="One of 1, 2, 3, 4, 5.")
    parser.add_argument("--strategy", choices=["zero-shot", "projector-only", "projector-lora"], required=True)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)

    parser.add_argument("--raw-dir", type=str, default=None, help="Directory containing raw JSON files and cached meme images.")
    parser.add_argument("--processed-dir", type=str, default=None, help="Directory containing processed JSON files.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for checkpoints and metrics.")

    parser.add_argument("--train-json", type=str, default=None)
    parser.add_argument("--val-json", type=str, default=None)
    parser.add_argument("--trainval-json", type=str, default="memes-trainval.ocr.json")
    parser.add_argument("--train-folds", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument("--val-fold", type=int, default=5)
    parser.add_argument("--fold-prefix", type=str, default="memes")
    parser.add_argument("--fold-suffix", type=str, default=".ocr.json")
    parser.add_argument("--auto-make-folds", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--eval-json", type=str, default="memes-test.ocr.json")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--checkpoint-name", type=str, default="best")

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--eval-max-samples", type=int, default=None)
    parser.add_argument("--save-name", type=str, default=None)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing it.")

    return parser.parse_args()


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.raw_dir is not None:
        env["RAW_DIR"] = str(Path(args.raw_dir).expanduser())
    if args.processed_dir is not None:
        env["PROCESSED_DIR"] = str(Path(args.processed_dir).expanduser())
    if args.output_dir is not None:
        env["OUTPUT_DIR"] = str(Path(args.output_dir).expanduser())
    return env


def get_processed_dir(args: argparse.Namespace) -> Path:
    if args.processed_dir is not None:
        return Path(args.processed_dir).expanduser()
    env_value = os.environ.get("PROCESSED_DIR")
    if env_value:
        return Path(env_value).expanduser()
    return Path("data") / "processed"


def fold_paths(args: argparse.Namespace) -> list[Path]:
    processed_dir = get_processed_dir(args)
    folds = [*args.train_folds, args.val_fold]
    return [
        processed_dir / f"{args.fold_prefix}-fold{fold_idx}{args.fold_suffix}"
        for fold_idx in sorted(set(folds))
    ]


def ensure_folds(args: argparse.Namespace, env: dict[str, str]) -> None:
    if args.train_json is not None or not args.auto_make_folds:
        return

    missing = [path for path in fold_paths(args) if not path.exists()]
    if not missing:
        return

    processed_dir = get_processed_dir(args)
    source_path = processed_dir / args.trainval_json
    if not args.dry_run and not source_path.exists():
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Fold files are missing and source trainval JSON was not found: {source_path}\n"
            f"Missing folds:\n{missing_text}"
        )

    k = max([*args.train_folds, args.val_fold])
    cmd = [
        sys.executable,
        "-m",
        "scripts.make_folds",
        "--input",
        args.trainval_json,
        "--k",
        str(k),
        "--seed",
        "42",
        "--prefix",
        args.fold_prefix,
        "--suffix",
        args.fold_suffix,
        "--save-manifest",
    ]

    print("Fold files missing; creating folds:")
    print(" ".join(cmd))

    if not args.dry_run:
        subprocess.run(cmd, check=True, env=env)


def add_common_paths(cmd: list[str], args: argparse.Namespace) -> None:
    if args.allow_download:
        cmd.append("--allow-download")


def build_train_command(args: argparse.Namespace) -> list[str]:
    if args.strategy == "zero-shot":
        raise ValueError("zero-shot has no training step. Use --mode eval --strategy zero-shot.")

    cmd = [
        sys.executable,
        "-m",
        "scripts.train",
        "--input-setting",
        args.input_setting,
        "--strategy",
        args.strategy,
        "--model-name",
        args.model_name,
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
    ]

    if args.save_name is not None:
        cmd.extend(["--save-name", args.save_name])

    if args.eval_max_samples is not None:
        cmd.extend(["--eval-max-samples", str(args.eval_max_samples)])

    if args.train_json is not None:
        cmd.extend(["--train-json", args.train_json])
        if args.val_json is not None:
            cmd.extend(["--val-json", args.val_json])
    else:
        cmd.append("--train-folds")
        cmd.extend(str(x) for x in args.train_folds)
        cmd.extend([
            "--val-fold",
            str(args.val_fold),
            "--fold-prefix",
            args.fold_prefix,
            "--fold-suffix",
            args.fold_suffix,
        ])

    add_common_paths(cmd, args)
    return cmd


def build_eval_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.evaluate",
        "--input-setting",
        args.input_setting,
        "--strategy",
        args.strategy,
        "--model-name",
        args.model_name,
        "--eval-json",
        args.eval_json,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]

    if args.strategy != "zero-shot":
        if args.model_path is None:
            raise ValueError("--model-path is required for projector-only and projector-lora evaluation.")
        cmd.extend(["--model-path", args.model_path])
        if args.checkpoint_name is not None:
            cmd.extend(["--checkpoint-name", args.checkpoint_name])

    if args.save_name is not None:
        cmd.extend(["--save-name", args.save_name])

    add_common_paths(cmd, args)
    return cmd


def main() -> None:
    args = parse_args()
    args.input_setting = resolve_setting_name(args.input_setting)
    env = build_env(args)

    if args.mode == "train":
        ensure_folds(args, env)
        cmd = build_train_command(args)
    else:
        cmd = build_eval_command(args)

    print("Running command:")
    print(" ".join(cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
