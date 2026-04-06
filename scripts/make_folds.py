# scripts/make_folds.py

from __future__ import annotations
from typing import List, Dict, Any

import json
import random
import argparse
from pathlib import Path

from src.config import PROCESSED_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into K folds (generic, OCR-agnostic)")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file (relative to PROCESSED_DIR or absolute)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of folds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output file prefix (default: derived from input filename)",
    )
    parser.add_argument(
        "--save-manifest",
        action="store_true",
        help="Save fold manifest file",
    )

    return parser.parse_args()


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(PROCESSED_DIR) / path


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_folds(data: List[Dict[str, Any]], k: int, seed: int):
    random.seed(seed)
    data = data.copy()
    random.shuffle(data)

    folds = [[] for _ in range(k)]
    for i, item in enumerate(data):
        folds[i % k].append(item)

    return folds


def build_manifest(folds: List[List[Dict[str, Any]]], seed: int):
    manifest = {
        "seed": seed,
        "num_folds": len(folds),
        "folds": {},
    }

    for i, fold in enumerate(folds, start=1):
        fold_name = f"fold{i}"
        manifest["folds"][fold_name] = [
            item.get("post_id", f"idx_{j}") for j, item in enumerate(fold)
        ]

    return manifest


def infer_prefix_and_suffix(input_path: Path, prefix_arg: str | None):
    if prefix_arg is not None:
        return prefix_arg, ".json"

    name = input_path.name
    if name.endswith(".json"):
        stem = name[:-5]
        suffix = ".json"
    else:
        stem = name
        suffix = ""

    return stem, suffix


def main():
    args = parse_args()

    input_path = resolve_path(args.input)
    print(f"Loading: {input_path}")

    data = load_json(input_path)
    print(f"Total samples: {len(data)}")

    folds = split_folds(data, k=args.k, seed=args.seed)

    prefix, suffix = infer_prefix_and_suffix(input_path, args.prefix)

    for i, fold_data in enumerate(folds, start=1):
        out_name = f"{prefix}-fold{i}{suffix}"
        out_path = Path(PROCESSED_DIR) / out_name

        save_json(fold_data, out_path)
        print(f"Saved fold {i}: {len(fold_data)} samples -> {out_path}")

    if args.save_manifest:
        manifest = build_manifest(folds, seed=args.seed)
        manifest_path = Path(PROCESSED_DIR) / f"{prefix}-folds-manifest.json"

        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()