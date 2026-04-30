from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import PROCESSED_DIR, RAW_DIR
from src.path_utils import get_image_path_from_record


def progress_iter(items, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(items, desc=desc)
    except ImportError:
        print(desc)
        return items


def load_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/cache meme images referenced by dataset JSON files.")
    parser.add_argument(
        "--json",
        nargs="+",
        default=["memes-trainval.ocr.json", "memes-test.ocr.json"],
        help="Dataset JSON files. Relative paths are resolved against PROCESSED_DIR first, then RAW_DIR.",
    )
    parser.add_argument("--raw-dir", type=str, default=None, help="Directory where images will be cached.")
    parser.add_argument("--processed-dir", type=str, default=None, help="Directory containing processed JSON files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum records per JSON, useful for testing.")
    return parser.parse_args()


def resolve_json_path(path: str, processed_dir: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p

    processed_candidate = processed_dir / path
    if processed_candidate.exists():
        return processed_candidate

    raw_candidate = Path(RAW_DIR) / path
    if raw_candidate.exists():
        return raw_candidate

    return processed_candidate


def main() -> None:
    args = parse_args()

    if args.raw_dir is not None:
        import src.config as config

        config.RAW_DIR = Path(args.raw_dir)
        import src.path_utils as path_utils

        path_utils.RAW_DIR = Path(args.raw_dir)

    processed_dir = Path(args.processed_dir) if args.processed_dir is not None else Path(PROCESSED_DIR)

    total_downloaded = 0
    total_failed = 0

    for json_arg in args.json:
        json_path = resolve_json_path(json_arg, processed_dir)
        records: Any = load_json(str(json_path))
        if not isinstance(records, list):
            raise TypeError(f"Expected list in {json_path}, got {type(records).__name__}")

        if args.limit is not None:
            records = records[: args.limit]

        print(f"Downloading images for {json_path} ({len(records)} records)")
        for record in progress_iter(records, desc=json_path.name):
            try:
                get_image_path_from_record(record, allow_download=True)
                total_downloaded += 1
            except Exception as exc:
                total_failed += 1
                post_id = record.get("post_id", "")
                print(f"Failed post_id={post_id}: {type(exc).__name__}: {exc}")

    print(f"Finished image cache. Success: {total_downloaded}; failed: {total_failed}")


if __name__ == "__main__":
    main()
