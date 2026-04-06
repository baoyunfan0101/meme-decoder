# scripts/run_ocr.py

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import RAW_DIR, PROCESSED_DIR
from src.path_utils import get_image_path_from_record
from src.ocr_utils import (
    EasyOCRProcessor,
    OCRConfig,
    enrich_dataset_with_ocr,
    load_json,
    save_json,
    summarize_ocr_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR over a meme dataset JSON file.")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file. Can be absolute or relative to RAW_DIR.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSON file. Can be absolute or relative to PROCESSED_DIR.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for EasyOCR.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU for EasyOCR.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.30,
        help="Minimum confidence threshold for keeping OCR text.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ocr_text fields.",
    )
    parser.add_argument(
        "--keep-ocr-items",
        action="store_true",
        help="Keep per-box OCR outputs in ocr_items.",
    )
    parser.add_argument(
        "--uppercase",
        action="store_true",
        help="Convert OCR text to uppercase.",
    )
    parser.add_argument(
        "--keep-linebreaks",
        action="store_true",
        help="Keep OCR lines separated by newlines instead of joining with spaces.",
    )
    parser.add_argument(
        "--error-mode",
        type=str,
        default="record",
        choices=["record", "raise"],
        help="How to handle OCR errors.",
    )

    return parser.parse_args()


def resolve_input_path(path: str) -> str:
    if Path(path).is_absolute():
        return path
    return str(Path(RAW_DIR) / path)


def resolve_output_path(path: str) -> str:
    if Path(path).is_absolute():
        return path
    return str(Path(PROCESSED_DIR) / path)


def main() -> None:
    args = parse_args()

    if args.gpu and args.cpu:
        raise ValueError("Choose only one of --gpu or --cpu")

    use_gpu = True
    if args.cpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True

    input_path = resolve_input_path(args.input)
    output_path = resolve_output_path(args.output)

    records = load_json(input_path)
    if not isinstance(records, list):
        raise TypeError(f"Expected top-level JSON list, got: {type(records).__name__}")

    processor = EasyOCRProcessor(
        OCRConfig(
            langs=("en",),
            gpu=use_gpu,
            min_confidence=args.min_confidence,
            uppercase=args.uppercase,
            keep_linebreaks=args.keep_linebreaks,
        )
    )

    enriched = enrich_dataset_with_ocr(
        records=records,
        image_path_resolver=get_image_path_from_record,
        processor=processor,
        raw_dir=RAW_DIR,
        overwrite=args.overwrite,
        keep_ocr_items=args.keep_ocr_items,
        error_mode=args.error_mode,
    )

    save_json(enriched, output_path)

    summary = summarize_ocr_dataset(enriched)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()