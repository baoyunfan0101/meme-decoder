# src/ocr_utils.py

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import re
import json
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass

import easyocr


@dataclass
class OCRConfig:
    langs: Sequence[str] = ("en",)
    gpu: bool = True
    min_confidence: float = 0.30
    paragraph: bool = False
    detail: int = 1
    normalize_whitespace: bool = True
    uppercase: bool = False
    keep_linebreaks: bool = False


def load_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_ocr_text(
    text: str,
    normalize_whitespace: bool = True,
    uppercase: bool = False,
    keep_linebreaks: bool = False,
) -> str:
    text = text.replace("\u00a0", " ").replace("\ufeff", " ").strip()

    if not keep_linebreaks:
        text = text.replace("\n", " ")

    if normalize_whitespace:
        if keep_linebreaks:
            text = "\n".join(re.sub(r"\s+", " ", line).strip() for line in text.splitlines())
            text = "\n".join(line for line in text.splitlines() if line)
        else:
            text = re.sub(r"\s+", " ", text).strip()

    if uppercase:
        text = text.upper()

    return text


def _bbox_sort_key(bbox: List[List[float]]) -> Tuple[float, float]:
    ys = [p[1] for p in bbox]
    xs = [p[0] for p in bbox]
    return (min(ys), min(xs))


class EasyOCRProcessor:
    def __init__(self, config: Optional[OCRConfig] = None) -> None:
        self.config = config or OCRConfig()
        self.reader = easyocr.Reader(list(self.config.langs), gpu=self.config.gpu)

    def read(self, image_path: str) -> Dict[str, Any]:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        raw_results = self.reader.readtext(
            image_path,
            detail=self.config.detail,
            paragraph=self.config.paragraph,
        )

        kept_items: List[Dict[str, Any]] = []
        texts: List[str] = []
        confidences: List[float] = []

        for item in raw_results:
            if len(item) != 3:
                continue

            bbox, text, confidence = item
            confidence = float(confidence)

            if confidence < self.config.min_confidence:
                continue

            text = clean_ocr_text(
                text=text,
                normalize_whitespace=self.config.normalize_whitespace,
                uppercase=self.config.uppercase,
                keep_linebreaks=self.config.keep_linebreaks,
            )

            if not text:
                continue

            kept_items.append(
                {
                    "bbox": bbox,
                    "text": text,
                    "confidence": confidence,
                }
            )

        kept_items.sort(key=lambda x: _bbox_sort_key(x["bbox"]))

        for item in kept_items:
            texts.append(item["text"])
            confidences.append(item["confidence"])

        joined_text = "\n".join(texts) if self.config.keep_linebreaks else " ".join(texts)
        joined_text = clean_ocr_text(
            text=joined_text,
            normalize_whitespace=self.config.normalize_whitespace,
            uppercase=self.config.uppercase,
            keep_linebreaks=self.config.keep_linebreaks,
        )

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "ocr_text": joined_text,
            "ocr_confidence": avg_conf,
            "ocr_items": kept_items,
            "ocr_engine": "easyocr",
        }


def enrich_record_with_ocr(
    record: Dict[str, Any],
    image_path: str,
    processor: EasyOCRProcessor,
    overwrite: bool = False,
    keep_ocr_items: bool = False,
) -> Dict[str, Any]:
    result = dict(record)

    if not overwrite and str(result.get("ocr_text", "")).strip():
        return result

    ocr_result = processor.read(image_path)

    result["ocr_text"] = ocr_result["ocr_text"]
    result["ocr_confidence"] = ocr_result["ocr_confidence"]
    result["ocr_engine"] = ocr_result["ocr_engine"]

    if keep_ocr_items:
        result["ocr_items"] = ocr_result["ocr_items"]

    return result


def enrich_dataset_with_ocr(
    records: Sequence[Dict[str, Any]],
    image_path_resolver,
    processor: EasyOCRProcessor,
    overwrite: bool = False,
    keep_ocr_items: bool = False,
    error_mode: str = "record",
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []

    for idx, record in enumerate(tqdm(records, desc="Running OCR")):
        try:
            image_path = image_path_resolver(record)
            enriched = enrich_record_with_ocr(
                record=record,
                image_path=image_path,
                processor=processor,
                overwrite=overwrite,
                keep_ocr_items=keep_ocr_items,
            )
            output.append(enriched)
        except Exception as e:
            if error_mode == "raise":
                raise
            failed = dict(record)
            failed["ocr_text"] = ""
            failed["ocr_confidence"] = 0.0
            failed["ocr_engine"] = "easyocr"
            failed["ocr_error"] = f"{type(e).__name__}: {e}"
            output.append(failed)

    return output


def summarize_ocr_dataset(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    with_text = 0
    failed = 0
    confidences: List[float] = []

    for record in records:
        text = str(record.get("ocr_text", "")).strip()
        if text:
            with_text += 1
        if "ocr_error" in record:
            failed += 1
        conf = record.get("ocr_confidence", None)
        if isinstance(conf, (int, float)):
            confidences.append(float(conf))

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "total_records": total,
        "records_with_ocr_text": with_text,
        "failed_records": failed,
        "avg_ocr_confidence": avg_conf,
    }