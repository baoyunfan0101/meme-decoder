# src/prompt_utils.py

from __future__ import annotations
from typing import Any, Dict, List


SETTINGS = {
    "meme_title": ["title"],
    "meme_imgcap": ["img_cap"],
    "meme_title_imgcap": ["title", "img_cap"],
    "meme_title_imgcap_ocr": ["title", "img_cap", "ocr"],
    "meme_title_imgcap_ocr_rationale": ["title", "img_cap", "ocr", "rationale"],
}


def format_img_captions(img_captions: List[str]) -> str:
    if not img_captions:
        return ""
    return " ".join(x.strip() for x in img_captions if str(x).strip())


def format_rationale(metaphors: List[Dict[str, Any]]) -> str:
    if not metaphors:
        return ""

    lines = []
    for item in metaphors:
        metaphor = str(item.get("metaphor", "")).strip()
        meaning = str(item.get("meaning", "")).strip()
        if metaphor and meaning:
            lines.append(f"- {metaphor} => {meaning}")

    return "\n".join(lines)


def get_target_text(record: Dict[str, Any]) -> str:
    meme_captions = record.get("meme_captions", [])
    if not meme_captions:
        return ""
    return str(meme_captions[0]).strip()


def build_prompt(record: Dict[str, Any], setting_name: str) -> str:
    if setting_name not in SETTINGS:
        raise ValueError(f"Unknown setting: {setting_name}")

    enabled = SETTINGS[setting_name]
    parts = [
        "Generate one concise explanatory meme caption.",
        "Output only one sentence.",
    ]

    if "title" in enabled:
        title = str(record.get("title", "")).strip()
        if title:
            parts.append(f"Title: {title}")

    if "img_cap" in enabled:
        img_caps = format_img_captions(record.get("img_captions", []))
        if img_caps:
            parts.append(f"Image captions: {img_caps}")

    if "ocr" in enabled:
        ocr_text = str(record.get("ocr_text", "")).strip()
        if ocr_text:
            parts.append(f"OCR text: {ocr_text}")

    if "rationale" in enabled:
        rationale = format_rationale(record.get("metaphors", []))
        if rationale:
            parts.append("Metaphor rationale:")
            parts.append(rationale)

    return "\n".join(parts)