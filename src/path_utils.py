# src/path_utils.py

from __future__ import annotations
from typing import Dict, Any

import requests
from pathlib import Path
from urllib.parse import urlparse

from src.config import RAW_DIR


def _descend_into_memes_dirs(base_dir: str) -> str:
    current = Path(base_dir).resolve()

    while True:
        next_dir = current / "memes"
        if next_dir.is_dir():
            current = next_dir
        else:
            break

    return str(current)


def _infer_extension_from_url(url: str, default_ext: str = ".png") -> str:
    parsed = urlparse(url)
    _, ext = Path(parsed.path).with_suffix("").suffix, Path(parsed.path).suffix
    return ext.lower() if ext else default_ext


def _download_file(url: str, save_path: str, timeout: int = 30) -> str:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    with Path(save_path).open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return save_path


def get_download_image_path(record: Dict[str, Any]) -> Path:
    img_fname = str(record.get("img_fname", "")).strip()
    if img_fname:
        return Path(_descend_into_memes_dirs(RAW_DIR)) / img_fname

    post_id = str(record.get("post_id", "")).strip()
    if not post_id:
        raise ValueError("record must contain either 'img_fname' or 'post_id'")

    url = str(record.get("url", "")).strip()
    ext = _infer_extension_from_url(url, default_ext=".png")
    return Path(_descend_into_memes_dirs(RAW_DIR)) / f"memes_{post_id}{ext}"


def _candidate_image_paths(record: Dict[str, Any]) -> list[Path]:
    raw_dir = Path(RAW_DIR).resolve()
    memes_dir = Path(_descend_into_memes_dirs(RAW_DIR))

    candidates: list[Path] = []

    img_fname = str(record.get("img_fname", "")).strip()
    if img_fname:
        candidates.extend([
            memes_dir / img_fname,
            raw_dir / img_fname,
            raw_dir / "memes" / img_fname,
        ])

    post_id = str(record.get("post_id", "")).strip()
    if post_id:
        url_ext = _infer_extension_from_url(str(record.get("url", "")).strip(), default_ext=".png")
        candidates.extend([
            memes_dir / f"memes_{post_id}.png",
            memes_dir / f"memes_{post_id}{url_ext}",
            raw_dir / f"memes_{post_id}.png",
            raw_dir / f"memes_{post_id}{url_ext}",
            raw_dir / "memes" / f"memes_{post_id}.png",
            raw_dir / "memes" / f"memes_{post_id}{url_ext}",
        ])

    deduped: list[Path] = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)

    return deduped


def get_image_path_from_record(
    record: Dict[str, Any],
    allow_download: bool = False,
) -> str:
    if "post_id" not in record:
        raise KeyError("record must contain 'post_id'")

    post_id = str(record["post_id"]).strip()
    if not post_id:
        raise ValueError("record['post_id'] is empty")

    candidates = _candidate_image_paths(record)
    for path in candidates:
        if path.is_file():
            return str(path)

    if not allow_download:
        candidate_text = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(
            f"Image not found locally for post_id={post_id}. Checked:\n{candidate_text}"
        )

    url = str(record.get("url", "")).strip()
    if not url:
        raise FileNotFoundError(
            f"No local image and no valid URL for post_id={post_id}"
        )

    download_path = str(get_download_image_path(record))

    if Path(download_path).is_file():
        return download_path

    return _download_file(url, download_path)
