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


def get_image_path_from_record(
    record: Dict[str, Any],
    allow_download: bool = False,
) -> str:
    if "post_id" not in record:
        raise KeyError("record must contain 'post_id'")

    post_id = str(record["post_id"]).strip()
    if not post_id:
        raise ValueError("record['post_id'] is empty")

    memes_dir = _descend_into_memes_dirs(RAW_DIR)

    filename = f"memes_{post_id}.png"
    local_path = str(Path(memes_dir) / filename)

    if Path(local_path).is_file():
        return local_path

    if not allow_download:
        raise FileNotFoundError(
            f"Image not found locally: {local_path} (post_id={post_id})"
        )

    url = str(record.get("url", "")).strip()
    if not url:
        raise FileNotFoundError(
            f"No local image and no valid URL for post_id={post_id}"
        )

    ext = _infer_extension_from_url(url, default_ext=".png")
    download_name = f"memes_{post_id}{ext}"
    download_path = str(Path(memes_dir) / download_name)

    if Path(download_path).is_file():
        return download_path

    return _download_file(url, download_path)