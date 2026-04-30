# src/dataset.py

from __future__ import annotations
from typing import Any, Dict, List, Sequence

from pathlib import Path
from torch.utils.data import Dataset

from src.ocr_utils import load_json
from src.path_utils import get_image_path_from_record
from src.prompt_utils import build_prompt, get_target_text, resolve_setting_name


class MemeCaptionDataset(Dataset):
    def __init__(
        self,
        json_path: str | Path | Sequence[str | Path],
        setting_name: str,
        allow_download: bool = False,
    ) -> None:
        self.setting_name = resolve_setting_name(setting_name)
        self.allow_download = allow_download

        if isinstance(json_path, (str, Path)):
            self.json_paths = [Path(json_path)]
        else:
            self.json_paths = [Path(p) for p in json_path]

        self.records: List[Dict[str, Any]] = []

        for path in self.json_paths:
            loaded = load_json(str(path))
            if not isinstance(loaded, list):
                raise TypeError(f"Dataset JSON must be a list of records: {path}")
            self.records.extend(loaded)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]

        image_path = get_image_path_from_record(
            record=record,
            allow_download=self.allow_download,
        )

        prompt = build_prompt(record, self.setting_name)
        target = get_target_text(record)

        return {
            "post_id": record.get("post_id", ""),
            "image_path": str(image_path),
            "prompt": prompt,
            "target": target,
            "record": record,
        }
