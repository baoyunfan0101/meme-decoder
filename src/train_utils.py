# src/train_utils.py

from __future__ import annotations

from torch.utils.data import DataLoader

from src.dataset import MemeCaptionDataset
from src.model_utils import collate_fn


def build_dataloader(
    json_path: str,
    setting_name: str,
    processor,
    batch_size: int = 1,
    shuffle: bool = True,
    allow_download: bool = False,
):
    dataset = MemeCaptionDataset(
        json_path=json_path,
        setting_name=setting_name,
        allow_download=allow_download,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(processor, batch),
    )
    return dataset, loader