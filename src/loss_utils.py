# src/loss_utils.py

from __future__ import annotations
from typing import Dict


import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class LossConfig:
    name: str = "ce"
    label_smoothing: float = 0.0
    bert_weight: float = 0.0
    ce_weight: float = 1.0


def shift_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return shift_logits, shift_labels


def compute_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    shift_logits, shift_labels = shift_logits_and_labels(logits, labels)

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


def compute_label_smoothing_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float = 0.1,
) -> torch.Tensor:
    shift_logits, shift_labels = shift_logits_and_labels(logits, labels)

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        label_smoothing=smoothing,
    )
    return loss


def compute_bert_placeholder_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError(
        "BERT-style training loss is not implemented yet. "
        "The training pipeline already supports pluggable losses through src/loss_utils.py."
    )


def compute_training_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_config: LossConfig,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if loss_config.name == "ce":
        loss = compute_ce_loss(logits, labels)
        return loss, {"loss_total": float(loss.detach().item()), "loss_ce": float(loss.detach().item())}

    if loss_config.name == "label_smoothing":
        loss = compute_label_smoothing_loss(
            logits=logits,
            labels=labels,
            smoothing=loss_config.label_smoothing,
        )
        return loss, {
            "loss_total": float(loss.detach().item()),
            "loss_label_smoothing": float(loss.detach().item()),
        }

    if loss_config.name == "bert":
        loss = compute_bert_placeholder_loss(logits, labels)
        return loss, {"loss_total": float(loss.detach().item())}

    if loss_config.name == "ce+bert":
        ce_loss = compute_ce_loss(logits, labels)
        bert_loss = compute_bert_placeholder_loss(logits, labels)
        total = loss_config.ce_weight * ce_loss + loss_config.bert_weight * bert_loss
        return total, {
            "loss_total": float(total.detach().item()),
            "loss_ce": float(ce_loss.detach().item()),
            "loss_bert": float(bert_loss.detach().item()),
        }

    raise ValueError(f"Unsupported loss name: {loss_config.name}")