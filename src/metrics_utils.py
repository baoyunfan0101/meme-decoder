# src/metrics_utils.py

from __future__ import annotations
from typing import Dict, List, Sequence

import evaluate


def normalize_text(text: str) -> str:
    return str(text).strip()


def prepare_references(references: Sequence[str]) -> List[List[str]]:
    return [[normalize_text(ref)] for ref in references]


def compute_generation_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    bert_lang: str = "en",
) -> Dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")

    predictions = [normalize_text(x) for x in predictions]
    references = [normalize_text(x) for x in references]

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    bleu_result = bleu.compute(
        predictions=predictions,
        references=prepare_references(references),
        max_order=4,
    )

    rouge_result = rouge.compute(
        predictions=predictions,
        references=references,
    )

    bert_result = bertscore.compute(
        predictions=predictions,
        references=references,
        lang=bert_lang,
    )

    bert_p = sum(bert_result["precision"]) / len(bert_result["precision"]) if bert_result["precision"] else 0.0
    bert_r = sum(bert_result["recall"]) / len(bert_result["recall"]) if bert_result["recall"] else 0.0
    bert_f1 = sum(bert_result["f1"]) / len(bert_result["f1"]) if bert_result["f1"] else 0.0

    return {
        "bleu4": float(bleu_result["bleu"]),
        "rougeL": float(rouge_result["rougeL"]),
        "bert_precision": float(bert_p),
        "bert_recall": float(bert_r),
        "bert_f1": float(bert_f1),
        "num_samples": float(len(predictions)),
    }