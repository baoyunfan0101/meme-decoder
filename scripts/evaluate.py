# scripts/evaluate.py

from __future__ import annotations
from typing import List, Dict, Any, Optional

import json
import argparse
from tqdm import tqdm
from pathlib import Path

from src.config import PROJECT_ROOT, PROCESSED_DIR, OUTPUT_DIR
from src.dataset import MemeCaptionDataset
from src.metrics_utils import compute_generation_metrics
from src.model_utils import DEFAULT_MODEL_NAME, load_processor_and_model_for_inference, generate_one
from src.prompt_utils import resolve_setting_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--eval-json", type=str, required=True)
    parser.add_argument("--setting", type=str, default=None)
    parser.add_argument("--input-setting", type=str, default=None, help="Input setting alias: 1, 2, 3, 4, or 5.")
    parser.add_argument("--strategy", type=str, default="projector-only", choices=["zero-shot", "projector-only", "projector-lora"])
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)

    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--bert-lang", type=str, default="en")
    parser.add_argument("--allow-download", action="store_true")

    parser.add_argument("--save-dir", type=str, default="metrics")
    parser.add_argument("--save-name", type=str, default=None)

    return parser.parse_args()


def resolve_processed_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)

    processed_candidate = Path(PROCESSED_DIR) / path
    if processed_candidate.exists():
        return str(processed_candidate)

    return str(Path(PROJECT_ROOT) / path)


def resolve_project_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p

    output_candidate = Path(OUTPUT_DIR) / path
    if output_candidate.exists():
        return output_candidate

    project_candidate = Path(PROJECT_ROOT) / path
    return project_candidate


def resolve_save_dir(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(OUTPUT_DIR) / path


def resolve_model_checkpoint(model_path: str, checkpoint_name: Optional[str]) -> Path:
    base_path = resolve_project_path(model_path)

    if checkpoint_name is not None:
        checkpoint_path = base_path / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
        return checkpoint_path

    if (base_path / "config.json").exists():
        return base_path

    if (base_path / "best").exists():
        return base_path / "best"

    raise FileNotFoundError(
        "Could not resolve a loadable model checkpoint. "
        "Pass a direct checkpoint path or use --checkpoint-name best / epoch_1 / epoch_2 ..."
    )


def make_save_name(
    checkpoint_path: Path,
    eval_json: str,
    setting: str,
    save_name: Optional[str],
) -> str:
    if save_name:
        return save_name

    checkpoint_stub = checkpoint_path.parent.name + "_" + checkpoint_path.name if checkpoint_path.parent != checkpoint_path else checkpoint_path.name
    eval_stub = Path(eval_json).stem
    return f"{setting}_{checkpoint_stub}_{eval_stub}_metrics.json"


def main() -> None:
    args = parse_args()
    if args.setting is None and args.input_setting is None:
        raise ValueError("Provide either --setting or --input-setting.")
    args.setting = resolve_setting_name(args.input_setting if args.input_setting is not None else args.setting)

    checkpoint_path = None
    if args.strategy != "zero-shot":
        if args.model_path is None:
            raise ValueError("--model-path is required unless --strategy zero-shot is used.")
        checkpoint_path = resolve_model_checkpoint(args.model_path, args.checkpoint_name)

    eval_json = resolve_processed_path(args.eval_json)

    save_dir = resolve_save_dir(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / make_save_name(
        checkpoint_path=checkpoint_path if checkpoint_path is not None else Path(args.model_name),
        eval_json=eval_json,
        setting=args.setting,
        save_name=args.save_name,
    )

    if args.strategy == "zero-shot":
        processor, model = load_processor_and_model_for_inference(
            model_name=args.model_name,
            training_strategy=args.strategy,
            adapter_path=None,
        )
        model_path_for_payload = args.model_name
    elif args.strategy == "projector-lora":
        processor, model = load_processor_and_model_for_inference(
            model_name=args.model_name,
            training_strategy=args.strategy,
            adapter_path=str(checkpoint_path),
        )
        model_path_for_payload = str(checkpoint_path)
    else:
        processor, model = load_processor_and_model_for_inference(
            model_name=str(checkpoint_path),
            training_strategy=args.strategy,
            adapter_path=None,
        )
        model_path_for_payload = str(checkpoint_path)

    dataset = MemeCaptionDataset(
        json_path=eval_json,
        setting_name=args.setting,
        allow_download=args.allow_download,
    )

    predictions: List[str] = []
    references: List[str] = []
    post_ids: List[str] = []

    total = len(dataset)
    if args.max_samples is not None:
        total = min(total, args.max_samples)

    for idx in tqdm(range(total), desc="Evaluating"):
        item = dataset[idx]
        pred = generate_one(
            processor=processor,
            model=model,
            image_path=item["image_path"],
            prompt=item["prompt"],
            max_new_tokens=args.max_new_tokens,
        )

        predictions.append(pred)
        references.append(item["target"])
        post_ids.append(item["post_id"])

    metrics = compute_generation_metrics(
        predictions=predictions,
        references=references,
        bert_lang=args.bert_lang,
    )

    payload: Dict[str, Any] = {
        "model_path": model_path_for_payload,
        "base_model_name": args.model_name,
        "strategy": args.strategy,
        "eval_json": eval_json,
        "setting": args.setting,
        "metrics": metrics,
        "predictions": [
            {
                "post_id": post_id,
                "prediction": prediction,
                "reference": reference,
            }
            for post_id, prediction, reference in zip(post_ids, predictions, references)
        ],
    }

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Evaluation finished.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
