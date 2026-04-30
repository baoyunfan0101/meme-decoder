# src/model_utils.py

from __future__ import annotations
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# Common projector / merger module name patterns used by VLMs.
PROJECTOR_NAME_KEYWORDS = (
    "multi_modal_projector",
    "multimodal_projector",
    "visual_projector",
    "vision_projector",
    "mm_projector",
    "merger",
    "visual_merger",
    "vision_merger",
    "projector",
)

TRAINING_STRATEGIES = ("zero-shot", "projector-only", "projector-lora")

DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def freeze_all_parameters(model) -> None:
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_projector_only(model) -> List[str]:
    matched_module_names: List[str] = []

    for module_name, module in model.named_modules():
        lowered = module_name.lower()
        if any(keyword in lowered for keyword in PROJECTOR_NAME_KEYWORDS):
            has_param = False
            for param in module.parameters(recurse=False):
                param.requires_grad = True
                has_param = True
            if has_param:
                matched_module_names.append(module_name)

    if not matched_module_names:
        matched_param_names = []
        for param_name, param in model.named_parameters():
            lowered = param_name.lower()
            if any(keyword in lowered for keyword in PROJECTOR_NAME_KEYWORDS):
                param.requires_grad = True
                matched_param_names.append(param_name)

        if matched_param_names:
            dedup_prefixes = sorted({name.rsplit(".", 1)[0] if "." in name else name for name in matched_param_names})
            return dedup_prefixes

        raise ValueError(
            "Could not find any projector-like parameters to unfreeze. "
            "Please inspect model.named_parameters() and adjust PROJECTOR_NAME_KEYWORDS."
        )

    return sorted(set(matched_module_names))


def apply_lora_adapters(
    model,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES,
    modules_to_save: List[str] | None = None,
):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "projector-lora strategy requires the peft package. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
        modules_to_save=modules_to_save,
    )
    return get_peft_model(model, config)


def module_suffixes(module_names: List[str]) -> List[str]:
    suffixes = set()
    for name in module_names:
        if name:
            suffixes.add(name.split(".")[-1])
    return sorted(suffixes)


def load_peft_adapters(model, adapter_path: str):
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            "Loading a LoRA checkpoint requires the peft package. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    return PeftModel.from_pretrained(model, adapter_path)


def get_parameter_summary(model) -> Dict[str, Any]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    trainable_names = [
        name for name, param in model.named_parameters()
        if param.requires_grad
    ]

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": (trainable_params / total_params) if total_params > 0 else 0.0,
        "trainable_parameter_names": trainable_names,
    }


def get_model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def build_processor_kwargs(min_pixels: int | None = None, max_pixels: int | None = None) -> Dict[str, int]:
    kwargs: Dict[str, int] = {}
    if min_pixels is not None:
        kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        kwargs["max_pixels"] = max_pixels
    return kwargs


def build_model_kwargs(load_in_4bit: bool = False) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "torch_dtype": "auto",
        "device_map": "auto",
    }

    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    return kwargs


def load_processor_and_model(
    model_name: str = DEFAULT_MODEL_NAME,
    training_strategy: str = "projector-only",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    load_in_4bit: bool = False,
):
    if training_strategy not in TRAINING_STRATEGIES:
        raise ValueError(f"Unknown training strategy: {training_strategy}. Valid: {TRAINING_STRATEGIES}")

    processor = AutoProcessor.from_pretrained(
        model_name,
        **build_processor_kwargs(min_pixels=min_pixels, max_pixels=max_pixels),
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        **build_model_kwargs(load_in_4bit=load_in_4bit),
    )

    freeze_all_parameters(model)

    unfrozen_modules: List[str] = []
    if training_strategy in {"projector-only", "projector-lora"}:
        unfrozen_modules = unfreeze_projector_only(model)

    if training_strategy == "projector-lora":
        model = apply_lora_adapters(
            model=model,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            modules_to_save=module_suffixes(unfrozen_modules),
        )

    summary = get_parameter_summary(model)
    print(f"Training strategy: {training_strategy}")
    if max_pixels is not None:
        print(f"Processor max_pixels: {max_pixels}")
    if load_in_4bit:
        print("Model loading: 4-bit quantized")
    if unfrozen_modules:
        print("Unfrozen projector-related modules:")
        for name in unfrozen_modules:
            print(f"  - {name}")
    else:
        print("No base model parameters unfrozen.")
    print(f"Trainable params: {summary['trainable_params']:,} / {summary['total_params']:,} ({summary['trainable_ratio']:.6f})")

    return processor, model


def load_processor_and_model_for_inference(
    model_name: str = DEFAULT_MODEL_NAME,
    training_strategy: str = "zero-shot",
    adapter_path: str | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    load_in_4bit: bool = False,
):
    processor_source = adapter_path if adapter_path is not None else model_name
    processor = AutoProcessor.from_pretrained(
        processor_source,
        **build_processor_kwargs(min_pixels=min_pixels, max_pixels=max_pixels),
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        **build_model_kwargs(load_in_4bit=load_in_4bit),
    )

    if adapter_path is not None:
        model = load_peft_adapters(model, adapter_path)

    freeze_all_parameters(model)
    model.eval()

    print(f"Inference strategy: {training_strategy}")
    if max_pixels is not None:
        print(f"Processor max_pixels: {max_pixels}")
    if load_in_4bit:
        print("Model loading: 4-bit quantized")
    if adapter_path is not None:
        print(f"Loaded adapter checkpoint: {adapter_path}")

    return processor, model


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def build_messages(image: Image.Image, prompt_text: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def build_prompt_from_item(processor, item: Dict[str, Any]) -> Tuple[Image.Image, str]:
    image = load_image(item["image_path"])
    messages = build_messages(image=image, prompt_text=item["prompt"])
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return image, prompt_text


def _build_labels(
    full_input_ids: torch.Tensor,
    full_attention_mask: torch.Tensor,
    prompt_only_attention_mask: torch.Tensor,
) -> torch.Tensor:
    labels = full_input_ids.clone()
    labels[full_attention_mask == 0] = -100

    prompt_lengths = prompt_only_attention_mask.sum(dim=1)

    for i in range(labels.size(0)):
        prompt_len = int(prompt_lengths[i].item())
        labels[i, :prompt_len] = -100

    return labels


def collate_fn(processor, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images: List[Image.Image] = []
    prompt_texts: List[str] = []
    target_texts: List[str] = []
    full_texts: List[str] = []
    meta: List[Dict[str, Any]] = []

    for item in batch:
        image, prompt_text = build_prompt_from_item(processor, item)
        target_text = str(item["target"]).strip()

        images.append(image)
        prompt_texts.append(prompt_text)
        target_texts.append(target_text)
        full_texts.append(prompt_text + target_text)
        meta.append(item)

    full_inputs = processor(
        text=full_texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    prompt_only_inputs = processor(
        text=prompt_texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    labels = _build_labels(
        full_input_ids=full_inputs["input_ids"],
        full_attention_mask=full_inputs["attention_mask"],
        prompt_only_attention_mask=prompt_only_inputs["attention_mask"],
    )

    batch_inputs = dict(full_inputs)
    batch_inputs["labels"] = labels
    batch_inputs["meta"] = meta
    batch_inputs["prompt_texts"] = prompt_texts
    batch_inputs["target_texts"] = target_texts

    return batch_inputs


def move_batch_to_device(batch: Dict[str, Any], device: torch.device | str) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def forward_step(model, batch: Dict[str, Any]):
    model_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
    }

    if "pixel_values" in batch:
        model_inputs["pixel_values"] = batch["pixel_values"]

    if "image_grid_thw" in batch:
        model_inputs["image_grid_thw"] = batch["image_grid_thw"]

    return model(**model_inputs)


@torch.no_grad()
def generate_one(
    processor,
    model,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 64,
) -> str:
    image = load_image(image_path)
    messages = build_messages(image=image, prompt_text=prompt)

    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = move_batch_to_device(inputs, get_model_device(model))

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    prompt_input_len = inputs["input_ids"].shape[1]
    generated_target_ids = generated_ids[:, prompt_input_len:]

    output_text = processor.batch_decode(
        generated_target_ids,
        skip_special_tokens=True,
    )[0]

    return output_text.strip()


@torch.no_grad()
def generate_batch(
    processor,
    model,
    items: List[Dict[str, Any]],
    max_new_tokens: int = 64,
) -> List[str]:
    predictions: List[str] = []

    for item in items:
        pred = generate_one(
            processor=processor,
            model=model,
            image_path=item["image_path"],
            prompt=item["prompt"],
            max_new_tokens=max_new_tokens,
        )
        predictions.append(pred)

    return predictions
