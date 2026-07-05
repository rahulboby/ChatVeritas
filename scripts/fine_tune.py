"""QLoRA fine-tuning entry point for ChatVeritas.

The script trains on the conversational JSONL produced by
``scripts/prepare_finetune_dataset.py``. Checkpoints are kept separately from
the final PEFT adapter so ``app.py`` only ever loads a complete adapter.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.config_loader import load_config


def project_path(value: str | Path) -> Path:
    """Resolve a configured path relative to the ChatVeritas project root."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    """Read and validate ChatVeritas conversational JSONL records."""

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {dataset_path}. "
            "Run scripts/prepare_finetune_dataset.py first."
        )

    records: list[dict[str, Any]] = []

    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line_number, raw_line in enumerate(dataset_file, start=1):
            if not raw_line.strip():
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {dataset_path}: {exc}"
                ) from exc

            messages = record.get("messages") if isinstance(record, dict) else None

            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(
                    f"Line {line_number} must contain at least two messages."
                )

            normalized_messages = []
            for message_number, message in enumerate(messages, start=1):
                if not isinstance(message, dict):
                    raise ValueError(
                        f"Line {line_number}, message {message_number} must be an object."
                    )

                role = message.get("role")
                content = message.get("content")

                if role not in {"system", "user", "assistant"}:
                    raise ValueError(
                        f"Line {line_number}, message {message_number} has invalid role: {role!r}."
                    )

                if not isinstance(content, str) or not content.strip():
                    raise ValueError(
                        f"Line {line_number}, message {message_number} has empty content."
                    )

                normalized_messages.append({"role": role, "content": content.strip()})

            if not any(message["role"] == "user" for message in normalized_messages):
                raise ValueError(f"Line {line_number} has no user message.")

            if normalized_messages[-1]["role"] != "assistant":
                raise ValueError(
                    f"Line {line_number} must end with an assistant response."
                )

            source = str(record.get("source_file", "unknown"))
            chunk_id = str(record.get("chunk_id", line_number))
            records.append(
                {
                    "messages": normalized_messages,
                    "group": f"{source}:{chunk_id}",
                }
            )

    if not records:
        raise ValueError(f"Training dataset is empty: {dataset_path}")

    return records


def split_by_chunk(
    records: list[dict[str, Any]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by source chunk so paraphrases of one answer cannot leak across sets."""

    if not 0 <= validation_fraction < 1:
        raise ValueError("training.validation_split must be between 0 and 1.")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["group"]].append(record)

    group_names = list(groups)
    if validation_fraction == 0 or len(group_names) < 2:
        return records, []

    random.Random(seed).shuffle(group_names)
    validation_groups = max(1, round(len(group_names) * validation_fraction))
    validation_names = set(group_names[:validation_groups])

    train_records = []
    validation_records = []
    for group_name, group_records in groups.items():
        destination = validation_records if group_name in validation_names else train_records
        destination.extend(group_records)

    if not train_records:
        raise ValueError("The validation split left no training samples.")

    return train_records, validation_records


def validate_config(config: dict[str, Any]) -> None:
    """Fail early when required training settings are absent or invalid."""

    required_sections = {"model", "paths", "training", "lora"}
    missing = sorted(required_sections - config.keys())
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    if not config["model"].get("base_model"):
        raise ValueError("model.base_model must be configured.")
    if not config["model"].get("adapter_path"):
        raise ValueError("model.adapter_path must be configured.")
    if not config["paths"].get("processed_data"):
        raise ValueError("paths.processed_data must be configured.")
    if not config["paths"].get("training_checkpoints"):
        raise ValueError("paths.training_checkpoints must be configured.")

    positive_training_values = (
        "epochs",
        "batch_size",
        "gradient_accumulation",
        "learning_rate",
        "max_sequence_length",
    )
    for key in positive_training_values:
        if config["training"].get(key, 0) <= 0:
            raise ValueError(f"training.{key} must be greater than zero.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ChatVeritas with QLoRA.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config and JSONL data without loading the model.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        metavar="CHECKPOINT",
        help="Resume from CHECKPOINT, or from the newest checkpoint when omitted.",
    )
    return parser.parse_args()


def load_training_dependencies():
    """Import the GPU training stack only after cheap validation succeeds."""

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers.trainer_utils import get_last_checkpoint
        from trl import SFTConfig, SFTTrainer
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The fine-tuning stack could not be imported. Install requirements.txt "
            "in a clean virtual environment. If the error mentions torchaudio, make "
            "sure torch and torchaudio use the same version and CUDA build."
        ) from exc

    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "get_last_checkpoint": get_last_checkpoint,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def main() -> None:
    args = parse_args()
    config = load_config()
    validate_config(config)

    dataset_path = project_path(config["paths"]["processed_data"])
    adapter_path = project_path(config["model"]["adapter_path"])
    checkpoint_path = project_path(config["paths"]["training_checkpoints"])

    records = validate_dataset(dataset_path)
    seed = int(config["training"].get("seed", 42))
    train_records, validation_records = split_by_chunk(
        records,
        float(config["training"].get("validation_split", 0.05)),
        seed,
    )

    print(f"Validated samples : {len(records)}")
    print(f"Training samples  : {len(train_records)}")
    print(f"Validation samples: {len(validation_records)}")
    print(f"Adapter output    : {adapter_path}")
    print(f"Checkpoints       : {checkpoint_path}")

    if args.validate_only:
        print("Validation complete; model loading was skipped.")
        return

    dependencies = load_training_dependencies()
    torch = dependencies["torch"]

    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA training requires a CUDA GPU. Use --validate-only to check the dataset."
        )

    Dataset = dependencies["Dataset"]
    AutoTokenizer = dependencies["AutoTokenizer"]
    AutoModelForCausalLM = dependencies["AutoModelForCausalLM"]
    BitsAndBytesConfig = dependencies["BitsAndBytesConfig"]
    LoraConfig = dependencies["LoraConfig"]
    SFTConfig = dependencies["SFTConfig"]
    SFTTrainer = dependencies["SFTTrainer"]

    model_name = config["model"]["base_model"]
    training = config["training"]
    lora = config["lora"]
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading 4-bit base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map={"": torch.cuda.current_device()},
        dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = dependencies["prepare_model_for_kbit_training"](
        model,
        use_gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
    )

    lora_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias=lora.get("bias", "none"),
        task_type="CAUSAL_LM",
        target_modules=lora["target_modules"],
    )

    train_dataset = Dataset.from_list(
        [{"messages": record["messages"]} for record in train_records]
    )
    eval_dataset = None
    if validation_records:
        eval_dataset = Dataset.from_list(
            [{"messages": record["messages"]} for record in validation_records]
        )

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    training_args = SFTConfig(
        output_dir=str(checkpoint_path),
        num_train_epochs=float(training["epochs"]),
        per_device_train_batch_size=int(training["batch_size"]),
        per_device_eval_batch_size=int(training.get("eval_batch_size", 1)),
        gradient_accumulation_steps=int(training["gradient_accumulation"]),
        learning_rate=float(training["learning_rate"]),
        lr_scheduler_type=training.get("lr_scheduler", "cosine"),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        logging_steps=int(training.get("logging_steps", 5)),
        save_strategy="steps",
        save_steps=int(training.get("save_steps", 50)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=int(training.get("eval_steps", 50)) if eval_dataset is not None else None,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=training.get("optimizer", "paged_adamw_8bit"),
        max_length=int(training["max_sequence_length"]),
        packing=bool(training.get("packing", False)),
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.model.print_trainable_parameters()

    resume_checkpoint = None
    if args.resume:
        if args.resume == "auto":
            resume_checkpoint = dependencies["get_last_checkpoint"](str(checkpoint_path))
            if resume_checkpoint is None:
                raise FileNotFoundError(f"No checkpoint found in: {checkpoint_path}")
        else:
            resume_checkpoint = str(project_path(args.resume))
            if not Path(resume_checkpoint).is_dir():
                raise FileNotFoundError(f"Checkpoint not found: {resume_checkpoint}")
        print(f"Resuming from     : {resume_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    adapter_path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_path, safe_serialization=True)
    tokenizer.save_pretrained(adapter_path)

    summary = {
        "base_model": model_name,
        "dataset": str(dataset_path),
        "training_samples": len(train_records),
        "validation_samples": len(validation_records),
        "adapter_path": str(adapter_path),
    }
    (adapter_path / "chatveritas_training.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Training complete. Adapter and tokenizer saved to: {adapter_path}")


if __name__ == "__main__":
    main()
