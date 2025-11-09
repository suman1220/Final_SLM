from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
PROMPT_HEADER = (
    "### Instruction:\n"
    "You are an expert clinical assistant focused on diabetes and closely related metabolic care. "
    "Respond with evidence-based, empathetic guidance that aligns with current standards of practice. "
    "Use no more than 40 words unless additional detail is essential for patient safety. "
    "Choose the clearest structure for the user—bulleted list, numbered steps, compact markdown table, or short paragraph—based on the question intent. "
    "Avoid greetings, avoid chit-chat, and reference prior conversation context when it improves clarity. "
    "If the patient's question is unrelated to diabetes or human healthcare, respond exactly with "
    "'I'm your virtual healthcare professional, and I can only assist with diabetes and health-related questions.'\n\n"
)

DEFAULT_FEEDBACK_PATH = Path("response_feedback.json")
DEFAULT_OUTPUT_ROOT = Path(".")
DEFAULT_LOCK_PATH = Path("retraining.lock")
DEFAULT_BASE_MODEL_PATH = Path("C:/Users/suman/OneDrive/Desktop/Testing/gemma3_270m_base")

GOOD_VERDICT = "good"


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
@dataclass
class FeedbackExample:
    question: str
    response: str
    timestamp: Optional[float] = None


def load_feedback_examples(feedback_path: Path) -> list[FeedbackExample]:
    if not feedback_path.exists():
        raise FileNotFoundError(f"Feedback file '{feedback_path}' does not exist.")

    with feedback_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Feedback payload must be a dictionary with 'good'/'bad' keys.")

    good_entries = data.get(GOOD_VERDICT, [])
    examples: list[FeedbackExample] = []
    for item in good_entries:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        answer = item.get("response")
        if not question or not answer:
            continue
        examples.append(
            FeedbackExample(
                question=question.strip(),
                response=answer.strip(),
                timestamp=item.get("timestamp"),
            )
        )

    return examples


class FeedbackDataset(Dataset):
    """Torch dataset that masks prompt tokens so only the answer contributes to loss."""

    def __init__(
        self,
        examples: Sequence[FeedbackExample],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self._examples = list(examples)
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self._examples[idx]
        prompt_prefix = (
            f"{PROMPT_HEADER}"
            f"### Patient's Question:\n{example.question}\n\n"
            f"### Answer:\n"
        )
        answer_text = f"{example.response}{self._tokenizer.eos_token}"
        full_text = prompt_prefix + answer_text

        tokenized = self._tokenizer(
            full_text,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        prefix_ids = self._tokenizer(
            prompt_prefix,
            truncation=True,
            max_length=self._max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        labels = input_ids.clone()
        prefix_len = min(prefix_ids.shape[0], labels.shape[0])
        labels[:prefix_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SupervisedCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    def __call__(self, batch: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        padded_inputs = nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self._tokenizer.pad_token_id,
        )
        padded_masks = nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )
        padded_labels = nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        return {
            "input_ids": padded_inputs,
            "attention_mask": padded_masks,
            "labels": padded_labels,
        }


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
def create_dataloaders(
    dataset: FeedbackDataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    validation_split: float,
    seed: int,
) -> tuple[DataLoader, Optional[DataLoader]]:
    if len(dataset) == 0:
        raise ValueError("No training examples available. Collect at least one 'good' feedback entry.")

    generator = torch.Generator().manual_seed(seed)
    if len(dataset) < 4 or validation_split <= 0.0:
        train_dataset = dataset
        val_dataset = None
    else:
        val_size = max(1, int(len(dataset) * validation_split))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    collator = SupervisedCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )

    return train_loader, val_loader


def setup_model(
    base_model_path: Path,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
    )

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def evaluate(model: torch.nn.Module, dataloader: DataLoader) -> float:
    if dataloader is None:
        return math.nan

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            for key in batch:
                batch[key] = batch[key].to(model.device)
            outputs = model(**batch)
            loss = outputs.loss.detach().float().item()
            losses.append(loss)
    if not losses:
        return math.nan
    return float(sum(losses) / len(losses))


@dataclass
class TrainingMetadata:
    start_time: str
    end_time: str
    epochs: int
    train_examples: int
    val_examples: int
    final_train_loss: float
    final_val_loss: float
    model_path: str
    gguf_path: Optional[str]


def run_training(args: argparse.Namespace) -> TrainingMetadata:
    lock_path = Path(args.lock_path)
    preexisting_lock = lock_path.exists()
    lock_acquired = False
    try:
        if not preexisting_lock:
            lock_path.touch(exist_ok=True)
            lock_acquired = True

        feedback_examples = load_feedback_examples(Path(args.feedback_path))
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dataset = FeedbackDataset(
            feedback_examples,
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
        )
        train_loader, val_loader = create_dataloaders(
            dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            seed=args.seed,
        )

        model = setup_model(
            Path(args.base_model_path),
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )

        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
            num_training_steps=total_steps,
        )

        start_time = dt.datetime.utcnow()
        global_step = 0
        last_train_loss = 0.0

        for epoch in range(1, args.epochs + 1):
            model.train()
            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
            for batch in progress:
                for key in batch:
                    batch[key] = batch[key].to(model.device)
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                last_train_loss = loss.detach().float().item()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                progress.set_postfix(loss=f"{last_train_loss:.4f}")

        val_loss = evaluate(model, val_loader)

        merged_model = model.merge_and_unload()
        merged_model.eval()

        timestamp = dt.datetime.now()
        month_stamp = timestamp.strftime("%Y-%m")
        output_root = Path(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        hf_output_dir = output_root / f"gemma3-270m-finetuned-{month_stamp}"
        hf_output_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(hf_output_dir)
        tokenizer.save_pretrained(hf_output_dir)

        gguf_output_path: Optional[Path] = None
        if args.gguf_output_prefix:
            base_name = args.gguf_output_prefix.rstrip(".gguf")
            gguf_output_path = (Path(args.gguf_output_dir) / f"{base_name}-{month_stamp}.gguf").resolve()
            if args.gguf_convert_script:
                convert_cmd = [
                    sys.executable,
                    str(args.gguf_convert_script),
                    "--model",
                    str(hf_output_dir),
                    "--outfile",
                    str(gguf_output_path),
                    "--outtype",
                    args.gguf_outtype,
                ]
                subprocess.run(convert_cmd, check=True)

        metadata = TrainingMetadata(
            start_time=start_time.isoformat() + "Z",
            end_time=dt.datetime.utcnow().isoformat() + "Z",
            epochs=args.epochs,
            train_examples=len(train_loader.dataset),
            val_examples=0 if not val_loader else len(val_loader.dataset),
            final_train_loss=float(last_train_loss),
            final_val_loss=float(val_loss) if not math.isnan(val_loss) else float("nan"),
            model_path=str(hf_output_dir.resolve()),
            gguf_path=str(gguf_output_path) if gguf_output_path else None,
        )

        metadata_path = hf_output_dir / "training_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata.__dict__, handle, indent=2)

        return metadata
    finally:
        if lock_acquired and lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# SCHEDULING
# ---------------------------------------------------------------------------
def compute_next_run(now: dt.datetime, run_hour: int, run_minute: int) -> dt.datetime:
    end_of_month = calendar.monthrange(now.year, now.month)[1]
    candidate = dt.datetime(
        year=now.year,
        month=now.month,
        day=end_of_month,
        hour=run_hour,
        minute=run_minute,
        second=0,
        tzinfo=now.tzinfo,
    )
    if candidate <= now:
        year = now.year + (1 if now.month == 12 else 0)
        month = 1 if now.month == 12 else now.month + 1
        end_of_next = calendar.monthrange(year, month)[1]
        candidate = dt.datetime(
            year=year,
            month=month,
            day=end_of_next,
            hour=run_hour,
            minute=run_minute,
            second=0,
            tzinfo=now.tzinfo,
        )
    return candidate


def schedule_loop(args: argparse.Namespace) -> None:
    print("Starting monthly retraining scheduler. Press Ctrl+C to exit.")
    while True:
        now = dt.datetime.now()
        next_run = compute_next_run(now, args.schedule_hour, args.schedule_minute)
        sleep_seconds = (next_run - now).total_seconds()
        hours, remainder = divmod(max(0, sleep_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(
            f"Next run at {next_run.isoformat(sep=' ', timespec='minutes')} "
            f"in {int(hours)}h {int(minutes)}m {int(seconds)}s."
        )
        try:
            time.sleep(max(0, sleep_seconds))
            print("Launching scheduled retraining run...")
            metadata = run_training(args)
            print(f"Retraining completed. New model at: {metadata.model_path}")
            if metadata.gguf_path:
                print(f"   GGUF artifact: {metadata.gguf_path}")
        except KeyboardInterrupt:
            print("\nScheduler interrupted by user.")
            break
        except Exception as exc:
            print(f"Warning: Retraining failed: {exc}")
            time.sleep(60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly retraining pipeline for Gemma3 270M.")

    parser.add_argument(
        "--base-model-path",
        type=Path,
        default=DEFAULT_BASE_MODEL_PATH,
        help=(
            "Local HuggingFace-format directory of the base Gemma3 270M model. "
            "Update DEFAULT_BASE_MODEL_PATH in retraining_pipeline.py when the base model location changes."
        ),
    )
    parser.add_argument(
        "--feedback-path",
        type=Path,
        default=DEFAULT_FEEDBACK_PATH,
        help="Path to response_feedback.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where finetuned model folders will be saved.",
    )
    parser.add_argument(
        "--gguf-output-dir",
        type=Path,
        default=Path("."),
        help="Directory to write GGUF artifacts (defaults to current directory).",
    )
    parser.add_argument(
        "--gguf-output-prefix",
        type=str,
        default="merged-Q8_0 (1)",
        help="Base filename (without month suffix) for GGUF outputs. Leave empty to skip GGUF export.",
    )
    parser.add_argument(
        "--gguf-convert-script",
        type=Path,
        default=None,
        help="Path to llama.cpp's convert-hf-to-gguf.py script. If omitted, GGUF export is skipped.",
    )
    parser.add_argument(
        "--gguf-outtype",
        type=str,
        default="f16",
        help="GGUF tensor type (e.g., f16, q4_K_M).",
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=DEFAULT_LOCK_PATH,
        help="Path to the retraining lock file shared with the inference app.",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run a persistent scheduler that triggers retraining at the end of each month.",
    )
    parser.add_argument("--schedule-hour", type=int, default=2)
    parser.add_argument("--schedule-minute", type=int, default=0)

    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Execute a retraining run immediately.",
    )

    args = parser.parse_args(argv)

    if args.gguf_output_prefix == "":
        args.gguf_output_prefix = None

    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if args.schedule:
        schedule_loop(args)
        return

    if not args.run_now:
        print("Nothing to do. Use '--run-now' for an immediate run or '--schedule' for monthly automation.")
        return

    metadata = run_training(args)
    print(f"Retraining completed. New model at: {metadata.model_path}")
    if metadata.gguf_path:
        print(f"   GGUF artifact: {metadata.gguf_path}")


if __name__ == "__main__":
    main()

