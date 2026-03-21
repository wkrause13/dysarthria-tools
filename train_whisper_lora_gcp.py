#!/usr/bin/env python3
"""Train Whisper Small with LoRA on a dysarthria dataset and export a merged model.

This script is designed for a single GCP GPU VM. It downloads the dataset from
Google Drive / Google Sheets, fine-tunes `openai/whisper-small` with PEFT LoRA,
merges the adapter into the base model, and optionally syncs all artifacts to GCS.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
import soundfile as sf
import librosa
from datasets import Dataset, DatasetDict
from jiwer import wer as jiwer_wer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def run(cmd: List[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_audio_path(audio_root: Path, file_name: str) -> Path:
    candidate = Path(file_name)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    direct = audio_root / file_name
    if direct.exists():
        return direct

    matches = list(audio_root.rglob(file_name))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Audio file listed in metadata.csv was not found: {file_name}")


def load_dataset_dict(audio_root: Path, metadata_path: Path, seed: int) -> DatasetDict:
    df = pd.read_csv(metadata_path)
    required = {"file_name", "transcription"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {sorted(missing)}")

    df["file_name"] = df["file_name"].astype(str).str.strip()
    df["transcription"] = df["transcription"].astype(str).map(lambda x: " ".join(x.strip().split()))
    df["audio"] = df["file_name"].map(lambda name: str(resolve_audio_path(audio_root, name)))

    base = Dataset.from_pandas(
        df[["audio", "transcription"]].rename(columns={"audio": "audio_path", "transcription": "sentence"}),
        preserve_index=False,
    )
    eval_size = max(1, int(round(0.1 * len(base))))
    if eval_size >= len(base):
        eval_size = 1

    split = base.train_test_split(test_size=eval_size, seed=seed)
    return DatasetDict(train=split["train"], validation=split["test"])


def build_processor(model_id: str, language: str, task: str) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(model_id, language=language, task=task)


def prepare_example(batch: Dict[str, Any], processor: WhisperProcessor) -> Dict[str, Any]:
    audio_array, sampling_rate = sf.read(batch["audio_path"], dtype="float32", always_2d=False)
    if getattr(audio_array, "ndim", 1) > 1:
        audio_array = np.mean(audio_array, axis=1)
    if sampling_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000
    batch["input_features"] = processor.feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def make_compute_metrics(processor: WhisperProcessor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * jiwer_wer(label_str, pred_str)
        return {"wer": wer}

    return compute_metrics


def sync_to_gcs(local_dir: Path, gcs_uri: str) -> None:
    run(["gsutil", "-m", "rsync", "-r", str(local_dir), gcs_uri])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-root", default="/opt/whisper-lora/run")
    parser.add_argument("--model-id", default="openai/whisper-small")
    parser.add_argument("--language", default="english")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=int, default=30)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--generation-max-length", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gcs-output-uri", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    run_root = Path(args.run_root)
    adapter_dir = run_root / "lora_adapter"
    merged_dir = run_root / "merged_model"
    training_dir = run_root / "trainer_output"
    logs_dir = run_root / "logs"
    for path in (run_root, adapter_dir, merged_dir, training_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")
    raw_datasets = load_dataset_dict(data_dir, metadata_path, args.seed)
    processor = build_processor(args.model_id, args.language, args.task)

    vectorized = raw_datasets.map(
        lambda batch: prepare_example(batch, processor),
        remove_columns=raw_datasets["train"].column_names,
        num_proc=1,
    )

    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    model.generation_config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    compute_metrics = make_compute_metrics(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(training_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=2,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=vectorized["train"],
        eval_dataset=vectorized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_state()
    trainer.model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    base_model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = peft_model.merge_and_unload()
    merged_model.generation_config.language = args.language
    merged_model.generation_config.task = args.task
    merged_model.generation_config.forced_decoder_ids = None
    merged_model.config.use_cache = True
    merged_model.generation_config.use_cache = True
    merged_model.save_pretrained(str(merged_dir), safe_serialization=False)
    processor.save_pretrained(str(merged_dir))

    metrics = {
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "run_root": str(run_root),
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
        "examples": {
            "train": len(vectorized["train"]),
            "validation": len(vectorized["validation"]),
        },
    }
    with open(run_root / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if args.gcs_output_uri:
        sync_to_gcs(run_root, args.gcs_output_uri.rstrip("/"))

    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
