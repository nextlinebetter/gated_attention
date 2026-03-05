#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pretrain GPT-2 (small) on OpenWebText (baseline) with extensible structure for later experiments:
- gating variants
- attention sink metrics
- stability tests (high LR, etc.)
"""

import os
import json
import math
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import torch
from torch.utils.data import Dataset

from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import (
    GPT2Config,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    set_seed,
)


# --------------------------
# Config
# --------------------------
@dataclass
class ExpConfig:
    # Experiment identity
    exp_name: str = "baseline"
    model_variant: str = "baseline"  # placeholder: baseline | gated_g1 | topk | rmsnorm | ...

    # Data
    dataset_name: str = "openwebtext"
    dataset_config: Optional[str] = None
    text_field: str = "text"
    cache_dir: str = "./data_cache"
    tokenized_cache_dir: str = "./data_cache/owt_tokenized"
    overwrite_cache: bool = False

    # Tokenization / packing
    block_size: int = 1024
    num_proc: int = 8  # adjust based on CPU cores
    val_ratio: float = 0.01

    # Model (GPT-2 small)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0

    # Training
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    lr_scheduler_type: str = "cosine"

    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8

    max_train_steps: int = 20000
    logging_steps: int = 20
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3

    # Output
    output_dir: str = "./runs/baseline"
    resume_from_checkpoint: Optional[str] = None


# --------------------------
# Callbacks (extensible)
# --------------------------
class StatsCallback(TrainerCallback):
    """
    Minimal stats logger for extensibility.
    Later you can:
    - read model attentions and compute first-token attention ratio (attention sink)
    - track activation outliers, gate sparsity, etc.
    """
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.path = os.path.join(out_dir, "train_stats.jsonl")
        os.makedirs(out_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        rec = {"step": state.global_step, **logs, "time": time.time()}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


# --------------------------
# Data pipeline
# --------------------------
def get_tokenizer() -> GPT2TokenizerFast:
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    # GPT-2 has no pad token by default; set to eos for batching convenience
    tok.pad_token = tok.eos_token
    return tok


def _tokenize_function(examples, tokenizer: GPT2TokenizerFast, text_field: str):
    # remove None / empty
    texts = [t for t in examples[text_field] if isinstance(t, str) and len(t) > 0]
    return tokenizer(texts, return_attention_mask=False)


def _group_texts(examples, block_size: int):
    """
    Concatenate all tokens and split into fixed blocks.
    Standard LM pretraining packing.
    """
    # examples["input_ids"] is List[List[int]]
    concatenated = {}
    for k in examples.keys():
        concatenated[k] = sum(examples[k], [])

    total_len = len(concatenated["input_ids"])
    # drop the remainder
    total_len = (total_len // block_size) * block_size
    if total_len == 0:
        return {"input_ids": [], "labels": []}

    result = {
        "input_ids": [concatenated["input_ids"][i : i + block_size] for i in range(0, total_len, block_size)]
    }
    # labels = input_ids for causal LM
    result["labels"] = [x[:] for x in result["input_ids"]]
    return result


def build_or_load_tokenized_datasets(cfg: ExpConfig, tokenizer: GPT2TokenizerFast) -> DatasetDict:
    """
    Loads raw OpenWebText, tokenizes, packs into blocks, creates train/val split.
    Saves to disk for reuse across experiments (baseline/gated/ablation).
    """
    tokenized_path = os.path.abspath(cfg.tokenized_cache_dir)

    if os.path.exists(tokenized_path) and (not cfg.overwrite_cache):
        print(f"[Data] Loading tokenized dataset from disk: {tokenized_path}")
        return load_from_disk(tokenized_path)

    print(f"[Data] Loading raw dataset: {cfg.dataset_name}")
    raw = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        cache_dir=cfg.cache_dir,
    )

    # OpenWebText typically has only "train" split. We'll create a val split.
    if "validation" not in raw:
        raw = raw["train"].train_test_split(test_size=cfg.val_ratio, seed=cfg.seed)
        raw = DatasetDict({"train": raw["train"], "validation": raw["test"]})

    print("[Data] Tokenizing...")
    tokenized = raw.map(
        lambda x: _tokenize_function(x, tokenizer=tokenizer, text_field=cfg.text_field),
        batched=True,
        num_proc=cfg.num_proc,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing",
    )

    print("[Data] Grouping into fixed-length blocks...")
    lm_ds = tokenized.map(
        lambda x: _group_texts(x, block_size=cfg.block_size),
        batched=True,
        num_proc=cfg.num_proc,
        desc=f"Grouping texts (block_size={cfg.block_size})",
    )

    # Filter possible empty rows (can happen if examples are too short)
    def _non_empty(ex):
        return ex["input_ids"] is not None and len(ex["input_ids"]) == cfg.block_size

    lm_ds = lm_ds.filter(_non_empty, num_proc=cfg.num_proc, desc="Filtering empty blocks")

    os.makedirs(tokenized_path, exist_ok=True)
    print(f"[Data] Saving tokenized dataset to disk: {tokenized_path}")
    lm_ds.save_to_disk(tokenized_path)
    return lm_ds


# --------------------------
# Model factory (extensible)
# --------------------------
def build_model(cfg: ExpConfig, tokenizer: GPT2TokenizerFast) -> GPT2LMHeadModel:
    """
    For now returns baseline GPT-2 small initialized from scratch.
    Later you can branch by cfg.model_variant:
      - gated: use your modified modeling_gpt2 with gate
      - topk attention: custom attention module
      - rmsnorm at SDPA output: etc.
    """
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        n_positions=cfg.block_size,
        n_ctx=cfg.block_size,
        resid_pdrop=cfg.dropout,
        embd_pdrop=cfg.dropout,
        attn_pdrop=cfg.dropout,
        # Enable SDPA when supported by your transformers version:
        # (If unsupported in your local copy, you can remove this line.)
        attn_implementation="sdpa",
    )

    if cfg.model_variant != "baseline":
        print(f"[Model] WARNING: model_variant={cfg.model_variant} is not implemented yet. Using baseline.")

    model = GPT2LMHeadModel(config)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Ensure pad_token_id set
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return model


# --------------------------
# Utils
# --------------------------
def estimate_tokens_per_step(cfg: ExpConfig, world_size: int) -> int:
    # tokens/step = batch_per_device * seq_len * world_size * grad_accum
    return cfg.per_device_train_batch_size * cfg.block_size * world_size * cfg.gradient_accumulation_steps


def parse_args() -> ExpConfig:
    p = argparse.ArgumentParser()
    # core
    p.add_argument("--output_dir", type=str, default="./runs/baseline")
    p.add_argument("--exp_name", type=str, default="baseline")
    p.add_argument("--model_variant", type=str, default="baseline")
    # data
    p.add_argument("--cache_dir", type=str, default="./data_cache")
    p.add_argument("--tokenized_cache_dir", type=str, default="./data_cache/owt_tokenized")
    p.add_argument("--overwrite_cache", action="store_true")
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--val_ratio", type=float, default=0.01)
    # train
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", type=lambda x: str(x).lower() == "true", default=True)
    p.add_argument("--gradient_checkpointing", type=lambda x: str(x).lower() == "true", default=True)

    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--max_train_steps", type=int, default=20000)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)

    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)

    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = p.parse_args()
    cfg = ExpConfig(
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        model_variant=args.model_variant,
        cache_dir=args.cache_dir,
        tokenized_cache_dir=args.tokenized_cache_dir,
        overwrite_cache=args.overwrite_cache,
        block_size=args.block_size,
        num_proc=args.num_proc,
        val_ratio=args.val_ratio,
        seed=args.seed,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_train_steps=args.max_train_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    return cfg


def main():
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # record config
    with open(os.path.join(cfg.output_dir, "exp_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    set_seed(cfg.seed)

    # world size for tokens/step estimate
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print(f"[Env] WORLD_SIZE={world_size}")
    print(f"[Info] Estimated tokens/step = {estimate_tokens_per_step(cfg, world_size):,}")

    tokenizer = get_tokenizer()
    ds = build_or_load_tokenized_datasets(cfg, tokenizer)

    model = build_model(cfg, tokenizer)

    # Data collator: causal LM
    # We already provide labels, but collator can still pad (pad==eos).
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # TrainingArguments
    # Important: max_steps (not num_train_epochs) to be explicit / comparable across variants
    train_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=False,
        bf16=cfg.bf16,
        fp16=False,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_steps=cfg.max_train_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to=["none"],  # change to ["tensorboard"] if you want
        dataloader_num_workers=min(cfg.num_proc, 8),
        remove_unused_columns=False,  # keep extensible for custom fields later
    )

    callbacks = [StatsCallback(cfg.output_dir)]

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print("[Train] Starting...")
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    # Final eval + save
    print("[Eval] Final evaluation...")
    metrics = trainer.evaluate()
    # Convert eval_loss to perplexity
    if "eval_loss" in metrics:
        metrics["eval_ppl"] = float(math.exp(metrics["eval_loss"])) if metrics["eval_loss"] < 20 else float("inf")

    with open(os.path.join(cfg.output_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[Save] Saving final model + tokenizer...")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print("[Done]")


if __name__ == "__main__":
    main()