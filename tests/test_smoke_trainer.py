import os
import pytest
import torch
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from train_owt_gpt2 import ExpConfig, get_tokenizer, build_model

def test_trainer_smoke(tmp_path):
    tok = get_tokenizer()

    # tiny synthetic tokenized dataset
    block_size = 32
    def mk(n):
        return {"input_ids": [list(range(block_size)) for _ in range(n)],
                "labels": [list(range(block_size)) for _ in range(n)]}
    ds = DatasetDict({
        "train": Dataset.from_dict(mk(8)),
        "validation": Dataset.from_dict(mk(4)),
    })

    cfg = ExpConfig(
        output_dir=str(tmp_path / "out"),
        block_size=block_size,
        n_layer=2, n_head=2, n_embd=64,
        bf16=False,
        gradient_checkpointing=False,
        max_train_steps=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=1,
        eval_steps=2,
        save_steps=10,
    )
    model = build_model(cfg, tok)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        max_steps=cfg.max_train_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=2,
        report_to=["none"],
        save_strategy="no",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        processing_class=tok,
    )

    trainer.train()
    metrics = trainer.evaluate()
    assert "eval_loss" in metrics