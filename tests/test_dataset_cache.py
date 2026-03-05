import os
import shutil
import pytest
from datasets import Dataset, DatasetDict

import train_owt_gpt2
from train_owt_gpt2 import ExpConfig, get_tokenizer, build_or_load_tokenized_datasets

def _fake_load_dataset(name, config=None, cache_dir=None):
    long_text = " ".join(["hello"] * 200)
    ds = Dataset.from_dict({"text": [long_text] * 20})
    return DatasetDict({"train": ds})

def test_build_tokenized_dataset_creates_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(train_owt_gpt2, "load_dataset", _fake_load_dataset)

    cfg = ExpConfig(
        tokenized_cache_dir=str(tmp_path / "tok"),
        cache_dir=str(tmp_path / "raw"),
        overwrite_cache=True,
        block_size=8,     # small for test
        num_proc=1,
        val_ratio=0.25,
        seed=123,
        output_dir=str(tmp_path / "out"),
    )
    tok = get_tokenizer()
    ds = build_or_load_tokenized_datasets(cfg, tok)

    assert "train" in ds and "validation" in ds
    assert len(ds["train"]) > 0
    assert len(ds["validation"]) > 0
    assert os.path.exists(cfg.tokenized_cache_dir)

def test_build_tokenized_dataset_reuses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(train_owt_gpt2, "load_dataset", _fake_load_dataset)

    cfg = ExpConfig(
        tokenized_cache_dir=str(tmp_path / "tok"),
        cache_dir=str(tmp_path / "raw"),
        overwrite_cache=True,
        block_size=8,
        num_proc=1,
        val_ratio=0.25,
        seed=123,
        output_dir=str(tmp_path / "out"),
    )
    tok = get_tokenizer()

    ds1 = build_or_load_tokenized_datasets(cfg, tok)
    # second time should NOT call load_dataset if overwrite_cache=False
    cfg.overwrite_cache = False
    ds2 = build_or_load_tokenized_datasets(cfg, tok)

    assert len(ds1["train"]) == len(ds2["train"])
    assert len(ds1["validation"]) == len(ds2["validation"])