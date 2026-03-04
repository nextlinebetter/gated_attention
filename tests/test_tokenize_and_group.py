import pytest
from train_owt_gpt2 import get_tokenizer, _tokenize_function, _group_texts

def test_tokenize_function_basic():
    tok = get_tokenizer()
    examples = {"text": ["hello world", "", None, "goodbye"]}
    out = _tokenize_function(examples, tokenizer=tok, text_field="text")

    assert "input_ids" in out
    assert isinstance(out["input_ids"], list)
    # should tokenize only non-empty strings -> 2 items
    assert len(out["input_ids"]) == 2
    assert all(isinstance(x, list) for x in out["input_ids"])
    assert all(len(x) > 0 for x in out["input_ids"])

def test_group_texts_exact_multiple():
    # 2 sequences total 8 tokens -> block_size=4 => 2 blocks
    examples = {"input_ids": [[1,2,3,4], [5,6,7,8]]}
    out = _group_texts(examples, block_size=4)

    assert out["input_ids"] == [[1,2,3,4],[5,6,7,8]]
    assert out["labels"] == out["input_ids"]

def test_group_texts_drops_remainder():
    # total 10 tokens, block_size=4 => keep 8 drop 2
    examples = {"input_ids": [[1,2,3,4,5], [6,7,8,9,10]]}
    out = _group_texts(examples, block_size=4)

    assert len(out["input_ids"]) == 2
    assert out["input_ids"][0] == [1,2,3,4]
    assert out["input_ids"][1] == [5,6,7,8]
    assert out["labels"] == out["input_ids"]

def test_group_texts_empty_when_too_short():
    examples = {"input_ids": [[1,2,3]]}
    out = _group_texts(examples, block_size=8)
    assert out["input_ids"] == []
    assert out["labels"] == []