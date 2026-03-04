import torch
from train_owt_gpt2 import ExpConfig, get_tokenizer, build_model

def test_build_model_shapes_forward():
    cfg = ExpConfig(
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        gradient_checkpointing=False,
        output_dir="./tmp_test_out",
    )
    tok = get_tokenizer()
    model = build_model(cfg, tok)

    # batch=2, seq=64
    input_ids = torch.randint(low=0, high=len(tok), size=(2, cfg.block_size))
    out = model(input_ids=input_ids, labels=input_ids)
    assert out.loss is not None
    assert out.logits.shape == (2, cfg.block_size, len(tok))

def test_build_model_sets_pad_token():
    cfg = ExpConfig(block_size=64, n_layer=2, n_head=2, n_embd=64, output_dir="./tmp_test_out")
    tok = get_tokenizer()
    model = build_model(cfg, tok)
    assert model.config.pad_token_id == tok.pad_token_id