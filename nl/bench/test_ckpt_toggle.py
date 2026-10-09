"""CPU-only unit check for the HF packed path's helpers in tuning_nl.py (2026-10-09 refactor).

Builds a 2-layer random-init Qwen3 model on CPU with attn_implementation='eager' and checks that
  * every decoder layer is a transformers GradientCheckpointingLayer (what per-layer toggling relies on);
  * Trainer-style gradient_checkpointing_enable sets the flag on every layer;
  * set_layer_checkpointing(layers, use_ckpt=True, ckpt_every=2) leaves layers[0] True and layers[1] False, and a
    forward/backward runs with that mixed setting;
  * eval_attn_sdpa switches the model to sdpa for the duration of an eval and restores the training implementation.
Run: /home/huan2073/.conda/envs/search/bin/python bench/test_ckpt_toggle.py   (no GPU, no flash-attn needed)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
os.environ.setdefault("NL_PACKING", "hf")

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_layers import GradientCheckpointingLayer

from tuning_nl import set_layer_checkpointing, resolve_model_parts, eval_attn_sdpa


def main():
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
        attn_implementation="eager",
    )
    model = Qwen3ForCausalLM(cfg)  # random init, CPU, fp32
    assert model.config._attn_implementation == "eager", model.config._attn_implementation

    parts = resolve_model_parts(model)
    layers = parts["inner"].layers
    assert parts["arch"] == "qwen" and parts["inner"] is model.model and parts["lm_head"] is model.lm_head
    assert len(layers) == 2
    assert all(isinstance(l, GradientCheckpointingLayer) for l in layers), [type(l) for l in layers]
    assert not any(l.gradient_checkpointing for l in layers), "flag must start False"

    # What Trainer does at train start (trainer.py:2447): flag True on every layer, use_reentrant=False.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert all(l.gradient_checkpointing for l in layers), "gradient_checkpointing_enable must set every layer"

    # The per-micro-batch toggle used by PackedSequenceTrainer._hf_forward_body.
    set_layer_checkpointing(layers, use_ckpt=True, ckpt_every=2)
    assert layers[0].gradient_checkpointing is True, layers[0].gradient_checkpointing
    assert layers[1].gradient_checkpointing is False, layers[1].gradient_checkpointing

    model.train()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    pos = torch.arange(12).unsqueeze(0)
    out = model.model(input_ids=ids, position_ids=pos, attention_mask=None, use_cache=False)
    assert out.past_key_values is None
    h = out.last_hidden_state[0]
    assert h.shape == (12, cfg.hidden_size), h.shape
    loss = model.lm_head(h).float().logsumexp(-1).mean()
    loss.backward()
    assert model.model.layers[0].mlp.down_proj.weight.grad is not None
    assert model.model.layers[1].mlp.down_proj.weight.grad is not None

    set_layer_checkpointing(layers, use_ckpt=False, ckpt_every=2)
    assert not any(l.gradient_checkpointing for l in layers)
    set_layer_checkpointing(layers, use_ckpt=True, ckpt_every=1)
    assert all(l.gradient_checkpointing for l in layers)

    # Eval pinning: eager -> sdpa inside, eager restored after (same mechanism as flash_attention_* -> sdpa on GPU).
    with eval_attn_sdpa(model):
        assert model.config._attn_implementation == "sdpa", model.config._attn_implementation
        assert model.model.layers[0].self_attn.config._attn_implementation == "sdpa"
    assert model.config._attn_implementation == "eager", model.config._attn_implementation

    print("test_ckpt_toggle: OK (layers[0]=True, layers[1]=False at ckpt_every=2; eval_attn_sdpa restores eager)")


if __name__ == "__main__":
    main()
