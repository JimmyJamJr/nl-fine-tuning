"""Legacy hand-rolled packed varlen forward: the training path of tuning_nl.py before the 2026-10-09 refactor.

Selected with NL_PACKING=custom (the default, hf, is transformers' built-in packed flash-attention path in
tuning_nl.PackedSequenceTrainer._hf_forward_body). Kept only so the parity test can run OLD and NEW from one trainer
with identical dataset / loss / metric code around the swap; delete after sign-off.

Contents, moved verbatim from tuning_nl.py (line numbers of archive/tuning_nl_pre_hfpacked_20261009.py):
  * the FA3/FA2 kernel-selection shim (L48-85): NL_ATTN_KERNEL=auto|fa3|fa2, auto = FA3 only on Hopper (sm90);
  * the RoPE import loop (L1284-1296), resolved lazily on the first forward (see _resolve_apply_rope);
  * _forward_layer_varlen (L1357-1453) and _rotate_half (L1455-1459);
  * the full _get_model_parts (L1461-1513), here get_model_parts (the trainer caches the dict);
  * the manual layer loop with torch.utils.checkpoint (L1588-1631), here forward_body.
Two additions for the deterministic parity regime (both off unless the environment asks for them):
  * flash_attn_varlen_func is called with deterministic=(FLASH_ATTENTION_DETERMINISTIC == "1"), the same switch
    transformers reads for its own FA path (modeling_flash_attention_utils.py:514-517); flash_attn 2.8.3
    (flash_attn/flash_attn_interface.py, `deterministic=False`) and flash_attn_3 3.0.0b1 (flash_attn_interface.py,
    `deterministic=False`) both accept the kwarg;
  * NL_LEGACY_NEOX_RESIDUAL_ORDER=hf sums the GPT-NeoX parallel residual in HF's order
    (mlp_output + attn_output + hidden_states, modeling_gpt_neox.py:245) instead of residual + attn + mlp, the only
    arithmetic difference between this loop and GPTNeoXLayer, so Pythia can be compared bit-for-bit.
"""
import os

import torch
import torch.utils.checkpoint as checkpoint

# Attention kernel for the packed varlen forward (merged from tuning_nl_fa3.py, 2026-10-09).
# FA3 (flash_attn_interface, Hopper GPUs) is tried first, then FA2 (flash_attn) for GPUs such as
# A100. NL_ATTN_KERNEL=fa3|fa2 forces one; auto (default) prefers FA3. The Qwen paper runs used FA2
# and the Pythia runs FA3; set NL_ATTN_KERNEL=fa2 to reproduce the Qwen kernel exactly.
_ATTN_KERNEL_REQ = os.environ.get("NL_ATTN_KERNEL", "auto").lower()
FLASH_ATTN_AVAILABLE = False
FLASH_ATTN_VERSION = None
flash_attn_varlen_func = None
# FA3 is built around Hopper (sm90a: H100/H200). The `search` env's build (flash-attn-3 3.0.0b1) also ships
# sm_80 kernels, but every non-Hopper run so far used FA2 (and older FA3 builds fail on those GPUs at the
# first forward), so auto mode uses FA3 only on Hopper. NL_ATTN_KERNEL=fa3 forces it elsewhere.
try:
    _IS_HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 9
except Exception:
    _IS_HOPPER = False
if _ATTN_KERNEL_REQ == "fa3" or (_ATTN_KERNEL_REQ == "auto" and _IS_HOPPER):
    try:
        from flash_attn_interface import flash_attn_varlen_func as _fa3_varlen

        def flash_attn_varlen_func(*args, **kwargs):
            out = _fa3_varlen(*args, **kwargs)
            if isinstance(out, tuple):   # some FA3 builds also return the softmax LSE
                out = out[0]
            return out

        FLASH_ATTN_AVAILABLE, FLASH_ATTN_VERSION = True, 3
    except ImportError:
        if _ATTN_KERNEL_REQ == "fa3":
            raise
if not FLASH_ATTN_AVAILABLE and _ATTN_KERNEL_REQ in ("auto", "fa2"):
    try:
        from flash_attn import flash_attn_varlen_func
        FLASH_ATTN_AVAILABLE, FLASH_ATTN_VERSION = True, 2
    except ImportError:
        flash_attn_varlen_func = None
if FLASH_ATTN_AVAILABLE:
    print(f"[FA{FLASH_ATTN_VERSION}] Using flash_attn_{FLASH_ATTN_VERSION} varlen kernel")

# Parity-regime switches (see module docstring). Read once at import; every launcher exports them before torchrun.
FLASH_ATTENTION_DETERMINISTIC = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
NEOX_RESIDUAL_ORDER = os.environ.get("NL_LEGACY_NEOX_RESIDUAL_ORDER", "legacy").lower()
if NEOX_RESIDUAL_ORDER not in ("legacy", "hf"):
    raise ValueError(f"NL_LEGACY_NEOX_RESIDUAL_ORDER must be 'legacy' or 'hf', got {NEOX_RESIDUAL_ORDER!r}")

_APPLY_ROPE = None
_APPLY_ROPE_RESOLVED = False


def _resolve_apply_rope():
    """RoPE implementation lookup (tuning_nl.py L1284-1296, verbatim). Resolved lazily on the first forward, as the
    original did at train start inside its override of Trainer's optimizer/scheduler loading hook (override removed
    2026-10-09), so that --use_liger's module-level patch of modeling_qwen3.apply_rotary_pos_emb (applied in main()
    before the model loads) is the function picked up."""
    global _APPLY_ROPE, _APPLY_ROPE_RESOLVED
    if _APPLY_ROPE_RESOLVED:
        return _APPLY_ROPE
    # Import RoPE implementation (try multiple model backends)
    _APPLY_ROPE = None
    for rope_module in [
        'transformers.models.qwen3.modeling_qwen3',
        'transformers.models.gpt_neox.modeling_gpt_neox',
        'transformers.models.llama.modeling_llama',
    ]:
        try:
            mod = __import__(rope_module, fromlist=['apply_rotary_pos_emb'])
            _APPLY_ROPE = mod.apply_rotary_pos_emb
            break
        except (ImportError, AttributeError):
            continue
    _APPLY_ROPE_RESOLVED = True
    return _APPLY_ROPE


def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _forward_layer_varlen(layer, hidden_states, cu_seqlens, max_seqlen,
                          num_heads, num_kv_heads, head_dim, cos, sin,
                          arch='qwen', parallel_residual=False, rotary_ndims=None):
    """Forward one layer using flash_attn_varlen_func. Supports Qwen and GPT-NeoX."""

    seq_len = hidden_states.shape[0]

    residual = hidden_states
    ln1_out = layer.input_layernorm(hidden_states)

    # QKV projection
    if arch == 'gpt_neox':
        # GPT-NeoX uses interleaved QKV: [q1|k1|v1 | q2|k2|v2 | ...] per head
        qkv = layer.attention.query_key_value(ln1_out)  # [seq, 3 * H]
        qkv = qkv.view(seq_len, num_heads, 3 * head_dim)
        q = qkv[..., :head_dim]              # [seq, num_heads, head_dim]
        k = qkv[..., head_dim:2*head_dim]
        v = qkv[..., 2*head_dim:]
    else:
        q = layer.self_attn.q_proj(ln1_out)
        k = layer.self_attn.k_proj(ln1_out)
        v = layer.self_attn.v_proj(ln1_out)
        # Reshape: [seq_len, num_heads, head_dim]
        q = q.view(seq_len, num_heads, head_dim)
        k = k.view(seq_len, num_kv_heads, head_dim)
        v = v.view(seq_len, num_kv_heads, head_dim)

    # QK Normalization (Qwen3 specific)
    if arch == 'qwen':
        if hasattr(layer.self_attn, 'q_norm') and layer.self_attn.q_norm is not None:
            q = layer.self_attn.q_norm(q)
            k = layer.self_attn.k_norm(k)

    # Add batch dim and transpose for RoPE: [1, num_heads, seq_len, head_dim]
    q = q.unsqueeze(0).transpose(1, 2)
    k = k.unsqueeze(0).transpose(1, 2)

    # Handle partial RoPE (e.g. Pythia rotary_pct=0.25)
    partial_rope = rotary_ndims is not None and rotary_ndims < head_dim
    if partial_rope:
        q_rot, q_pass = q[..., :rotary_ndims], q[..., rotary_ndims:]
        k_rot, k_pass = k[..., :rotary_ndims], k[..., rotary_ndims:]
    else:
        q_rot, k_rot = q, k

    # Apply pre-computed RoPE cos/sin
    if _APPLY_ROPE is not None:
        q_rot, k_rot = _APPLY_ROPE(q_rot, k_rot, cos, sin)
    else:
        cos_u = cos.unsqueeze(1)
        sin_u = sin.unsqueeze(1)
        q_rot = (q_rot * cos_u) + (_rotate_half(q_rot) * sin_u)
        k_rot = (k_rot * cos_u) + (_rotate_half(k_rot) * sin_u)

    if partial_rope:
        q = torch.cat((q_rot, q_pass), dim=-1)
        k = torch.cat((k_rot, k_pass), dim=-1)
    else:
        q, k = q_rot, k_rot

    # Reshape for flash_attn_varlen: [seq_len, num_heads, head_dim]
    q = q.squeeze(0).transpose(0, 1).contiguous()
    k = k.squeeze(0).transpose(0, 1).contiguous()
    v = v.contiguous()

    # Flash attention. deterministic= is the parity-regime addition (kernel default False, as before).
    attn_output = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
        deterministic=FLASH_ATTENTION_DETERMINISTIC,
    )

    # Output projection
    attn_output = attn_output.reshape(seq_len, num_heads * head_dim)
    if arch == 'gpt_neox':
        attn_output = layer.attention.dense(attn_output)
    else:
        attn_output = layer.self_attn.o_proj(attn_output)

    # Residual + MLP
    if parallel_residual:
        # GPT-NeoX parallel: x = x + attn(ln1(x)) + mlp(ln2(x))
        ln2_out = layer.post_attention_layernorm(residual)
        mlp_output = layer.mlp(ln2_out)
        if NEOX_RESIDUAL_ORDER == "hf":
            hidden_states = mlp_output + attn_output + residual   # GPTNeoXLayer order (modeling_gpt_neox.py:245)
        else:
            hidden_states = residual + attn_output + mlp_output
    else:
        # Sequential (Qwen/Llama): x = x + attn(ln1(x)); x = x + mlp(ln2(x))
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

    return hidden_states


def get_model_parts(model):
    """Model internals for the legacy loop (tuning_nl.py L1461-1513; the trainer caches the result)."""
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    try:
        from peft import PeftModel
        if isinstance(unwrapped, PeftModel):
            unwrapped = unwrapped.base_model.model
    except ImportError:
        pass
    config = unwrapped.config

    # Auto-detect architecture
    if hasattr(unwrapped, 'gpt_neox'):
        # GPT-NeoX (Pythia)
        inner = unwrapped.gpt_neox
        arch = 'gpt_neox'
        embed = inner.embed_in
        norm = inner.final_layer_norm
        lm_head = unwrapped.embed_out
        parallel_residual = getattr(config, 'use_parallel_residual', True)
    else:
        # Qwen / Llama-style
        inner = unwrapped.model
        if not hasattr(inner, 'embed_tokens') and hasattr(inner, 'model'):
            inner = inner.model
        arch = 'qwen'
        embed = inner.embed_tokens
        norm = inner.norm
        lm_head = unwrapped.lm_head
        parallel_residual = False

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    rotary_pct = getattr(config, 'rotary_pct', 1.0)
    rotary_ndims = int(head_dim * rotary_pct) if rotary_pct < 1.0 else head_dim

    return {
        'arch': arch,
        'embed': embed,
        'layers': inner.layers,
        'norm': norm,
        'rotary_emb': inner.rotary_emb,
        'lm_head': lm_head,
        'num_heads': config.num_attention_heads,
        'num_kv_heads': getattr(config, "num_key_value_heads", config.num_attention_heads),
        'head_dim': head_dim,
        'rotary_ndims': rotary_ndims,
        'parallel_residual': parallel_residual,
    }


def forward_body(mp, input_ids, position_ids, cu_seqlens_list, max_seqlen_list, use_ckpt, ckpt_every):
    """The pre-refactor manual layer loop (tuning_nl.py L1588-1631). mp is get_model_parts(model); input_ids and
    position_ids are the [1, total_tokens] packed tensors; use_ckpt is `gradient_checkpointing and model.training`
    and a layer is recomputed when use_ckpt and (layer_index % ckpt_every == 0). Returns (h, lm_head) with h the
    [total_tokens, H] hidden states after the final norm."""
    _resolve_apply_rope()
    embed = mp['embed']
    layers = mp['layers']
    norm = mp['norm']
    rotary_emb = mp['rotary_emb']
    lm_head = mp['lm_head']
    num_heads = mp['num_heads']
    num_kv_heads = mp['num_kv_heads']
    head_dim = mp['head_dim']
    arch = mp['arch']
    parallel_residual = mp['parallel_residual']
    rotary_ndims = mp['rotary_ndims']
    device = input_ids.device

    # Single row, no padding — directly use the tensors
    flat_ids = input_ids[0]           # [total_tokens]
    all_pos = position_ids[0:1]       # [1, total_tokens]
    merged_cu = torch.tensor(cu_seqlens_list[0], dtype=torch.int32, device=device)
    global_max_seqlen = max_seqlen_list[0]

    # Custom layer-by-layer forward with flash_attn_varlen_func
    # Embed tokens
    h = embed(flat_ids)  # [total_tokens, H]

    # Compute RoPE cos/sin once (same for all layers)
    cos, sin = rotary_emb(h, all_pos)

    for _li, layer in enumerate(layers):
        if use_ckpt and (_li % ckpt_every == 0):
            h = checkpoint.checkpoint(
                _forward_layer_varlen,
                layer, h, merged_cu, global_max_seqlen,
                num_heads, num_kv_heads, head_dim,
                cos, sin, arch, parallel_residual, rotary_ndims,
                use_reentrant=False,
            )
        else:
            h = _forward_layer_varlen(
                layer, h, merged_cu, global_max_seqlen,
                num_heads, num_kv_heads, head_dim,
                cos, sin, arch, parallel_residual, rotary_ndims,
            )

    h = norm(h)          # [total_tokens, H]
    return h, lm_head
