"""CPU-only unit check for the sparse output head in tuning_nl.py (NL_HEAD=sparse, 2026-10-09).

Builds a tiny random lm_head (nn.Linear(16, 50)) and random shift_h [200, 16] in fp32 with labels at ~5% of the rows
(-100 elsewhere) and checks that _head_ce_and_preds in mode='sparse' matches mode='full' (the pre-sparse-head
chunked loop, verbatim):
  (a) the per-row CE at the valid rows agrees to 1e-5 (and so does the summed loss);
  (b) the argmax at the valid rows is identical, preds is -1 at every unlabelled row in sparse mode;
  (c) with first_token_soft_weight 0.3 and fake valid_first_targets, the blended per-row CE (_blend_first_token_ce)
      agrees to 1e-5, and gradients through both paths agree;
  (d) head_rows is n_valid in sparse mode and Tm1 in full mode; chunking the sparse rows (chunk_size < n_valid) gives
      the same result; the pretrain-style case where every row is labelled also agrees.
Run: /home/huan2073/.conda/envs/search/bin/python bench/test_sparse_head.py   (no GPU, no flash-attn needed)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
os.environ.setdefault("NL_PACKING", "hf")

import torch
import torch.nn as nn

from tuning_nl import _head_ce_and_preds, _blend_first_token_ce, NL_HEAD

TOL = 1e-5


def make_case(seed, Tm1=200, H=16, V=50, p_valid=0.05, all_valid=False):
    g = torch.Generator().manual_seed(seed)
    lm_head = nn.Linear(H, V, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.randn(V, H, generator=g) * 0.3)
    shift_h = (torch.randn(Tm1, H, generator=g)).requires_grad_(True)
    labels = torch.randint(0, V, (Tm1,), generator=g)
    if not all_valid:
        keep = torch.rand(Tm1, generator=g) < p_valid
        labels = torch.where(keep, labels, torch.full_like(labels, -100))
        # Make sure at least a handful of rows are labelled (random draw could otherwise be too sparse).
        for r in (3, 57, 120, 199):
            labels[r] = (r * 7) % V
    valid_mask = labels != -100
    return lm_head, shift_h, labels, valid_mask


def run_head(mode, lm_head, shift_h, labels, valid_mask, chunk_size):
    # Fresh graph per call: detach + re-require grad so each path owns its own backward.
    h = shift_h.detach().clone().requires_grad_(True)
    ce, preds, head_rows = _head_ce_and_preds(lm_head, h, labels, valid_mask, mode=mode, chunk_size=chunk_size)
    return ce, preds, head_rows, h


def fake_first_rows(valid_mask, n_seq=4, V=50):
    """Pretend the batch holds n_seq sequences whose first answer rows are the first n_seq valid rows (first rows are
    always valid rows in compute_loss: first_valid = first_in_range & valid_mask[first_indices]). One sequence has an
    empty target list (skipped by the blend), one is marked invalid, the rest carry 2-3 fake targets each."""
    rows = valid_mask.nonzero(as_tuple=True)[0]
    assert rows.numel() >= n_seq + 1, rows.numel()
    first_indices = rows[:n_seq].clone()
    first_valid = torch.ones(n_seq, dtype=torch.bool)
    first_valid[2] = False                           # e.g. a pretrain sequence (is_search False)
    targets = [[1, 5, 9], [], [4, 4], [7, 12, 30]]   # seq 1: empty list -> skipped
    return first_indices, first_valid, targets


def check_case(seed, chunk_full, chunk_sparse, all_valid=False):
    lm_head, shift_h, labels, valid_mask = make_case(seed, all_valid=all_valid)
    Tm1 = shift_h.size(0)
    n_valid = int(valid_mask.sum())
    rows = valid_mask.nonzero(as_tuple=True)[0]

    ce_f, preds_f, hr_f, h_f = run_head("full", lm_head, shift_h, labels, valid_mask, chunk_full)
    ce_s, preds_s, hr_s, h_s = run_head("sparse", lm_head, shift_h, labels, valid_mask, chunk_sparse)

    # (a) per-row CE at valid rows (valid-row order) and the loss
    assert ce_f.shape == ce_s.shape == (n_valid,), (ce_f.shape, ce_s.shape, n_valid)
    assert torch.allclose(ce_f, ce_s, atol=TOL, rtol=0), (ce_f - ce_s).abs().max().item()
    assert abs((ce_f.sum() / n_valid).item() - (ce_s.sum() / n_valid).item()) < TOL
    # (b) argmax at valid rows identical; unlabelled rows -1 in sparse, a real argmax in full
    assert preds_f.shape == preds_s.shape == (Tm1,) and preds_s.dtype == torch.long
    assert torch.equal(preds_f[rows], preds_s[rows])
    assert bool((preds_s[~valid_mask] == -1).all()), "sparse preds must be -1 off the labelled rows"
    assert bool((preds_f >= 0).all())
    assert int((preds_s >= 0).sum()) == n_valid
    # (d) head_rows accounting
    assert hr_f == Tm1 and hr_s == n_valid, (hr_f, hr_s, Tm1, n_valid)

    # (c) soft first-token blend (w=0.3) on both paths, then the gradients through both
    w = 0.3
    first_indices, first_valid, targets = fake_first_rows(valid_mask)
    ce_f_b = ce_f   # the blend is in place, exactly as compute_loss uses it
    nb_f = _blend_first_token_ce(ce_f_b, lm_head, h_f, valid_mask, first_indices, first_valid, targets, w)
    nb_s = _blend_first_token_ce(ce_s, lm_head, h_s, valid_mask, first_indices, first_valid, targets, w)
    assert nb_f == nb_s == int(first_valid.sum()), (nb_f, nb_s)
    assert torch.allclose(ce_f_b, ce_s, atol=TOL, rtol=0), (ce_f_b - ce_s).abs().max().item()
    # The blended rows really changed (not a vacuous comparison), the skipped ones did not.
    ce_indices = torch.cumsum(valid_mask.int(), dim=0) - 1
    blended = [int(ce_indices[first_indices[k]]) for k in range(len(targets)) if first_valid[k] and targets[k]]
    assert blended, "test must blend at least one row"
    raw_s = _head_ce_and_preds(lm_head, h_s.detach(), labels, valid_mask, mode="sparse", chunk_size=chunk_sparse)[0]
    assert all(abs(ce_s[i].item() - raw_s[i].item()) > 1e-4 for i in blended), "blend had no effect"
    untouched = [i for i in range(n_valid) if i not in blended]
    assert torch.allclose(ce_s[untouched], raw_s[untouched], atol=TOL, rtol=0)

    loss_f = ce_f_b.sum() / n_valid
    loss_s = ce_s.sum() / n_valid
    assert abs(loss_f.item() - loss_s.item()) < TOL
    lm_head.weight.grad = None
    loss_f.backward()
    gw_f, gh_f = lm_head.weight.grad.clone(), h_f.grad.clone()
    lm_head.weight.grad = None
    loss_s.backward()
    gw_s, gh_s = lm_head.weight.grad.clone(), h_s.grad.clone()
    assert torch.allclose(gw_f, gw_s, atol=TOL, rtol=0), (gw_f - gw_s).abs().max().item()
    assert torch.allclose(gh_f, gh_s, atol=TOL, rtol=0), (gh_f - gh_s).abs().max().item()
    assert bool((gh_s[~valid_mask] == 0).all()), "unlabelled rows must get zero gradient from the head"
    return n_valid, Tm1, loss_s.item()


def main():
    torch.manual_seed(0)
    print(f"NL_HEAD (module default) = {NL_HEAD}")
    # 1) default chunking: one sparse call (n_valid << chunk) vs the 64-row chunked full loop
    n_valid, Tm1, loss = check_case(seed=0, chunk_full=64, chunk_sparse=4096)
    print(f"case 1: n_valid={n_valid}/{Tm1} rows, loss={loss:.6f}  OK")
    # 2) sparse rows chunked (chunk_size 3 < n_valid) vs a single full chunk
    n_valid, Tm1, loss = check_case(seed=1, chunk_full=4096, chunk_sparse=3)
    print(f"case 2: n_valid={n_valid}/{Tm1} rows, sparse chunk=3, loss={loss:.6f}  OK")
    # 3) pretrain-style: every row labelled (sparse rows == all rows, chunked both ways)
    n_valid, Tm1, loss = check_case(seed=2, chunk_full=32, chunk_sparse=32, all_valid=True)
    assert n_valid == Tm1
    print(f"case 3: all {Tm1} rows labelled, loss={loss:.6f}  OK")
    # 4) no labelled rows at all: sparse returns empty ce, -1 preds, 0 head rows (compute_loss never calls it then,
    #    but the helper must not crash on it)
    lm_head, shift_h, labels, _ = make_case(3)
    labels = torch.full_like(labels, -100)
    ce, preds, hr = _head_ce_and_preds(lm_head, shift_h, labels, labels != -100, mode="sparse", chunk_size=8)
    assert ce.numel() == 0 and hr == 0 and bool((preds == -1).all())
    print("case 4: no labelled rows  OK")
    try:
        _head_ce_and_preds(lm_head, shift_h, labels, labels != -100, mode="bogus", chunk_size=8)
        raise AssertionError("bogus mode must raise")
    except ValueError:
        pass
    print("ALL OK")


if __name__ == "__main__":
    main()
