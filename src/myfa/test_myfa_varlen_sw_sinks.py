"""从仓库根目录运行：PYTHONPATH=src python -m pytest src/myfa/test_myfa_varlen_sw_sinks.py -v。"""

import math

import pytest
import torch

from myfa.myfa_varlen_sw_sinks import myfa_varlen_sw_sinks


def ref_varlen_attn(
    q, k, v, cu_seqlens_q, cu_seqlens_k,
    sinks=None, is_causal=True, window_size=None,
):
    """Reference implementation for varlen GQA attention with SWA and sinks."""
    total_q, HQ, d = q.shape
    _, HK, _ = k.shape
    groups = HQ // HK
    scaling = 1.0 / math.sqrt(d)
    batch_size = len(cu_seqlens_q) - 1

    output = torch.zeros_like(q)

    for b in range(batch_size):
        q_start = int(cu_seqlens_q[b])
        q_end = int(cu_seqlens_q[b + 1])
        k_start = int(cu_seqlens_k[b])
        k_end = int(cu_seqlens_k[b + 1])

        q_len = q_end - q_start
        k_len = k_end - k_start
        if q_len == 0:
            continue

        q_seq = q[q_start:q_end]  # [q_len, HQ, d]
        k_seq = k[k_start:k_end]  # [k_len, HK, d]
        v_seq = v[k_start:k_end]  # [k_len, HK, d]

        # GQA reshape
        q_seq = q_seq.view(q_len, HK, groups, d)
        k_seq = k_seq.unsqueeze(2)  # [k_len, HK, 1, d]
        v_seq = v_seq.unsqueeze(2)  # [k_len, HK, 1, d]

        # [HK, groups, q_len, k_len]
        logits = torch.einsum("qhgd,khgd->hgqk", q_seq.float(), k_seq.float()) * scaling

        offset = k_len - q_len
        pos_keys = torch.arange(k_len, device=q.device)
        pos_queries = torch.arange(q_len, device=q.device) + offset

        mask = torch.zeros(q_len, k_len, device=q.device)
        if is_causal:
            causal_mask = pos_keys[None, :] > pos_queries[:, None]
            mask.masked_fill_(causal_mask, float("-inf"))
        if window_size is not None:
            too_old = pos_keys[None, :] < (pos_queries[:, None] - window_size + 1)
            mask.masked_fill_(too_old, float("-inf"))

        logits = logits + mask[None, None, :, :]

        if sinks is not None:
            sinks_expanded = sinks.view(HK, groups, 1, 1).float()
            logits_max = torch.max(logits, dim=-1, keepdim=True).values
            logits_or_sinks_max = torch.maximum(sinks_expanded, logits_max)
            sinks_exp = torch.exp(sinks_expanded - logits_or_sinks_max)
            unnormalized_scores = torch.exp(logits - logits_or_sinks_max)
            normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks_exp
            scores = unnormalized_scores / normalizer
        else:
            scores = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)

        out = torch.einsum("hgqk,khgd->qhgd", scores, v_seq.float())
        out = out.reshape(q_len, HQ, d).to(q.dtype)
        output[q_start:q_end] = out

    return output


CASES = [
    # causal, no SWA, no sinks
    dict(seqlens=[64], HQ=4, HK=4, d=128, is_causal=True, window_size=None, sinks=False),
    dict(seqlens=[128, 256], HQ=4, HK=4, d=128, is_causal=True, window_size=None, sinks=False),
    dict(seqlens=[33, 1023, 3, 128], HQ=4, HK=4, d=128, is_causal=True, window_size=None, sinks=False),
    dict(seqlens=[7, 13, 65, 129, 255], HQ=4, HK=4, d=128, is_causal=True, window_size=None, sinks=False),
    dict(seqlens=[4096], HQ=4, HK=4, d=128, is_causal=True, window_size=None, sinks=False),
    # causal + sinks
    dict(seqlens=[64], HQ=4, HK=4, d=128, is_causal=True, window_size=None, sinks=True),
    dict(seqlens=[128, 256], HQ=8, HK=2, d=128, is_causal=True, window_size=None, sinks=True),
    dict(seqlens=[33, 1023, 3, 128], HQ=12, HK=2, d=128, is_causal=True, window_size=None, sinks=True),
    dict(seqlens=[17, 97, 503, 1], HQ=8, HK=2, d=128, is_causal=True, window_size=None, sinks=True),
    dict(seqlens=[4096], HQ=8, HK=2, d=128, is_causal=True, window_size=None, sinks=True),
    # causal + SWA
    dict(seqlens=[512], HQ=4, HK=4, d=128, is_causal=True, window_size=128, sinks=False),
    dict(seqlens=[1024], HQ=4, HK=4, d=128, is_causal=True, window_size=256, sinks=False),
    dict(seqlens=[128, 256, 512], HQ=8, HK=2, d=128, is_causal=True, window_size=128, sinks=False),
    dict(seqlens=[73, 511, 200], HQ=4, HK=4, d=128, is_causal=True, window_size=128, sinks=False),
    dict(seqlens=[4096], HQ=4, HK=4, d=128, is_causal=True, window_size=512, sinks=False),
    # causal + SWA + sinks
    dict(seqlens=[512], HQ=4, HK=4, d=128, is_causal=True, window_size=128, sinks=True),
    dict(seqlens=[1024], HQ=12, HK=2, d=128, is_causal=True, window_size=256, sinks=True),
    dict(seqlens=[128, 256, 512], HQ=8, HK=2, d=128, is_causal=True, window_size=128, sinks=True),
    dict(seqlens=[63, 127, 513], HQ=8, HK=2, d=128, is_causal=True, window_size=128, sinks=True),
    dict(seqlens=[2048], HQ=4, HK=4, d=128, is_causal=True, window_size=512, sinks=True),
    dict(seqlens=[4096], HQ=4, HK=4, d=128, is_causal=True, window_size=512, sinks=True),
    # non-causal
    dict(seqlens=[128, 256], HQ=4, HK=4, d=128, is_causal=False, window_size=None, sinks=False),
    dict(seqlens=[128, 256], HQ=4, HK=4, d=128, is_causal=False, window_size=None, sinks=True),
    dict(seqlens=[37, 199, 5], HQ=4, HK=4, d=128, is_causal=False, window_size=None, sinks=True),
    dict(seqlens=[4096], HQ=4, HK=4, d=128, is_causal=False, window_size=None, sinks=True),
    # GQA with different group sizes
    dict(seqlens=[256, 128], HQ=24, HK=2, d=128, is_causal=True, window_size=None, sinks=True),
    dict(seqlens=[256, 128], HQ=24, HK=2, d=128, is_causal=True, window_size=128, sinks=True),
    dict(seqlens=[4096], HQ=24, HK=2, d=128, is_causal=True, window_size=512, sinks=True),
    # d=256
    dict(seqlens=[256, 128], HQ=12, HK=2, d=256, is_causal=True, window_size=None, sinks=True),
    dict(seqlens=[256, 128], HQ=12, HK=2, d=256, is_causal=True, window_size=128, sinks=True),
    dict(seqlens=[2048], HQ=12, HK=2, d=256, is_causal=True, window_size=512, sinks=True),
    # edge case: single token
    dict(seqlens=[1, 1, 1, 1], HQ=4, HK=4, d=128, is_causal=True, window_size=None, sinks=True),
    # edge case: window_size >= seq_len
    dict(seqlens=[64], HQ=4, HK=4, d=128, is_causal=True, window_size=128, sinks=True),

    # kv longer than q (packed-CP zigzag Q chunk vs causal K prefix)
    dict(seqlens=[256], seqlens_k=[1024], HQ=8, HK=2, d=128, is_causal=True, window_size=None, sinks=True),
    dict(seqlens=[256], seqlens_k=[1024], HQ=8, HK=2, d=128, is_causal=True, window_size=128, sinks=True),
]


def _case_tag(c):
    ws = c['window_size']
    ws_str = f"sw{ws}" if ws is not None else "causal" if c['is_causal'] else "full"
    tag = f"hq{c['HQ']}hk{c['HK']}d{c['d']}_{ws_str}"
    if c['sinks']:
        tag += "_sink"
    tag += f"_seqs{'x'.join(str(s) for s in c['seqlens'][:3])}"
    if len(c['seqlens']) > 3:
        tag += "..."
    if 'seqlens_k' in c:
        tag += f"_kv{'x'.join(str(s) for s in c['seqlens_k'][:3])}"
    return tag


def _make_inputs(seqlens, HQ, HK, d, seqlens_k=None, dtype=torch.float16):
    cu_seqlens_q = [0]
    for sl in seqlens:
        cu_seqlens_q.append(cu_seqlens_q[-1] + sl)
    total_q = cu_seqlens_q[-1]

    if seqlens_k is None:
        seqlens_k = seqlens
    cu_seqlens_k = [0]
    for sl in seqlens_k:
        cu_seqlens_k.append(cu_seqlens_k[-1] + sl)
    total_k = cu_seqlens_k[-1]

    rng = torch.Generator(device='cuda').manual_seed(1919810)
    q = torch.randn(total_q, HQ, d, generator=rng, device="cuda", dtype=dtype)
    k = torch.randn(total_k, HK, d, generator=rng, device="cuda", dtype=dtype)
    v = torch.randn(total_k, HK, d, generator=rng, device="cuda", dtype=dtype)
    sinks = torch.randn(HQ, generator=rng, device="cuda", dtype=dtype)

    cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda")
    cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="cuda")
    max_seqlen = max(seqlens)

    return q, k, v, sinks, cu_seqlens_q_t, cu_seqlens_k_t, max_seqlen


def _check(tag, a, b, atol_avg=0.01, atol_max=0.05):
    avg = (a - b).abs().mean().item()
    mx = (a - b).abs().max().item()
    print(f'{tag} | avg {avg:.6e} | max {mx:.6e}')
    assert avg < atol_avg, f'{tag} avg {avg} >= {atol_avg}'
    assert mx < atol_max, f'{tag} max {mx} >= {atol_max}'


@pytest.mark.parametrize("case", CASES, ids=[_case_tag(c) for c in CASES])
def test_fwd_bwd(case):
    """验证 varlen attention 的 forward 和 Q/K/V/sinks 梯度。"""
    seqlens = case['seqlens']
    seqlens_k = case.get('seqlens_k', None)
    HQ, HK, d = case['HQ'], case['HK'], case['d']
    is_causal = case['is_causal']
    window_size = case['window_size']
    has_sinks = case['sinks']

    q, k, v, sinks_val, cu_seqlens_q, cu_seqlens_k, max_seqlen = _make_inputs(
        seqlens, HQ, HK, d, seqlens_k=seqlens_k, dtype=torch.float16)
    max_seqlen_k = max(seqlens_k if seqlens_k is not None else seqlens)

    sinks_arg = sinks_val if has_sinks else None

    # reference forward + backward
    with torch.enable_grad():
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)
        sinks_ref = sinks_val.detach().clone().requires_grad_(True) if has_sinks else None
        ref_o = ref_varlen_attn(q_ref, k_ref, v_ref, cu_seqlens_q, cu_seqlens_k,
                                sinks=sinks_ref, is_causal=is_causal, window_size=window_size)
        dO = torch.rand_like(ref_o)
        ref_o.backward(dO)
    ref_dQ = q_ref.grad
    ref_dK = k_ref.grad
    ref_dV = v_ref.grad
    ref_dsinks = sinks_ref.grad if has_sinks else None

    # myfa forward + backward (via autograd wrapper)
    with torch.enable_grad():
        q_my = q.detach().clone().requires_grad_(True)
        k_my = k.detach().clone().requires_grad_(True)
        v_my = v.detach().clone().requires_grad_(True)
        sinks_my = sinks_val.detach().clone().requires_grad_(True) if has_sinks else None
        my_o = myfa_varlen_sw_sinks(
            q_my, k_my, v_my, cu_seqlens_q, cu_seqlens_k, max_seqlen, max_seqlen_k,
            sinks=sinks_my, is_causal=is_causal, window_size=window_size)
        my_o.backward(dO)
    my_dQ = q_my.grad
    my_dK = k_my.grad
    my_dV = v_my.grad
    my_dsinks = sinks_my.grad if has_sinks else None

    tag = _case_tag(case)
    rtol_avg, rtol_max = (0.015, 0.08) if window_size is not None else (0.01, 0.05)
    _check(f'[{tag}] O', my_o, ref_o, atol_avg=rtol_avg, atol_max=rtol_max)
    _check(f'[{tag}] dQ', my_dQ, ref_dQ, atol_avg=rtol_avg, atol_max=rtol_max)
    _check(f'[{tag}] dK', my_dK, ref_dK, atol_avg=rtol_avg, atol_max=rtol_max)
    _check(f'[{tag}] dV', my_dV, ref_dV, atol_avg=rtol_avg, atol_max=rtol_max)
    if has_sinks:
        _check(f'[{tag}] dSinks', my_dsinks, ref_dsinks, atol_avg=rtol_avg, atol_max=rtol_max)
