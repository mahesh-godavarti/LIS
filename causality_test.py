#!/usr/bin/env python3
# Copyright (c) 2026 Mahesh Godavarti
# Licensed under CC BY-NC-SA 4.0. See LICENSE file for details.

"""Causality of the scored positions.

The task scores the text characters only. This test checks that no text logit
depends on a later token: for a sequence  fact | fact | text  it perturbs one
token at position p and reports the largest change in any logit at a text
position before p, for every p in the text. A causal text side gives 0.0.

It also reports the same quantity for positions inside the KG facts. Those
positions are inputs and are never scored; in the content-address models (J,
Bl) a fact's address is pooled over the whole fact and applied to its own
characters, so a fact character's logit can depend on later characters of the
same fact. That is reported, not treated as a failure.

Runs on CPU with small random models; no data needed.
"""

import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pz", os.path.join(HERE, "lis.py"))
pz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pz)

torch.set_grad_enabled(False)
V = 40


def probe(run, T, seg_bounds, seg_labels, text_start):
    torch.manual_seed(0)
    x = torch.randint(3, V, (1, T))
    base = run(x)
    worst_text, worst_fact = 0.0, 0.0
    for p in range(1, T):
        xp = x.clone()
        xp[0, p] = (xp[0, p] + 3) % V if xp[0, p] > 3 else 4
        d = (base[:p] - run(xp)[:p]).abs().max(dim=-1).values
        if p > text_start:
            worst_text = max(worst_text, d[text_start:p].max().item())
        if p < text_start:
            worst_fact = max(worst_fact, d[:p].max().item())
    return worst_text, worst_fact


def main():
    rels = list(pz.KG_RELATIONS)
    r0 = rels[0]
    ok = True
    print(f"{'model':6s} {'text positions':>16s} {'fact positions':>16s}")

    # J family: two 2-ary facts (entity lens 6,7 | 5,6 incl. spaces), separator, text of 20
    el = [[6, 7, 5, 6, 1, 20]]
    rs = [[0, 1, 0, 1, pz.MAX_ARITY, pz.MAX_ARITY]]
    rn = [[r0] * 6]
    fi = [[0, 0, 1, 1, -1, -1]]
    T = sum(el[0]); text_start = T - 20
    for name, mode in [("J", "content"), ("Jr", "random"), ("J0", "none")]:
        torch.manual_seed(1)
        m = pz.ModelJ(vocab_size=V, n_embed=32, n_layers=2, block_size=128, dropout=0.0,
                      address_mode=mode).eval()
        def run(inp, m=m):
            torch.manual_seed(7)  # Jr: same random addresses for both passes
            return m.forward_paired(inp, None, el, rn, rs, fact_ids=fi)[0][0]
        wt, wf = probe(run, T, None, None, text_start)
        ok &= wt < 1e-6
        print(f"{name:6s} {wt:16.2e} {wf:16.2e}")

    # B family: segments of 10, 12, then text of 20
    seg = [[10, 12, 20]]
    T = 42; text_start = 22
    torch.manual_seed(1)
    b = pz.ModelB(vocab_size=V, n_embed=32, n_layers=2, block_size=128, dropout=0.0).eval()
    wt, wf = probe(lambda inp: b.forward(inp)[0][0], T, None, None, text_start)
    ok &= wt < 1e-6; print(f"{'B':6s} {wt:16.2e} {wf:16.2e}")
    wt, wf = probe(lambda inp: b.forward_segmented(inp, None, seg)[0][0], T, None, None, text_start)
    ok &= wt < 1e-6; print(f"{'Bs':6s} {wt:16.2e} {wf:16.2e}")
    torch.manual_seed(1)
    bl = pz.ModelBl(vocab_size=V, n_embed=32, n_layers=2, block_size=128, dropout=0.0).eval()
    wt, wf = probe(lambda inp: bl.forward_content_addressed(inp, None, seg)[0][0], T, None, None, text_start)
    ok &= wt < 1e-6; print(f"{'Bl':6s} {wt:16.2e} {wf:16.2e}")

    print()
    print("text positions (scored): " + ("all 0 -> PASS" if ok else "NONZERO -> FAIL"))
    print("fact positions (inputs, not scored): nonzero for content-address models is expected")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
