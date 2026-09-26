#!/usr/bin/env python3
# Copyright (c) 2026 Mahesh Godavarti
# Licensed under CC BY-NC-SA 4.0. See LICENSE file for details.

"""Does J read the VALUE of the fact address, or only its presence?

Loads a trained J checkpoint and re-scores the same-relation prompts (true
fact at a random position) with the address computed five ways:
  content          as trained: pooled from each fact's own characters
  shared_content   every fact gets fact 0's pooled address (one value for all)
  constant         one fixed random vector on every fact
  random_per_fact  a different random vector per fact, fresh per prompt
  none             no address (J0 geometry)

    python address_ablation.py --seed 0 --n 600
"""

import argparse
import contextlib
import importlib.util
import io
import os
import random

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pz", os.path.join(HERE, "lis.py"))
pz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--checkpoint_dir", default="checkpoints_n3000_n120_l3")
    args = ap.parse_args()
    cfg = pz.cfg
    cfg.n_embed, cfg.n_layers, cfg.use_softmax, cfg.batch_size = 120, 3, True, 32
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    tn, te, tc, tec, tf, tts, tl = pz.generate_train_test_data(n_train=3000, n_test=100, seed=args.seed)
    vocab = pz.Vocabulary(); vocab.build_from_sentences(tts + tl); vocab.build_from_sentences(tn + te)
    model = pz.create_model("J", vocab.size, cfg)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f"model_J_seed{args.seed}.pt"),
                                     map_location=cfg.device))
    model.to(cfg.device).eval()
    orig = model._kg_angles
    half = cfg.n_embed // 2
    FIXED = torch.randn(half) * 0.5

    def make(mode):
        def patched(entity_lens, seq_len, batch_size, rel_names, device, role_slots=None,
                    fact_ids=None, tok_emb=None):
            if mode == "content":
                return orig(entity_lens, seq_len, batch_size, rel_names, device,
                            role_slots=role_slots, fact_ids=fact_ids, tok_emb=tok_emb)
            if mode == "none":
                return orig(entity_lens, seq_len, batch_size, rel_names, device,
                            role_slots=role_slots, fact_ids=None, tok_emb=None)
            if mode == "shared_content":
                fi = [[0 if f >= 0 else -1 for f in row] for row in fact_ids]
                return orig(entity_lens, seq_len, batch_size, rel_names, device,
                            role_slots=role_slots, fact_ids=fi, tok_emb=tok_emb)
            out = orig(entity_lens, seq_len, batch_size, rel_names, device,
                       role_slots=role_slots, fact_ids=None, tok_emb=None).clone()
            for bi in range(batch_size):
                pos = 0; per_fact = {}
                for L, f in zip(entity_lens[bi], fact_ids[bi]):
                    if f >= 0:
                        v = FIXED if mode == "constant" else per_fact.setdefault(f, torch.randn(half) * 0.5)
                        out[bi, pos:pos + L] += v.to(out.device)
                    pos += L
            return out
        return patched

    def score(prompts, label):
        torch.manual_seed(3)
        with contextlib.redirect_stdout(io.StringIO()):
            res = pz.evaluate_zero_shot(model, prompts, vocab, cfg, model_name="J", model_type="J")
        summ = res[0] if isinstance(res, tuple) else res
        k2 = summ[('kg_to_text', 'chain2', 'test', 'same_rel')]
        k3 = summ[('kg_to_text', 'chain3', 'test', 'same_rel')]
        print(f"RESULT {label:48s} chain2 h@5={k2['hit5']:.3f} ppl={k2['ppl']:6.2f}   "
              f"chain3 h@5={k3['hit5']:.3f} ppl={k3['ppl']:6.2f}", flush=True)

    def prompts_at(pos, seed):
        pz.EVAL_TRUE_POS = pos; random.seed(seed)
        ps = [p for p in pz.build_primed_eval_prompts(tec, vocab, n_priming=4, name_set_label="test",
                                                      model_type="J") if p["priming_mode"] == "same_rel"]
        random.shuffle(ps); return ps[:args.n]

    print("== address ablation, true fact at a random position")
    ps = prompts_at("random", 7)
    for mode in ("content", "shared_content", "constant", "random_per_fact", "none"):
        model._kg_angles = make(mode); score(ps, f"address = {mode}")
    model._kg_angles = orig


if __name__ == "__main__":
    main()
