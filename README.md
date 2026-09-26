# LIS: knowledge-graph facts as attention structure

Character-level transformers are trained on paired sequences, a knowledge-graph (KG) fact followed by a sentence about it, and then asked to write the sentence for names they have never seen, with the true fact hidden among distractor facts in the context. The question is which representation of the fact lets the model pick out the right one.

Paper: Mahesh Godavarti, *Content-based addressing for long context* — [arXiv:2609.07314](https://arxiv.org/abs/2609.07314).

## Task

- **Names**: 3000 random a-z strings for training, 100 disjoint ones for testing. Five domains (family, work, military, school, church), each a chain of six roles; every adjacent pair is a 2-ary relation (chain2) and every adjacent triple a 3-ary relation (chain3).
- **Training sequence**: `distractor | true fact | text`. The text is one of 12 sentence skeletons per arity with domain context, for example "if adam is the son, the father is brian." The distractor is drawn per example with equal probability from the same relation with different entities, or from the same entities with a different relation. The true fact is the one adjacent to the text.
- **Evaluation**: the KG prefix and the text up to the first entity to be predicted are given; the model must produce the remaining entities in their role slots. Only entity characters are scored. h@1 and h@5 require every scored character correct (argmax, top-5); perplexity is over scored characters.
  - *unprimed*: one fact, then text.
  - *same entities*: the true fact plus four facts over the same entities with other relations. The model must read the role in the text.
  - *same relation*: the true fact plus four facts with the same relation and other entities. The model must match the entity in the text to the right fact.
- **Position of the true fact**: in the primed settings the true fact is placed at each of the five positions among the four distractors in turn, and the reported number is the average over the five.

## Models

All share the same transformer (3 layers, width 120, softmax attention, causal over the whole sequence) and the same text handling. They differ only in how the KG facts are presented.

| Model | KG representation | Positions | Fact address |
|---|---|---|---|
| **B** | Linearized with role tokens, `<son> adam <father> brian` | Continuous RoPE over the whole sequence | none |
| Bs | Linearized | RoPE reset at each segment (fact, separator, text) | none |
| Bl | Linearized | Reset RoPE | Content: mean of the segment's rotated embeddings, LayerNorm, linear projection |
| J0 | Native: entity characters only, ` adam  brian ` | RoPE reset inside each entity, plus a learned angle per (relation, role) | none |
| Jr | Native | Same as J0 | Random: an i.i.d. N(0,1) angle vector per fact, redrawn each forward pass |
| **J** | Native | Same as J0 | Content: mean of the fact's rotated character embeddings, LayerNorm, linear projection |

In the native models no role or relation token appears in the sequence. The fact address is added to the rotary angle of every character in the fact. The text carries no address.

## Results

Three training seeds per model, 100K iterations, batch 32, lr 5e-4. Unseen names. Mean ± sample standard deviation over seeds, each seed's number being its average over the five positions of the true fact. The same evaluation examples are used for every model and every seed. Regenerate with `python results/aggregate.py`.

### hits@5

| | J | B | Jr | J0 | Bs | Bl |
|---|---|---|---|---|---|---|
| chain2, unprimed | **0.840 ± 0.042** | 0.597 ± 0.310 | 0.311 ± 0.213 | 0.570 ± 0.203 | 0.663 ± 0.345 | 0.621 ± 0.330 |
| chain2, same entities | **0.922 ± 0.008** | 0.679 ± 0.350 | 0.391 ± 0.297 | 0.740 ± 0.108 | 0.731 ± 0.327 | 0.679 ± 0.334 |
| chain2, same relation | **0.556 ± 0.091** | 0.319 ± 0.166 | 0.136 ± 0.108 | 0.163 ± 0.061 | 0.224 ± 0.124 | 0.302 ± 0.202 |
| chain3, unprimed | **0.669 ± 0.092** | 0.376 ± 0.285 | 0.091 ± 0.100 | 0.348 ± 0.238 | 0.505 ± 0.394 | 0.423 ± 0.338 |
| chain3, same entities | **0.739 ± 0.079** | 0.396 ± 0.301 | 0.171 ± 0.193 | 0.483 ± 0.228 | 0.578 ± 0.432 | 0.509 ± 0.400 |
| chain3, same relation | **0.222 ± 0.090** | 0.054 ± 0.052 | 0.019 ± 0.018 | 0.028 ± 0.018 | 0.098 ± 0.082 | 0.065 ± 0.058 |

### perplexity

| | J | B | Jr | J0 | Bs | Bl |
|---|---|---|---|---|---|---|
| chain2, unprimed | **2.12 ± 0.39** | 13.37 ± 19.21 | 49.16 ± 55.59 | 5.47 ± 3.82 | 5.61 ± 5.34 | 19.56 ± 30.07 |
| chain2, same entities | **1.67 ± 0.06** | 7.55 ± 9.85 | 34.72 ± 45.84 | 2.83 ± 0.45 | 3.40 ± 2.93 | 10.48 ± 15.00 |
| chain2, same relation | **5.08 ± 0.70** | 28.55 ± 37.07 | 98.03 ± 105.62 | 44.60 ± 15.90 | 100.27 ± 40.75 | 42.74 ± 62.88 |
| chain3, unprimed | **2.67 ± 0.55** | 22.16 ± 32.91 | 85.44 ± 94.00 | 9.09 ± 9.03 | 7.60 ± 8.53 | 34.15 ± 54.22 |
| chain3, same entities | **2.20 ± 0.20** | 18.71 ± 26.52 | 62.80 ± 83.19 | 4.58 ± 2.47 | 5.27 ± 5.50 | 18.29 ± 27.66 |
| chain3, same relation | **8.11 ± 1.60** | 69.86 ± 87.54 | 142.13 ± 133.82 | 69.66 ± 21.14 | 81.91 ± 15.24 | 65.55 ± 95.68 |

### Reading the results

- J leads every row on both measures, with the smallest spread across seeds in all but one row.
- Same-relation retrieval is the hard case: the five facts share their relation and roles, so the model must match the entity named in the text to the fact that contains it. J reaches 0.56 on chain2; the linearized B reaches 0.32.
- The address content is what J uses. Re-scoring a trained J with one shared address on all five facts drops same-relation chain2 from 0.63 to 0.16, the level of J0; replacing the trained address with a constant or a random vector gives 0.00 (`results/address_ablation_seed0.log`, `address_ablation.py`). Jr, which trains with random per-fact addresses, does not learn the task.
- Seed variance is large for B, Jr, Bs and Bl: each has at least one of three seeds that generalises poorly to unseen names although its training loss is normal. J and J0 are stable across seeds.

## Causality

```bash
python causality_test.py
```

The scored positions are the text. For every model the test perturbs one token and confirms that no text logit before it changes: all six give exactly 0.0. Positions inside a KG fact are inputs and are never scored; in the two content-address models (J, Bl) a fact's address is pooled over the whole fact and applied to its own characters, so a fact character's logit can depend on later characters of the same fact. The test reports that separately and does not count it as a failure.

## Usage

```bash
pip install torch numpy tqdm

# smoke test
python lis.py --smoke --models B J

# train three seeds and evaluate with the true fact at index 0
python lis.py --models J B Jr J0 Bs Bl --n_embed 120 --n_layers 3 --iters 100000 \
    --n_train 3000 --n_test 100 --n_priming 4 --softmax --seeds 3 --checkpoint_dir ckpt

# re-evaluate the saved checkpoints with the true fact at each other index
for k in 1 2 3 4; do
  python lis.py --models J B Jr J0 Bs Bl --n_embed 120 --n_layers 3 --iters 100000 \
      --n_train 3000 --n_test 100 --n_priming 4 --softmax --seeds 3 --checkpoint_dir ckpt \
      --load_checkpoints --eval_true_pos pos$k
done
```

`--eval_true_pos` takes `pos0` .. `pos4` (0 = furthest from the text), `far`, `near` or `random`. `--seed_offset` starts the seed loop at a given seed. Results are written to `lis_results_<timestamp>.json` with averaged and per-seed summaries; `results/aggregate.py` averages the five position files.

## Files

```
lis.py    — data generation, the six models, training, evaluation
causality_test.py     — scored-position causality check
address_ablation.py   — address-value test on a trained J
results/              — one result file per true-fact position, aggregation script, ablation log
```

## License

Copyright (c) 2026 Mahesh Godavarti. Licensed under CC BY-NC-SA 4.0. See [LICENSE](LICENSE).
