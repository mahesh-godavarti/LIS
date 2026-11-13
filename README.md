# LIS: knowledge-graph facts as attention structure

Character-level transformers are trained on paired sequences, knowledge-graph (KG) facts followed by a sentence about one of them, and then asked to write the sentence for names they have never seen, with the true fact hidden among distractor facts in the context. The question is which representation of the fact lets the model pick out the right one, and whether carrying the fact's rotation into the attention values (value transport) helps.

Paper: Mahesh Godavarti, *Content-based addressing for long context* — [arXiv:2609.07314](https://arxiv.org/abs/2609.07314).

## Task

- **Names**: 3000 random a-z strings for training, 100 disjoint ones for testing. Five domains (family, work, military, school, church), each a chain of six roles; every adjacent pair is a 2-ary relation (chain2) and every adjacent triple a 3-ary relation (chain3).
- **Training sequence**: two facts, then the text: `fact | fact | text`. One fact is the true one; the other is a distractor drawn per example with equal probability from the same relation with different entities, or from the same entities with a different relation. The true fact is first or second at random (`--train_shuffle`). The entities inside a fact are written in random order. The text is one of 12 sentence skeletons per arity with domain context, for example "if adam is the son, the father is brian."
- **Evaluation**: the KG prefix and the text up to the first entity to be predicted are given; the model must produce the remaining entities in their role slots. Only entity characters are scored. hits@5 requires every scored character correct (top-5); perplexity is over scored characters.
  - *unprimed*: one fact, then text.
  - *same entities*: the true fact plus four facts over the same entities with other relations. The model must read the role in the text.
  - *same relation*: the true fact plus four facts with the same relation and other entities. The model must match the entity in the text to the right fact.
- **Position of the true fact**: in the two five-fact settings the true fact is placed at each of the five positions in turn, and the reported number is the average over the five.

An earlier version of this table trained with the true fact always adjacent to the text (`distractor | true fact | text`); those results are kept in `results_fixed_position/` and summarised below.

## Models

All share the same transformer (3 layers, width 120, softmax attention, causal over the whole sequence) and the same text handling. They differ in how the KG facts are presented and in whether the attention values are rotated.

| Model | KG representation | Positions | Fact address |
|---|---|---|---|
| **B** | Linearized with role tokens, `<son> adam <father> brian` | Continuous RoPE over the whole sequence | none |
| Bs | Linearized | RoPE reset at each segment (fact, separator, text) | none |
| Bl | Linearized | Reset RoPE | Content: mean of the segment's rotated embeddings, LayerNorm, linear projection |
| J0 | Native: entity characters only, ` adam  brian ` | RoPE reset inside each entity, plus a learned angle per (relation, role) | none |
| Jr | Native | Same as J0 | Random: an i.i.d. N(0,1) angle vector per fact, redrawn each forward pass |
| **J** | Native | Same as J0 | Content: mean of the fact's rotated character embeddings, LayerNorm, linear projection |

In the native models no role or relation token appears in the sequence. The fact address is added to the rotary angle of every character in the fact. The text carries no address.

A prime (B', Bs', Bl', J0', Jr', J') marks **value transport**: the values are rotated by the same angles as the queries and keys, and the attention output is rotated back by the query's angle. Nothing else changes.

## Results

Six training seeds per model, 100K iterations, batch 32, lr 5e-4, dropout 0.2. Unseen names. Mean ± sample standard deviation over seeds, each seed's number being its average over the five positions of the true fact. The same evaluation examples are used for every model and every seed. Regenerate with `python results/aggregate.py`.

### hits@5

| | J' | J | Bl' | Bl | B' | B | Bs' | Bs | J0' | J0 | Jr' | Jr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain2, unprimed | 0.884 ± 0.046 | 0.633 ± 0.181 | 0.833 ± 0.131 | 0.550 ± 0.233 | 0.817 ± 0.038 | 0.748 ± 0.115 | 0.659 ± 0.240 | 0.815 ± 0.206 | **0.900 ± 0.081** | 0.822 ± 0.104 | 0.003 ± 0.005 | 0.381 ± 0.205 |
| chain2, same entities | **0.960 ± 0.022** | 0.703 ± 0.184 | 0.949 ± 0.050 | 0.670 ± 0.239 | 0.953 ± 0.026 | 0.815 ± 0.077 | 0.723 ± 0.198 | 0.817 ± 0.204 | 0.940 ± 0.022 | 0.822 ± 0.118 | 0.003 ± 0.006 | 0.489 ± 0.280 |
| chain2, same relation | **0.756 ± 0.140** | 0.380 ± 0.212 | 0.521 ± 0.150 | 0.324 ± 0.169 | 0.490 ± 0.041 | 0.413 ± 0.099 | 0.203 ± 0.143 | 0.241 ± 0.121 | 0.387 ± 0.126 | 0.237 ± 0.072 | 0.004 ± 0.006 | 0.173 ± 0.113 |
| chain3, unprimed | **0.775 ± 0.093** | 0.382 ± 0.252 | 0.681 ± 0.185 | 0.324 ± 0.246 | 0.650 ± 0.043 | 0.525 ± 0.143 | 0.445 ± 0.365 | 0.643 ± 0.295 | 0.766 ± 0.148 | 0.655 ± 0.182 | 0.000 ± 0.000 | 0.150 ± 0.119 |
| chain3, same entities | 0.871 ± 0.051 | 0.446 ± 0.263 | **0.876 ± 0.108** | 0.451 ± 0.293 | 0.767 ± 0.061 | 0.458 ± 0.096 | 0.439 ± 0.363 | 0.629 ± 0.322 | 0.792 ± 0.099 | 0.547 ± 0.169 | 0.000 ± 0.000 | 0.264 ± 0.218 |
| chain3, same relation | **0.420 ± 0.168** | 0.118 ± 0.128 | 0.208 ± 0.092 | 0.074 ± 0.066 | 0.115 ± 0.029 | 0.057 ± 0.027 | 0.079 ± 0.102 | 0.105 ± 0.095 | 0.172 ± 0.108 | 0.064 ± 0.039 | 0.000 ± 0.000 | 0.026 ± 0.023 |

### perplexity

| | J' | J | Bl' | Bl | B' | B | Bs' | Bs | J0' | J0 | Jr' | Jr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain2, unprimed | 1.63 ± 0.19 | 4.21 ± 2.17 | 2.28 ± 1.44 | 12.08 ± 15.37 | 2.18 ± 0.35 | 2.86 ± 0.97 | 3.88 ± 2.03 | 2.39 ± 1.43 | **1.46 ± 0.33** | 2.02 ± 0.53 | 499.75 ± 79.76 | 43.84 ± 60.80 |
| chain2, same entities | 1.39 ± 0.10 | 3.05 ± 1.74 | **1.37 ± 0.23** | 6.86 ± 9.46 | 1.40 ± 0.06 | 2.13 ± 0.44 | 2.93 ± 1.26 | 2.36 ± 1.47 | 1.45 ± 0.18 | 2.11 ± 0.62 | 499.69 ± 74.40 | 34.27 ± 52.76 |
| chain2, same relation | **2.58 ± 0.71** | 19.71 ± 31.27 | 10.11 ± 10.72 | 26.44 ± 37.90 | 4.26 ± 0.75 | 7.34 ± 3.05 | 118.73 ± 15.34 | 67.64 ± 29.23 | 26.03 ± 14.21 | 33.89 ± 11.45 | 501.77 ± 76.00 | 98.58 ± 125.64 |
| chain3, unprimed | 1.98 ± 0.28 | 6.71 ± 4.13 | 2.96 ± 2.18 | 21.46 ± 30.26 | 2.55 ± 0.19 | 3.71 ± 1.18 | 5.81 ± 3.44 | 3.16 ± 2.28 | **1.75 ± 0.51** | 2.48 ± 0.88 | 652.72 ± 111.28 | 77.77 ± 108.26 |
| chain3, same entities | 1.76 ± 0.14 | 4.82 ± 3.38 | **1.67 ± 0.36** | 13.02 ± 20.81 | 1.98 ± 0.16 | 6.38 ± 4.72 | 5.68 ± 3.29 | 3.65 ± 3.02 | 1.80 ± 0.24 | 3.35 ± 1.17 | 639.54 ± 105.49 | 60.95 ± 95.27 |
| chain3, same relation | **3.70 ± 0.80** | 29.37 ± 42.19 | 11.76 ± 11.72 | 47.21 ± 71.15 | 15.53 ± 3.54 | 23.90 ± 15.11 | 116.60 ± 29.23 | 68.30 ± 27.15 | 37.94 ± 20.75 | 51.76 ± 17.41 | 639.30 ± 107.28 | 137.55 ± 166.11 |

### Reading the results

- J' is first or within one point of first on every hits@5 row, and has the lowest perplexity on both same-relation rows. The two rows it does not lead are chain2 unprimed (J0' 0.900) and chain3 same entities (Bl' 0.876), both within the seed spread.
- Value transport raises every model that has a content address (J, Bl) or no address (B, J0) on every row. It lowers Bs on every row, and takes Jr, whose address is random and redrawn each pass, to zero.
- The content address is what makes same-relation retrieval independent of where the true fact sits. Averaged over seeds, J' scores 0.76, 0.77, 0.76, 0.75, 0.74 on chain2 same relation at the five positions (furthest from the text to adjacent). J0', the same model without the address, scores 0.72, 0.52, 0.29, 0.22, 0.19 at the same positions; B' 0.54, 0.66, 0.58, 0.40, 0.29; Bl' 0.61, 0.59, 0.52, 0.47, 0.42.
- Without transport, B is above J on chain2 (0.748 vs 0.633 unprimed; 0.413 vs 0.380 same relation) and J is above B on chain3. Under the fixed-position training below, J was above B on every row.
- Seed spread is large for every unprimed model except J0 and B; among the primed models it is small for J', B' and J0' and large for Bs'.

### Fixed-position training (earlier protocol, three seeds)

Trained with `distractor | true fact | text`, the true fact always adjacent to the text; everything else as above. Files in `results_fixed_position/`.

| hits@5 | J | B | Jr | J0 | Bs | Bl |
|---|---|---|---|---|---|---|
| chain2, unprimed | **0.840 ± 0.042** | 0.597 ± 0.310 | 0.311 ± 0.213 | 0.570 ± 0.203 | 0.663 ± 0.345 | 0.621 ± 0.330 |
| chain2, same entities | **0.922 ± 0.008** | 0.679 ± 0.350 | 0.391 ± 0.297 | 0.740 ± 0.108 | 0.731 ± 0.327 | 0.679 ± 0.334 |
| chain2, same relation | **0.556 ± 0.091** | 0.319 ± 0.166 | 0.136 ± 0.108 | 0.163 ± 0.061 | 0.224 ± 0.124 | 0.302 ± 0.202 |
| chain3, unprimed | **0.669 ± 0.092** | 0.376 ± 0.285 | 0.091 ± 0.100 | 0.348 ± 0.238 | 0.505 ± 0.394 | 0.423 ± 0.338 |
| chain3, same entities | **0.739 ± 0.079** | 0.396 ± 0.301 | 0.171 ± 0.193 | 0.483 ± 0.228 | 0.578 ± 0.432 | 0.509 ± 0.400 |
| chain3, same relation | **0.222 ± 0.090** | 0.054 ± 0.052 | 0.019 ± 0.018 | 0.028 ± 0.018 | 0.098 ± 0.082 | 0.065 ± 0.058 |

Under this protocol J leads every row. With value transport added to it, J' matched or beat J on the five-fact rows but three of six seeds scored below 0.40 on the one-fact prompt while scoring above 0.94 with same-entity distractors; randomising the true fact's position in training (the main table) removed that failure. The address ablation in `results_fixed_position/address_ablation_seed0.log` re-scores a fixed-position J with one shared address on all five facts: same-relation chain2 drops from 0.63 to 0.16; a constant or random address gives 0.00.

## Causality

```bash
python causality_test.py
```

The scored positions are the text. For every model the test perturbs one token and confirms that no text logit before it changes: all six give exactly 0.0. Positions inside a KG fact are inputs and are never scored; in the two content-address models (J, Bl) a fact's address is pooled over the whole fact and applied to its own characters, so a fact character's logit can depend on later characters of the same fact. The test reports that separately and does not count it as a failure.

## Usage

```bash
pip install torch numpy tqdm

# smoke test
python lis.py --smoke --models B "J'"

# train six seeds (true fact first or second at random) and evaluate with the true fact at index 0
python lis.py --models "J'" J "Bl'" Bl "B'" B "Bs'" Bs "J0'" J0 "Jr'" Jr \
    --n_embed 120 --n_layers 3 --iters 100000 --n_train 3000 --n_test 100 --n_priming 4 \
    --softmax --train_shuffle --seeds 6 --checkpoint_dir ckpt

# re-evaluate the saved checkpoints with the true fact at each other index
for k in 1 2 3 4; do
  python lis.py --models "J'" J "Bl'" Bl "B'" B "Bs'" Bs "J0'" J0 "Jr'" Jr \
      --n_embed 120 --n_layers 3 --iters 100000 --n_train 3000 --n_test 100 --n_priming 4 \
      --softmax --train_shuffle --seeds 6 --checkpoint_dir ckpt --load_checkpoints --eval_true_pos pos$k
done
```

`--eval_true_pos` takes `pos0` .. `pos4` (0 = furthest from the text), `far`, `near` or `random`. `--seed_offset` starts the seed loop at a given seed. Without `--train_shuffle` the training layout is the fixed-position one. `--train_mix` (0 to 4 distractors, true fact anywhere), `--dropout`, `--weight_decay` and `--cosine` are also available; they were not used for the tables above. Results are written to `lis_results_<timestamp>.json` with averaged and per-seed summaries; `results/aggregate.py` averages the five position files.

## Files

```
lis.py                   — data generation, the twelve models, training, evaluation
causality_test.py        — scored-position causality check
address_ablation.py      — address-value test on a trained J
results/                 — main table: one result file per true-fact position, aggregation script, tables.md
results_fixed_position/  — earlier fixed-position protocol: result files, aggregation script, tables.md, address ablation log
```

## License

Copyright (c) 2026 Mahesh Godavarti. Licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0): free to use, change and share for any noncommercial purpose; commercial use is not permitted. See [LICENSE](LICENSE).
