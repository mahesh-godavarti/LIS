# Copyright (c) 2026 Mahesh Godavarti
# Licensed under CC BY-NC-SA 4.0. See LICENSE file for details.
#
# -----------------------------------------------------------------------------
# Primed Zero-Shot KG+Text Paired Experiment
#
# Models (KG representation / positions / fact address):
#   B    linearized <role> tokens / continuous RoPE / none
#   Bs   linearized / RoPE reset at each segment / none
#   Bl   linearized / reset RoPE / content address per segment
#   J0   native (entity characters + slot angles) / reset RoPE / none
#   Jr   native / reset RoPE / random address per fact
#   J    native / reset RoPE / content address per fact
#   X'   any model with V rotated as well as Q and K
#
# Task. Each training sequence is  distractor | true fact | text , where the
# text is a natural-language sentence about the true fact ("if adam is the
# son, the father is brian.") from 12 skeletons per arity with domain context.
# The distractor is drawn per example, with equal probability, from the same
# relation with different entities or from the same entities with a different
# relation. The true fact is always adjacent to the text in training.
#
# Evaluation on names never seen in training. The model is given the KG prefix
# and the text up to the first predicted entity, and must produce the remaining
# entities in their role slots; only entity characters are scored.
#   unprimed        1 true fact | text
#   same_entities   true fact + 4 distractors with the same entities, other relations
#   same_rel        true fact + 4 distractors with the same relation, other entities
# --eval_true_pos places the true fact among the 4 distractors: pos0 .. pos4
# (pos0 furthest from the text, pos4 adjacent), far (= pos0), near (= pos4),
# random. Reported numbers are averaged over pos0 .. pos4.
#
# Scoring: h@1 / h@5 require every scored character correct (argmax / top-5);
# PPL = exp(-sum of scored log-probs / number of scored characters).
#
# Five domains (family, work, military, school, church), each a chain of six
# roles; chain2 relations link adjacent roles, chain3 relations link three.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse
import os
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

class Config:
    n_embed = 24
    n_layers = 1
    dropout = 0.2
    block_size = 384      # increased from 192 for primed sequences
    batch_size = 32
    max_iters = 5000
    lr = 5e-4
    eval_interval = 500
    eval_iters = 20
    n_seeds = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_softmax = False

cfg = Config()

# Where the true fact sits among the priming facts at evaluation (training always puts it
# adjacent to the text). "posK": index K among the n_priming + 1 facts, 0 = furthest from the
# text. "far" = pos0, "near" = last, "random" = uniformly random.
EVAL_TRUE_POS = "far"


def _place_true_fact(true_pair, distractor_pairs):
    if EVAL_TRUE_POS == "far":
        return [true_pair] + distractor_pairs
    if EVAL_TRUE_POS == "near":
        return distractor_pairs + [true_pair]
    if EVAL_TRUE_POS.startswith("pos"):
        # fixed index among the n_priming + 1 facts
        k = int(EVAL_TRUE_POS[3:])
        return distractor_pairs[:k] + [true_pair] + distractor_pairs[k:]
    pairs = distractor_pairs + [true_pair]
    random.shuffle(pairs)
    return pairs

# ============================================================================
# Names
# ============================================================================

def get_names(n_needed, seed=12345, exclude=None, min_len=2, max_len=8):
    import string
    rng = random.Random(seed)
    letters = string.ascii_lowercase
    names = set(exclude) if exclude else set()
    result = []
    while len(result) < n_needed:
        name_len = rng.randint(min_len, max_len)
        name = "".join(rng.choice(letters) for _ in range(name_len))
        if name not in names:
            names.add(name)
            result.append(name)
    return result

# ============================================================================
# Domains: Relations, Templates, KG Schema
# ============================================================================

CHAIN_ROLES = {
    "family": ["great_grandson", "grandson", "son", "father", "grandfather", "great_grandfather"],
    "work": ["intern", "employee", "boss", "superboss", "director", "executive"],
    "military": ["recruit", "private", "sergeant", "captain", "major", "colonel"],
    "school": ["freshman", "sophomore", "junior", "senior", "graduate", "alumnus"],
    "church": ["acolyte", "deacon", "priest", "bishop", "archbishop", "cardinal"],
}
CHAIN3_WINDOWS = [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5)]  # all sliding 3-windows

# Build relation schemas: only chain3 and chain2 (no forward/inverse redundancy)
RELATION_SCHEMA = {}
KG_RELATIONS = []

for _domain_name, _roles in CHAIN_ROLES.items():
    # 3-ary windows (sliding)
    for _wi, (_i, _j, _k) in enumerate(CHAIN3_WINDOWS):
        _rel = f"chain3_{_domain_name}_w{_wi}"
        RELATION_SCHEMA[_rel] = {"arity": 3, "roles": [_roles[_i], _roles[_j], _roles[_k]]}
        KG_RELATIONS.append(_rel)
    # 2-ary windows (consecutive pairs)
    for _pi in range(5):
        _rel = f"chain2_{_domain_name}_p{_pi}"
        RELATION_SCHEMA[_rel] = {"arity": 2, "roles": [_roles[_pi], _roles[_pi + 1]]}
        KG_RELATIONS.append(_rel)

# Collect all unique role names for KG tokens
KG_ROLES = sorted(set(r for s in RELATION_SCHEMA.values() for r in s["roles"]))

MAX_ARITY = max(s["arity"] for s in RELATION_SCHEMA.values())

# Display names for roles in text templates
ROLE_DISPLAY = {
    "great_grandson": "great-grandson", "grandson": "grandson",
    "son": "son", "father": "father",
    "grandfather": "grandfather", "great_grandfather": "great-grandfather",
    "intern": "intern", "employee": "employee", "boss": "boss",
    "superboss": "superboss", "director": "director", "executive": "executive",
    "recruit": "recruit", "private": "private", "sergeant": "sergeant",
    "captain": "captain", "major": "major", "colonel": "colonel",
    "freshman": "freshman", "sophomore": "sophomore", "junior": "junior",
    "senior": "senior", "graduate": "graduate", "alumnus": "alumnus",
    "acolyte": "acolyte", "deacon": "deacon", "priest": "priest",
    "bishop": "bishop", "archbishop": "archbishop", "cardinal": "cardinal",
}

# Domain context words for template skeletons
DOMAIN_CONTEXT = {
    "family": ("family", "in the family"),
    "work": ("company", "at the company"),
    "military": ("regiment", "in the regiment"),
    "school": ("school", "at the school"),
    "church": ("parish", "in the parish"),
}

def _get_domain(rel):
    """Extract domain name from relation name (e.g. 'chain2_family_p0' -> 'family')."""
    return rel.split("_")[1]


# Skeleton markers
GIVEN = "<<GIVEN>>"
PRED0 = "<<PRED0>>"
PRED1 = "<<PRED1>>"

# 12 skeletons for 2-ary relations (1 predicted entity)
SKELETONS_2ARY = [
    ["if ", GIVEN, " is the {rg}, the {rp} is ", PRED0, "."],
    ["", GIVEN, " is the {rg}. the {rp} is ", PRED0, "."],
    ["the {rg} is ", GIVEN, " and the {rp} is ", PRED0, "."],
    ["", GIVEN, " is the {rg}, so ", PRED0, " is the {rp}."],
    ["since ", GIVEN, " is the {rg}, the {rp} is ", PRED0, "."],
    ["given that ", GIVEN, " is the {rg}, ", PRED0, " is the {rp}."],
    ["{in_ctx}, ", GIVEN, " is the {rg} and ", PRED0, " is the {rp}."],
    ["{in_ctx}, the {rg} is ", GIVEN, ". the {rp} is ", PRED0, "."],
    ["{in_ctx}, if ", GIVEN, " is the {rg}, then ", PRED0, " is the {rp}."],
    ["the {rg} {in_ctx} is ", GIVEN, ", and the {rp} is ", PRED0, "."],
    ["", GIVEN, " is the {rg} {in_ctx}. ", PRED0, " is the {rp}."],
    ["{in_ctx}, since ", GIVEN, " is the {rg}, ", PRED0, " is the {rp}."],
]

# 12 skeletons for 3-ary relations (2 predicted entities)
SKELETONS_3ARY = [
    ["if ", GIVEN, " is the {rg}, the {rp1} is ", PRED0, " and the {rp2} is ", PRED1, "."],
    ["", GIVEN, " is the {rg}. the {rp1} is ", PRED0, " and the {rp2} is ", PRED1, "."],
    ["the {rg} is ", GIVEN, ", the {rp1} is ", PRED0, " and the {rp2} is ", PRED1, "."],
    ["since ", GIVEN, " is the {rg}, ", PRED0, " is the {rp1} and ", PRED1, " is the {rp2}."],
    ["", GIVEN, " is the {rg}, so ", PRED0, " is the {rp1} and ", PRED1, " is the {rp2}."],
    ["{in_ctx}, ", GIVEN, " is the {rg}. the {rp1} is ", PRED0, " and the {rp2} is ", PRED1, "."],
    ["{in_ctx}, the {rg} is ", GIVEN, ". the {rp1} is ", PRED0, " and the {rp2} is ", PRED1, "."],
    ["given that ", GIVEN, " is the {rg}, the {rp1} is ", PRED0, " and the {rp2} is ", PRED1, "."],
    ["{in_ctx}, if ", GIVEN, " is the {rg}, then ", PRED0, " is the {rp1} and ", PRED1, " is the {rp2}."],
    ["the {rg} {in_ctx} is ", GIVEN, ". the {rp1} is ", PRED0, ", the {rp2} is ", PRED1, "."],
    ["", GIVEN, " is the {rg} {in_ctx}. ", PRED0, " is the {rp1} and ", PRED1, " is the {rp2}."],
    ["{in_ctx}, since ", GIVEN, " is the {rg}, ", PRED0, " is the {rp1} and ", PRED1, " is the {rp2}."],
]


def generate_template_variants(rel):
    """Generate all text template variants for a relation.

    For 2-ary: 12 skeletons × 2 given perms = 24 variants
    For 3-ary: 12 skeletons × 6 given/pred perms = 72 variants

    Returns list of {skeleton_idx, given_idx, pred_indices}.
    """
    arity = RELATION_SCHEMA[rel]["arity"]
    skeletons = SKELETONS_3ARY if arity == 3 else SKELETONS_2ARY
    variants = []
    for skel_idx in range(len(skeletons)):
        if arity == 3:
            for given in range(3):
                others = [i for i in range(3) if i != given]
                variants.append({"skeleton_idx": skel_idx, "given_idx": given, "pred_indices": list(others)})
                variants.append({"skeleton_idx": skel_idx, "given_idx": given, "pred_indices": list(others[::-1])})
        elif arity == 2:
            for given in range(2):
                variants.append({"skeleton_idx": skel_idx, "given_idx": given, "pred_indices": [1 - given]})
    return variants


def _substitute_skeleton(skeleton, rn_g, rn_preds, in_ctx, arity):
    """Substitute role names and context into skeleton text segments.

    Returns the skeleton with {rg}, {rp}/{rp1}/{rp2}, {in_ctx} replaced.
    """
    result = []
    for piece in skeleton:
        if piece in (GIVEN, PRED0, PRED1):
            result.append(piece)
        else:
            s = piece.replace("{rg}", rn_g).replace("{in_ctx}", in_ctx)
            if arity == 2:
                s = s.replace("{rp}", rn_preds[0])
            else:
                s = s.replace("{rp1}", rn_preds[0]).replace("{rp2}", rn_preds[1])
            result.append(s)
    return result


def format_template(entities, roles, variant, rel):
    """Build text string from skeleton, entities, roles, and domain context."""
    g = variant["given_idx"]
    preds = variant["pred_indices"]
    skel_idx = variant["skeleton_idx"]
    arity = len(entities)
    rn = [ROLE_DISPLAY[r] for r in roles]

    domain = _get_domain(rel)
    _, in_ctx = DOMAIN_CONTEXT[domain]
    skeletons = SKELETONS_3ARY if arity == 3 else SKELETONS_2ARY
    skeleton = skeletons[skel_idx]

    rn_preds = [rn[p] for p in preds]
    filled = _substitute_skeleton(skeleton, rn[g], rn_preds, in_ctx, arity)

    # Replace markers with actual entity text
    parts = []
    for piece in filled:
        if piece == GIVEN:
            parts.append(entities[g])
        elif piece == PRED0:
            parts.append(entities[preds[0]])
        elif piece == PRED1:
            parts.append(entities[preds[1]])
        else:
            parts.append(piece)
    return "".join(parts)


def build_template_segments(entities, roles, variant, vocab, rel):
    """Build tokenized segments for multi-target eval.

    Returns (prefix_text, segments) where:
      prefix_text: string up to (but not including) first PRED marker (not scored)
      segments: list of {"tokens": [...], "score": bool}
    """
    g = variant["given_idx"]
    preds = variant["pred_indices"]
    skel_idx = variant["skeleton_idx"]
    arity = len(entities)
    rn = [ROLE_DISPLAY[r] for r in roles]

    domain = _get_domain(rel)
    _, in_ctx = DOMAIN_CONTEXT[domain]
    skeletons = SKELETONS_3ARY if arity == 3 else SKELETONS_2ARY
    skeleton = skeletons[skel_idx]

    rn_preds = [rn[p] for p in preds]
    filled = _substitute_skeleton(skeleton, rn[g], rn_preds, in_ctx, arity)

    # Replace GIVEN with actual entity text, split at PRED markers
    # Build: prefix_text (everything before first PRED),
    #        then alternating scored/unscored segments
    prefix_parts = []
    segments = []
    found_first_pred = False

    for piece in filled:
        if piece == GIVEN:
            if not found_first_pred:
                prefix_parts.append(entities[g])
            else:
                # GIVEN after a PRED (shouldn't happen in our skeletons, but handle it)
                segments.append({"tokens": vocab.encode_sentence(entities[g]), "score": False})
        elif piece == PRED0:
            found_first_pred = True
            segments.append({"tokens": vocab.encode_entity(entities[preds[0]]), "score": True})
        elif piece == PRED1:
            segments.append({"tokens": vocab.encode_entity(entities[preds[1]]), "score": True})
        else:
            if not found_first_pred:
                prefix_parts.append(piece)
            else:
                segments.append({"tokens": vocab.encode_sentence(piece), "score": False})

    prefix = "".join(prefix_parts)
    return prefix, segments


# Pre-compute variants for all relations
TEMPLATE_VARIANTS = {rel: generate_template_variants(rel) for rel in RELATION_SCHEMA}


# ============================================================================
# Vocabulary
# ============================================================================

class Vocabulary:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.size = 0

        self.PAD = self._add("<PAD>")
        self.BOS = self._add("<BOS>")
        self.EOS = self._add("<EOS>")
        self.MASK = self._add("<MASK>")
        self.SEP = self._add("|")  # separator token

        self.kg_relations = {}
        for rel in KG_RELATIONS:
            self.kg_relations[rel] = self._add(f"<{rel}>")
            self.kg_relations[f"inverse_{rel}"] = self._add(f"<inverse_{rel}>")

        self.kg_roles = {}
        for role in KG_ROLES:
            self.kg_roles[role] = self._add(f"<{role}>")

    def _add(self, token):
        if token not in self.char2idx:
            idx = self.size
            self.char2idx[token] = idx
            self.idx2char[idx] = token
            self.size += 1
        return self.char2idx[token]

    def build_from_sentences(self, sentences):
        for s in sentences:
            for ch in s:
                self._add(ch)

    def encode_sentence(self, sentence):
        tokens = []
        i = 0
        while i < len(sentence):
            if sentence[i] == '<':
                j = sentence.index('>', i)
                token_str = sentence[i:j+1]
                tokens.append(self.char2idx[token_str])
                i = j + 1
            elif sentence[i] == '|':
                tokens.append(self.SEP)
                i += 1
            else:
                tokens.append(self.char2idx[sentence[i]])
                i += 1
        return tokens

    def encode_entity(self, entity):
        return [self.char2idx[ch] for ch in entity]

    def decode(self, indices):
        return "".join(self.idx2char.get(i, "?") for i in indices)


# ============================================================================
# Fact Generation (simplified — no tiers)
# ============================================================================

def _make_fact(rel, entities):
    schema = RELATION_SCHEMA[rel]
    return {
        "rel": rel,
        "arity": schema["arity"],
        "entities": list(entities),
        "roles": list(schema["roles"]),
    }


def generate_all_facts(chain, domain_name):
    """Generate chain2 and chain3 facts from one chain.

    chain2: one fact per consecutive pair, per chain2 relation
    chain3: one fact per consecutive triple, per chain3 relation
    """
    n = len(chain)
    facts = []

    # chain2: consecutive pairs (need at least 2 names)
    for pos in range(n - 1):
        A, B = chain[pos], chain[pos + 1]
        for pi in range(5):
            rel = f"chain2_{domain_name}_p{pi}"
            if rel in RELATION_SCHEMA:
                facts.append(_make_fact(rel, [A, B]))

    # chain3: consecutive triples (need at least 3 names)
    for pos in range(n - 2):
        A, B, C = chain[pos], chain[pos + 1], chain[pos + 2]
        for wi in range(len(CHAIN3_WINDOWS)):
            rel = f"chain3_{domain_name}_w{wi}"
            if rel in RELATION_SCHEMA:
                facts.append(_make_fact(rel, [A, B, C]))

    # Deduplicate
    seen = set()
    unique = []
    for f in facts:
        key = (f["rel"], tuple(f["entities"]))
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def generate_text_sentences(facts, all_variations=True):
    """Generate text sentences for vocabulary building."""
    sentences = []
    for fact in facts:
        rel = fact["rel"]
        entities = fact["entities"]
        roles = fact["roles"]
        variants = TEMPLATE_VARIANTS[rel]
        if all_variations:
            for v in variants:
                sentences.append(format_template(entities, roles, v, rel))
        else:
            sentences.append(format_template(entities, roles, variants[0], rel))
    return sentences


def linearize_kg_facts(facts):
    """Linearize KG facts for vocabulary building."""
    sentences = []
    for fact in facts:
        parts = []
        for entity, role in zip(fact["entities"], fact["roles"]):
            parts.append(f"<{role}> {entity}")
        sentences.append(" ".join(parts))
    return sentences


# ============================================================================
# Data Generation
# ============================================================================

def generate_train_test_data(n_train=500, n_test=100, seed=42,
                             min_name_len=2, max_name_len=8):
    """Generate disjoint train/test name sets and all facts."""
    rng = random.Random(seed)

    train_names = get_names(n_train, seed=seed,
                            min_len=min_name_len, max_len=max_name_len)
    test_names = get_names(n_test, seed=seed + 5000, exclude=set(train_names),
                           min_len=min_name_len, max_len=max_name_len)

    # Build chains per domain (random permutations)
    train_chains = {}
    test_chains = {}
    for domain in CHAIN_ROLES.keys():
        perm = train_names.copy()
        rng.shuffle(perm)
        train_chains[domain] = perm

        perm = test_names.copy()
        rng.shuffle(perm)
        test_chains[domain] = perm

    # Generate all facts (no tiers)
    train_facts = []
    for domain_name in CHAIN_ROLES.keys():
        train_facts.extend(generate_all_facts(train_chains[domain_name], domain_name))

    # Generate text sentences
    train_text_sentences = generate_text_sentences(train_facts)

    # Linearized KG
    train_linearized = linearize_kg_facts(train_facts)

    return (train_names, test_names, train_chains, test_chains,
            train_facts, train_text_sentences, train_linearized)


# ============================================================================
# Paired Dataset
# ============================================================================

class PairedDataset:
    """Dataset that yields paired KG+text sequences separated by |.

    v5: Each training example has 1 true fact + 1 same-relation distractor.
    """

    def __init__(self, facts, vocab, block_size):
        self.vocab = vocab
        self.block_size = block_size
        self.facts = facts
        self.fact_list = facts

        # Pre-compute template variants for each fact
        self.fact_variants = []
        for fact in facts:
            self.fact_variants.append(TEMPLATE_VARIANTS[fact["rel"]])

        # Group fact indices by relation for distractor selection
        self.facts_by_rel = defaultdict(list)
        for i, fact in enumerate(facts):
            self.facts_by_rel[fact["rel"]].append(i)

        # Group fact indices by entity set for distractor selection
        self.facts_by_entities = defaultdict(list)
        for i, fact in enumerate(facts):
            key = frozenset(fact["entities"])
            self.facts_by_entities[key].append(i)

    def _pick_distractor_same_rel(self, true_idx):
        """Pick a random distractor from the SAME RELATION (different entities)."""
        rel = self.fact_list[true_idx]["rel"]
        candidates = self.facts_by_rel[rel]
        while True:
            d_idx = random.choice(candidates)
            if d_idx != true_idx:
                return d_idx

    def _pick_distractor_same_entities(self, true_idx):
        """Pick a random distractor with the SAME ENTITIES but different relation."""
        fact = self.fact_list[true_idx]
        key = frozenset(fact["entities"])
        candidates = self.facts_by_entities[key]
        # Filter to different relation
        diff_rel = [i for i in candidates if self.fact_list[i]["rel"] != fact["rel"]]
        if not diff_rel:
            # Fallback: pick any distractor with different relation
            while True:
                d_idx = random.randint(0, len(self.fact_list) - 1)
                if d_idx != true_idx and self.fact_list[d_idx]["rel"] != fact["rel"]:
                    return d_idx
        return random.choice(diff_rel)

    def _linearize_fact_permuted(self, fact):
        """Linearize a fact with random entity permutation."""
        entities = fact["entities"]
        roles = fact["roles"]
        arity = len(entities)
        perm = list(range(arity))
        random.shuffle(perm)
        parts = []
        for orig_idx in perm:
            parts.append(f"<{roles[orig_idx]}> {entities[orig_idx]}")
        return " ".join(parts)

    def _build_native_kg_data(self, fact):
        """Build native KG data for J: space-wrapped entities with permutation."""
        entities = fact["entities"]
        arity = len(entities)
        perm = list(range(arity))
        random.shuffle(perm)
        space_id = self.vocab.char2idx[" "]
        seq = []
        perm_lens = []
        for orig_idx in perm:
            ent_tokens = self.vocab.encode_entity(entities[orig_idx])
            seq.append(space_id)
            seq.extend(ent_tokens)
            seq.append(space_id)
            perm_lens.append(len(ent_tokens) + 2)
        return seq, perm_lens, perm

    def get_paired_batch_B(self, batch_size, device):
        """Get a batch of paired linearized KG | text sequences for Model B.

        v5: 1 true fact + 1 same-rel distractor + 1 same-entities distractor,
        shuffled order, then text.
        Returns segment_lens for position resets at " | " boundary.
        """
        indices = torch.randint(0, len(self.fact_list), (batch_size,))
        seqs = []
        segment_lens_list = []
        for idx in indices:
            idx = idx.item()
            fact = self.fact_list[idx]
            variant = random.choice(self.fact_variants[idx])

            # Build true KG tokens
            true_kg_str = self._linearize_fact_permuted(fact)
            true_kg_tokens = self.vocab.encode_sentence(true_kg_str)

            # Pick one distractor at random: same-rel or same-entities
            if random.random() < 0.5:
                d_idx = self._pick_distractor_same_rel(idx)
            else:
                d_idx = self._pick_distractor_same_entities(idx)
            d_fact = self.fact_list[d_idx]
            d_kg_str = self._linearize_fact_permuted(d_fact)
            d_kg_tokens = self.vocab.encode_sentence(d_kg_str)

            # Distractor first, true fact closest to text
            # Concatenate: distractor | true_fact | text
            sep_tokens = self.vocab.encode_sentence(" | ")
            encoded = d_kg_tokens + sep_tokens + true_kg_tokens
            seg_lens = [len(d_kg_tokens), len(sep_tokens), len(true_kg_tokens)]

            text_str = format_template(fact["entities"], fact["roles"], variant, fact["rel"])
            text_tokens = self.vocab.encode_sentence(" | " + text_str)
            encoded += text_tokens
            seg_lens.append(len(text_tokens))

            if len(encoded) > self.block_size:
                encoded = encoded[:self.block_size]
                total = 0
                new_segs = []
                for sl in seg_lens:
                    if total + sl <= len(encoded):
                        new_segs.append(sl)
                        total += sl
                    else:
                        remaining = len(encoded) - total
                        if remaining > 0:
                            new_segs.append(remaining)
                        break
                seg_lens = new_segs

            seqs.append(encoded)
            segment_lens_list.append(seg_lens)

        max_len = max(len(s) for s in seqs)
        x = torch.full((batch_size, max_len - 1), self.vocab.PAD, dtype=torch.long)
        y = torch.full((batch_size, max_len - 1), -100, dtype=torch.long)
        for i, seq in enumerate(seqs):
            seq_t = torch.tensor(seq, dtype=torch.long)
            seq_len = len(seq) - 1
            x[i, :seq_len] = seq_t[:-1]
            y[i, :seq_len] = seq_t[1:]

        # Adjust segment_lens for x (which is one shorter than seq due to :-1)
        for i in range(batch_size):
            total = sum(segment_lens_list[i])
            if total > max_len - 1:
                diff = total - (max_len - 1)
                segment_lens_list[i][-1] -= diff
                if segment_lens_list[i][-1] <= 0:
                    segment_lens_list[i].pop()

        return x.to(device), y.to(device), segment_lens_list

    def get_paired_batch_J(self, batch_size, device):
        """Get a batch of paired native KG | text sequences for Model J.

        v5: 1 true fact + 1 same-rel distractor + 1 same-entities distractor,
        shuffled order, then text.
        """
        indices = torch.randint(0, len(self.fact_list), (batch_size,))
        seqs = []
        entity_lens_list = []
        role_slots_list = []
        rel_names = []
        fact_ids_list = []

        TEXT_SLOT = MAX_ARITY

        for idx in indices:
            idx = idx.item()
            fact = self.fact_list[idx]
            variant = random.choice(self.fact_variants[idx])

            # Build true KG data
            true_seq, true_lens, true_perm = self._build_native_kg_data(fact)

            # Pick one distractor at random: same-rel or same-entities
            if random.random() < 0.5:
                d_idx = self._pick_distractor_same_rel(idx)
            else:
                d_idx = self._pick_distractor_same_entities(idx)
            d_fact = self.fact_list[d_idx]
            d_seq, d_lens, d_perm = self._build_native_kg_data(d_fact)

            # Distractor first, true fact closest to text
            # fact_id: 0=true, 1=distractor
            d_item = (d_seq, d_lens, d_perm, d_fact["rel"], 1)
            t_item = (true_seq, true_lens, true_perm, fact["rel"], 0)

            # Concatenate: distractor | true_fact | text
            sep_tokens = self.vocab.encode_sentence(" | ")
            seq = d_item[0] + sep_tokens + t_item[0]
            e_lens = (d_item[1]
                      + [len(sep_tokens)]
                      + t_item[1])
            r_slots = (d_item[2]
                       + [TEXT_SLOT]
                       + t_item[2])
            r_names = ([d_item[3]] * len(d_item[1])
                       + [d_item[3]]
                       + [t_item[3]] * len(t_item[1]))
            f_ids = ([d_item[4]] * len(d_item[1])
                     + [-1]
                     + [t_item[4]] * len(t_item[1]))

            # Add text
            text_str = format_template(fact["entities"], fact["roles"], variant, fact["rel"])
            sep_and_text = " | " + text_str
            text_tokens = self.vocab.encode_sentence(sep_and_text)
            seq += text_tokens
            e_lens.append(len(text_tokens))
            r_slots.append(TEXT_SLOT)
            r_names.append(fact["rel"])
            f_ids.append(-1)

            if len(seq) > self.block_size:
                seq = seq[:self.block_size]
                total = 0
                new_lens = []
                new_slots = []
                new_names = []
                new_fids = []
                for el, rs, rn, fi in zip(e_lens, r_slots, r_names, f_ids):
                    if total + el <= len(seq):
                        new_lens.append(el)
                        new_slots.append(rs)
                        new_names.append(rn)
                        new_fids.append(fi)
                        total += el
                    else:
                        remaining = len(seq) - total
                        if remaining > 0:
                            new_lens.append(remaining)
                            new_slots.append(rs)
                            new_names.append(rn)
                            new_fids.append(fi)
                        break
                e_lens = new_lens
                r_slots = new_slots
                r_names = new_names
                f_ids = new_fids

            seqs.append(seq)
            entity_lens_list.append(e_lens)
            role_slots_list.append(r_slots)
            rel_names.append(r_names)
            fact_ids_list.append(f_ids)

        max_len = max(len(s) for s in seqs)
        x = torch.full((batch_size, max_len - 1), self.vocab.PAD, dtype=torch.long)
        y = torch.full((batch_size, max_len - 1), -100, dtype=torch.long)
        for i, seq in enumerate(seqs):
            seq_t = torch.tensor(seq, dtype=torch.long)
            seq_len = len(seq) - 1
            x[i, :seq_len] = seq_t[:-1]
            y[i, :seq_len] = seq_t[1:]

        for i in range(batch_size):
            total = sum(entity_lens_list[i])
            if total > max_len - 1:
                diff = total - (max_len - 1)
                entity_lens_list[i][-1] -= diff
                if entity_lens_list[i][-1] <= 0:
                    entity_lens_list[i].pop()
                    role_slots_list[i].pop()
                    rel_names[i].pop()
                    fact_ids_list[i].pop()

        return (x.to(device), y.to(device),
                entity_lens_list, rel_names, role_slots_list, fact_ids_list)


# ============================================================================
# Rotation Utilities
# ============================================================================

def apply_rotation(x, angles):
    B, T, C = x.shape
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    x_pairs = x.reshape(B, T, C // 2, 2)
    x_even = x_pairs[..., 0]
    x_odd = x_pairs[..., 1]
    r_even = x_even * cos_a - x_odd * sin_a
    r_odd = x_even * sin_a + x_odd * cos_a
    result = torch.stack([r_even, r_odd], dim=-1)
    return result.reshape(B, T, C)


def apply_inverse_rotation(x, angles):
    return apply_rotation(x, -angles)


# ============================================================================
# Shared Building Blocks
# ============================================================================

class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.2):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class RotaryAttention(nn.Module):
    def __init__(self, n_embed, block_size, dropout=0.2, rotate_v=False, use_softmax=False):
        super().__init__()
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.n_embed = n_embed
        self.rotate_v = rotate_v
        self.use_softmax = use_softmax
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, angles, causal=True, pad_mask=None):
        B, T, C = x.shape
        k = self.keys(x)
        q = self.queries(x)
        v = self.values(x)
        k = apply_rotation(k, angles)
        q = apply_rotation(q, angles)
        if self.rotate_v:
            v = apply_rotation(v, angles)
        wei = k @ q.transpose(-1, -2) * C ** (-0.5)
        if causal:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        if pad_mask is not None:
            pad_mask_k = pad_mask.unsqueeze(1)
            wei = wei.masked_fill(~pad_mask_k, float('-inf'))
        if self.use_softmax:
            wei = F.softmax(wei, dim=-1)
        else:
            wei = torch.log(torch.exp(wei) + 1)
            wei = wei / (wei.sum(dim=-1, keepdim=True) + 1e-6)
        wei = self.dropout(wei)
        out = wei @ v
        if self.rotate_v:
            out = apply_inverse_rotation(out, angles)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, n_embed, block_size, dropout=0.2, rotate_v=False, use_softmax=False):
        super().__init__()
        self.sa_head = RotaryAttention(n_embed, block_size, dropout, rotate_v, use_softmax)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, angles, causal=True, pad_mask=None):
        x = x + self.sa_head(self.ln1(x), angles, causal, pad_mask)
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================================
# Model Ba/Bas: ALiBi (linear attention bias, no rotation)
# ============================================================================

class ALiBiAttention(nn.Module):
    """Attention with ALiBi positional bias (no rotation)."""

    def __init__(self, n_embed, block_size, dropout=0.2, rotate_v=False,
                 use_softmax=False, slope=0.004):
        super().__init__()
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.n_embed = n_embed
        self.use_softmax = use_softmax
        self.slope = slope
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, angles=None, causal=True, pad_mask=None):
        B, T, C = x.shape
        k = self.keys(x)
        q = self.queries(x)
        v = self.values(x)
        wei = k @ q.transpose(-1, -2) * C ** (-0.5)
        # ALiBi bias: -slope * |i - j|
        pos = torch.arange(T, device=x.device)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).float().abs()
        wei = wei - self.slope * dist.unsqueeze(0)
        if causal:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        if pad_mask is not None:
            pad_mask_k = pad_mask.unsqueeze(1)
            wei = wei.masked_fill(~pad_mask_k, float('-inf'))
        # ALiBi always uses softmax (the linear bias is designed for it)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        out = self.proj(out)
        out = self.dropout(out)
        return out


class ALiBiBlock(nn.Module):
    def __init__(self, n_embed, block_size, dropout=0.2, rotate_v=False,
                 use_softmax=False, slope=0.004):
        super().__init__()
        self.sa_head = ALiBiAttention(n_embed, block_size, dropout,
                                      rotate_v, use_softmax, slope)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, angles=None, causal=True, pad_mask=None):
        x = x + self.sa_head(self.ln1(x), causal=causal, pad_mask=pad_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class ModelBa(nn.Module):
    """ALiBi model: no rotation, linear distance bias with configurable slope."""

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout=0.2,
                 rotate_v=False, use_softmax=False, slope=0.004):
        super().__init__()
        self.n_embed = n_embed
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList([
            ALiBiBlock(n_embed, block_size, dropout, rotate_v, use_softmax, slope)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pad_mask = (idx != 0)
        x = self.token_embedding(idx)
        for block in self.blocks:
            x = block(x, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward(idx)
        return logits


# ============================================================================
# Model B/B': Standard RoPE
# ============================================================================

class ModelB(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout=0.2,
                 rotate_v=False, use_softmax=False):
        super().__init__()
        self.n_embed = n_embed
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.register_buffer('base_freq',
            1.0 / (10000 ** (torch.arange(0, n_embed // 2).float() / (n_embed // 2))))
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v, use_softmax)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def _rope_angles(self, T, device):
        positions = torch.arange(T, device=device, dtype=torch.float)
        angles = torch.outer(positions, self.base_freq)
        return angles.unsqueeze(0)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pad_mask = (idx != 0)
        x = self.token_embedding(idx)
        angles = self._rope_angles(T, idx.device).expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward(idx)
        return logits

    def forward_segmented(self, idx, targets, segment_lens):
        """Forward with position resets at segment boundaries (for primed eval)."""
        B, T = idx.shape
        pad_mask = (idx != 0)
        x = self.token_embedding(idx)
        # Build positions that reset at each segment boundary
        pos_np = np.zeros((B, T), dtype=np.float32)
        for bi in range(B):
            pos = 0
            for seg_len in segment_lens[bi]:
                end = min(pos + seg_len, T)
                n = end - pos
                if n > 0:
                    pos_np[bi, pos:end] = np.arange(n, dtype=np.float32)
                pos = end
                if pos >= T:
                    break
        positions = torch.from_numpy(pos_np).to(idx.device)
        angles = positions.unsqueeze(-1) * self.base_freq
        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss


# ============================================================================
# Model Bl/Bl': Linearized B + LISformer content-based segment addressing
# ============================================================================

class ModelBl(nn.Module):
    """Like ModelB but with content-based segment addressing (LISformer).

    Uses B-style linearized data. RoPE positions reset at segment boundaries.
    Each completed segment gets a content-based angle offset via rotate
    embeddings by within-segment RoPE -> scatter-add per segment -> mean-pool
    -> LayerNorm -> Linear -> angle offset. Current (last) segment gets zero
    offset.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout=0.2,
                 rotate_v=False, use_softmax=False):
        super().__init__()
        self.n_embed = n_embed
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.register_buffer('base_freq',
            1.0 / (10000 ** (torch.arange(0, n_embed // 2).float() / (n_embed // 2))))
        self.line_ln = nn.LayerNorm(n_embed)
        self.line_proj = nn.Linear(n_embed, n_embed // 2)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v, use_softmax)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def _rope_angles(self, T, device):
        positions = torch.arange(T, device=device, dtype=torch.float)
        angles = torch.outer(positions, self.base_freq)
        return angles.unsqueeze(0)

    def forward(self, idx, targets=None):
        """Standard continuous RoPE forward (for unprimed text eval)."""
        B, T = idx.shape
        pad_mask = (idx != 0)
        x = self.token_embedding(idx)
        angles = self._rope_angles(T, idx.device).expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward(idx)
        return logits

    def forward_content_addressed(self, idx, targets, segment_lens):
        """Forward with content-based segment addressing."""
        B, T = idx.shape
        device = idx.device
        pad_mask = (idx != 0)
        x = self.token_embedding(idx)

        # Build segment_ids and pos_in_segment from segment_lens
        seg_ids_np = np.zeros((B, T), dtype=np.int64)
        pos_in_seg_np = np.zeros((B, T), dtype=np.float32)
        for bi in range(B):
            pos = 0
            for seg_idx, seg_len in enumerate(segment_lens[bi]):
                end = min(pos + seg_len, T)
                n = end - pos
                if n > 0:
                    seg_ids_np[bi, pos:end] = seg_idx
                    pos_in_seg_np[bi, pos:end] = np.arange(n, dtype=np.float32)
                pos = end
                if pos >= T:
                    break

        seg_ids = torch.from_numpy(seg_ids_np).to(device)
        pos_in_seg = torch.from_numpy(pos_in_seg_np).to(device)
        rope = pos_in_seg.unsqueeze(-1) * self.base_freq  # (B, T, C//2)

        # Content-based segment offset: rotate -> scatter-add -> mean-pool -> LN -> proj
        rotated = apply_rotation(x, rope)
        n_segs = seg_ids.max().item() + 1
        sid_exp = seg_ids.unsqueeze(-1).expand(-1, -1, self.n_embed)
        sums = torch.zeros(B, n_segs, self.n_embed, device=device)
        sums.scatter_add_(1, sid_exp, rotated)
        counts = torch.zeros(B, n_segs, 1, device=device)
        counts.scatter_add_(1, seg_ids.unsqueeze(-1),
                            torch.ones(B, T, 1, device=device))
        pooled = sums / counts.clamp(min=1)

        seg_angles = self.line_proj(self.line_ln(pooled))  # (B, n_segs, C//2)
        offset = seg_angles.gather(
            1, seg_ids.unsqueeze(-1).expand(-1, -1, self.n_embed // 2))

        # Zero out current (last) segment — incomplete
        max_seg = seg_ids.max(dim=1, keepdim=True).values
        offset = offset * (seg_ids != max_seg).unsqueeze(-1).float()

        angles = rope + offset
        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss


# ============================================================================
# Model J/J': RoPE + Per-Relation Slot Angles
# ============================================================================

class ModelJ(nn.Module):
    """Native KG: entity characters only; relation and role enter through
    slot angles; each fact carries a fact address on every one of its characters.

    address_mode:
      "content"  address = LayerNorm -> Linear of the mean of the fact's rotated
                 token embeddings (J)
      "random"   one i.i.d. N(0,1) angle vector per fact, drawn each forward (Jr)
      "none"     no fact address; reset RoPE + slot angles only (J0)
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, n_relations=None,
                 dropout=0.2, rotate_v=False, use_softmax=False, address_mode="content"):
        super().__init__()
        assert address_mode in ("content", "random", "none")
        self.address_mode = address_mode
        if n_relations is None:
            n_relations = len(KG_RELATIONS)
        self.n_embed = n_embed
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.slot_angles = nn.Parameter(
            torch.randn(n_relations, MAX_ARITY + 1, n_embed // 2) * 0.1)
        with torch.no_grad():
            self.slot_angles[:, MAX_ARITY, :] = 0.0
        self.register_buffer('base_freq',
            1.0 / (10000 ** (torch.arange(0, n_embed // 2).float() / (n_embed // 2))))
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v, use_softmax)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.rel_to_idx = {rel: i for i, rel in enumerate(KG_RELATIONS)}

        # Fact embedding: pool entity token embeddings → normalize → project to angle space
        self.fact_norm = nn.LayerNorm(n_embed)
        self.fact_proj = nn.Linear(n_embed, n_embed // 2)

    def _rope_angles(self, T, device):
        positions = torch.arange(T, device=device, dtype=torch.float)
        angles = torch.outer(positions, self.base_freq)
        return angles.unsqueeze(0)

    def _kg_angles(self, entity_lens, seq_len, batch_size, rel_names, device,
                   role_slots=None, fact_ids=None, tok_emb=None):
        # rel_names: list of lists [batch_size][n_entities] — one relation per entity chunk
        # fact_ids: list of lists [batch_size][n_entities] — which fact each entity belongs to
        #   fact_id >= 0 for KG entities, -1 for text/separators (no fact offset)
        # tok_emb: pre-computed token embeddings [B, T, n_embed] (avoids redundant embedding lookup)
        #
        # Build on CPU with numpy slice assignments, then transfer once to GPU
        slot_ids_np = np.zeros((batch_size, seq_len), dtype=np.int64)
        pos_in_slot_np = np.zeros((batch_size, seq_len), dtype=np.int64)
        rel_ids_np = np.zeros((batch_size, seq_len), dtype=np.int64)
        fact_ids_np = np.full((batch_size, seq_len), -1, dtype=np.int64)
        for bi in range(batch_size):
            pos = 0
            for slot_idx, ent_len in enumerate(entity_lens[bi]):
                actual_slot = role_slots[bi][slot_idx] if role_slots is not None else slot_idx
                end = min(pos + ent_len, seq_len)
                n = end - pos
                if n > 0:
                    slot_ids_np[bi, pos:end] = actual_slot
                    pos_in_slot_np[bi, pos:end] = np.arange(n)
                    rel_ids_np[bi, pos:end] = self.rel_to_idx[rel_names[bi][slot_idx]]
                    if fact_ids is not None:
                        fact_ids_np[bi, pos:end] = fact_ids[bi][slot_idx]
                pos = end
                if pos >= seq_len:
                    break
        slot_ids = torch.from_numpy(slot_ids_np).to(device)
        pos_in_slot = torch.from_numpy(pos_in_slot_np).to(device)
        rel_ids = torch.from_numpy(rel_ids_np).to(device)
        rope = pos_in_slot.unsqueeze(-1).float() * self.base_freq
        slot_offset = self.slot_angles[rel_ids, slot_ids]

        if self.address_mode == "none" or fact_ids is None:
            return rope + slot_offset

        if self.address_mode == "random":
            # One random angle vector per fact; text/separators (fact_id -1) get none.
            fact_ids_t = torch.from_numpy(fact_ids_np).to(device)
            n_facts = int(fact_ids_np.max()) + 1
            kg_mask = (fact_ids_t >= 0)
            rand = torch.randn(batch_size, max(n_facts, 1), self.n_embed // 2, device=device)
            fid_safe = fact_ids_t.clamp(min=0)
            fact_offset = torch.gather(rand, 1, fid_safe.unsqueeze(-1).expand(-1, -1, self.n_embed // 2))
            fact_offset = fact_offset * kg_mask.unsqueeze(-1).float()
            return rope + slot_offset + fact_offset

        # Content address: compute fact embeddings if fact_ids and tok_emb are provided
        if fact_ids is not None and tok_emb is not None:
            fact_ids_t = torch.from_numpy(fact_ids_np).to(device)  # [B, T]
            full_angles = rope + slot_offset  # [B, T, n_embed//2]
            max_fid = fact_ids_np.max()
            n_facts = max_fid + 1  # number of distinct facts (0..max_fid)

            kg_mask = (fact_ids_t >= 0)  # [B, T]
            rotated = apply_rotation(tok_emb, full_angles)  # [B, T, n_embed]

            # Vectorized: scatter-add rotated embeddings per (batch, fact_id),
            # then gather back to each token's position
            # Use fact_ids_t clamped to 0 for non-KG tokens (they'll be masked out)
            fid_safe = fact_ids_t.clamp(min=0)  # [B, T]

            # Scatter-add per fact: sum rotated embeddings
            # Index: [B, T] -> [B, n_facts, n_embed]
            sums = torch.zeros(batch_size, n_facts, self.n_embed, device=device)
            counts = torch.zeros(batch_size, n_facts, 1, device=device)
            fid_expand = fid_safe.unsqueeze(-1).expand(-1, -1, self.n_embed)  # [B, T, n_embed]
            masked_rotated = rotated * kg_mask.unsqueeze(-1).float()
            sums.scatter_add_(1, fid_expand, masked_rotated)
            counts.scatter_add_(1, fid_safe.unsqueeze(-1), kg_mask.float().unsqueeze(-1))
            counts = counts.clamp(min=1)

            # Mean pool per fact, then project
            pooled = sums / counts  # [B, n_facts, n_embed]
            # Reshape for LayerNorm + Linear
            pooled_flat = pooled.reshape(-1, self.n_embed)  # [B*n_facts, n_embed]
            projected_flat = self.fact_proj(self.fact_norm(pooled_flat))  # [B*n_facts, n_embed//2]
            projected = projected_flat.reshape(batch_size, n_facts, self.n_embed // 2)

            # Gather back: each token gets its fact's projected embedding
            fid_expand_half = fid_safe.unsqueeze(-1).expand(-1, -1, self.n_embed // 2)
            fact_offset = torch.gather(projected, 1, fid_expand_half)  # [B, T, n_embed//2]
            # Zero out non-KG positions
            fact_offset = fact_offset * kg_mask.unsqueeze(-1).float()

            return rope + slot_offset + fact_offset
        else:
            return rope + slot_offset

    def forward_text(self, idx, targets=None):
        B, T = idx.shape
        pad_mask = (idx != 0)
        x = self.token_embedding(idx)
        angles = self._rope_angles(T, idx.device).expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def forward_kg(self, char_tokens, targets, entity_lens, rel_names,
                   role_slots=None, negate_angles=None, fact_ids=None):
        B, T = char_tokens.shape
        pad_mask = (char_tokens != 0)
        x = self.token_embedding(char_tokens)
        angles = self._kg_angles(entity_lens, T, B, rel_names, char_tokens.device,
                                 role_slots=role_slots, fact_ids=fact_ids,
                                 tok_emb=x)
        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def forward_paired(self, idx, targets, entity_lens, rel_names, role_slots,
                       fact_ids=None):
        B, T = idx.shape
        pad_mask = (idx != 0)
        x = self.token_embedding(idx)
        angles = self._kg_angles(entity_lens, T, B, rel_names, idx.device,
                                 role_slots=role_slots, fact_ids=fact_ids,
                                 tok_emb=x)
        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward_text(idx)
        return logits


# ============================================================================
# Training
# ============================================================================

def train_model_paired(model, dataset, config, model_type="B", name="?",
                       checkpoint_path=None):
    """Train on paired KG+text sequences."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    model.to(config.device)
    model.train()
    losses_log = {"loss": [], "iter": []}

    for it in tqdm(range(config.max_iters), desc=f"Model {name}"):
        if model_type == "B":
            x, y, _segment_lens = dataset.get_paired_batch_B(config.batch_size, config.device)
            _, loss = model.forward(x, y)
        elif model_type == "Ba":
            x, y, _segment_lens = dataset.get_paired_batch_B(config.batch_size, config.device)
            _, loss = model.forward(x, y)
        elif model_type == "Bs":
            x, y, segment_lens = dataset.get_paired_batch_B(config.batch_size, config.device)
            _, loss = model.forward_segmented(x, y, segment_lens)
        elif model_type == "Bl":
            x, y, segment_lens = dataset.get_paired_batch_B(config.batch_size, config.device)
            _, loss = model.forward_content_addressed(x, y, segment_lens)
        elif model_type == "J":
            x, y, entity_lens, rel_names, role_slots, fact_ids = dataset.get_paired_batch_J(
                config.batch_size, config.device)
            _, loss = model.forward_paired(x, y, entity_lens, rel_names, role_slots,
                                           fact_ids=fact_ids)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % config.eval_interval == 0:
            losses_log["iter"].append(it)
            losses_log["loss"].append(loss.item())
            print(f"  [{name}] iter {it}, loss: {loss.item():.4f}")

    # Save checkpoint
    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  [{name}] Checkpoint saved: {checkpoint_path}")

    return losses_log


# ============================================================================
# Priming Utilities
# ============================================================================

def _collect_facts_by_rel(chains, domain_name):
    """Group all (rel, entities) by relation across all chain positions.

    Returns: dict {rel: [(entities, pos), ...]}
    """
    chain = chains[domain_name]
    n = len(chain)
    rel_facts = defaultdict(list)

    # chain2
    for pos in range(n - 1):
        A, B = chain[pos], chain[pos + 1]
        for pi in range(5):
            rel = f"chain2_{domain_name}_p{pi}"
            if rel in RELATION_SCHEMA:
                rel_facts[rel].append(([A, B], pos))

    # chain3
    for pos in range(n - 2):
        A, B, C = chain[pos], chain[pos + 1], chain[pos + 2]
        for wi in range(len(CHAIN3_WINDOWS)):
            rel = f"chain3_{domain_name}_w{wi}"
            if rel in RELATION_SCHEMA:
                rel_facts[rel].append(([A, B, C], pos))

    return rel_facts


def linearize_kg_fact_permuted(rel, entities):
    """Linearize a KG fact with random entity permutation for priming."""
    schema = RELATION_SCHEMA[rel]
    roles = schema["roles"]
    arity = len(entities)
    perm = list(range(arity))
    random.shuffle(perm)
    parts = []
    for idx in perm:
        parts.append(f"<{roles[idx]}> {entities[idx]}")
    return " ".join(parts)


# ============================================================================
# Primed Evaluation Prompts
# ============================================================================

def _build_native_priming_tokens(priming_facts, vocab):
    """Build native KG priming tokens for J model (KG prefix only, no text).

    priming_facts: list of (rel, entities) tuples
    Returns: (kg_tokens, entity_lens, role_slots, rel_names, fact_ids)
    """
    space_id = vocab.char2idx[" "]
    all_tokens = []
    entity_lens = []
    role_slots = []
    rel_names = []
    fact_ids = []

    for fact_idx, (priming_rel, priming_entities) in enumerate(priming_facts):
        arity = len(priming_entities)
        perm = list(range(arity))
        random.shuffle(perm)

        for orig_idx in perm:
            ent_tokens = vocab.encode_entity(priming_entities[orig_idx])
            full_tokens = [space_id] + ent_tokens + [space_id]
            all_tokens.extend(full_tokens)
            entity_lens.append(len(full_tokens))
            role_slots.append(orig_idx)
            rel_names.append(priming_rel)
            fact_ids.append(fact_idx)

        # Separator " | " between facts
        if fact_idx < len(priming_facts) - 1:
            sep_tokens = vocab.encode_sentence(" | ")
            all_tokens.extend(sep_tokens)
            entity_lens.append(len(sep_tokens))
            role_slots.append(MAX_ARITY)  # TEXT_SLOT
            rel_names.append(priming_rel)
            fact_ids.append(-1)

    return (all_tokens, entity_lens, role_slots, rel_names, fact_ids)


def _build_linearized_priming_tokens(priming_facts, vocab):
    """Build linearized KG priming tokens for B model (KG prefix only, no text).

    Each priming fact is one segment; separators between facts are their own segments.

    Returns: (kg_tokens, prefix_segment_lens)
    """
    all_tokens = []
    prefix_segment_lens = []

    for fact_idx, (priming_rel, priming_entities) in enumerate(priming_facts):
        fact_str = linearize_kg_fact_permuted(priming_rel, priming_entities)
        fact_tokens = vocab.encode_sentence(fact_str)
        all_tokens.extend(fact_tokens)
        prefix_segment_lens.append(len(fact_tokens))

        if fact_idx < len(priming_facts) - 1:
            sep_tokens = vocab.encode_sentence(" | ")
            all_tokens.extend(sep_tokens)
            prefix_segment_lens.append(len(sep_tokens))

    return (all_tokens, prefix_segment_lens)


def _build_primed_prompt(rel, true_entities, priming_pairs, variant, vocab,
                         model_type, name_set_label, priming_mode):
    """Build one primed eval prompt (shared by same_rel and same_entities)."""
    roles = RELATION_SCHEMA[rel]["roles"]
    fact_type = "chain3" if RELATION_SCHEMA[rel]["arity"] == 3 else "chain2"
    prefix_text, segments = build_template_segments(
        true_entities, roles, variant, vocab, rel)
    sep_and_prefix = " | " + prefix_text
    prefix_tokens = vocab.encode_sentence(sep_and_prefix)
    target_len = sum(len(s["tokens"]) for s in segments if s["score"])
    target_entities = [true_entities[p] for p in variant["pred_indices"]]

    if model_type == "J":
        (kg_tokens, p_elens, p_rslots, p_rnames,
         p_fids) = _build_native_priming_tokens(priming_pairs, vocab)
        prompt_tokens = kg_tokens + prefix_tokens
        return {
            "direction": "kg_to_text",
            "fact_type": fact_type,
            "name_set": name_set_label,
            "priming_mode": priming_mode,
            "relation": rel,
            "target": "/".join(target_entities),
            "prompt_tokens": prompt_tokens,
            "segments": segments,
            "target_len": target_len,
            "forward_mode": "paired",
            "j_prefix_entity_lens": p_elens,
            "j_prefix_role_slots": p_rslots,
            "j_growing_slot": MAX_ARITY,
            "j_initial_growing_len": len(prefix_tokens),
            "j_prefix_rel_names": p_rnames,
            "j_growing_rel": rel,
            "j_prefix_fact_ids": p_fids,
        }
    else:
        (kg_tokens, p_seglens) = _build_linearized_priming_tokens(
            priming_pairs, vocab)
        prompt_tokens = kg_tokens + prefix_tokens
        return {
            "direction": "kg_to_text",
            "fact_type": fact_type,
            "name_set": name_set_label,
            "priming_mode": priming_mode,
            "relation": rel,
            "target": "/".join(target_entities),
            "prompt_tokens": prompt_tokens,
            "segments": segments,
            "target_len": target_len,
            "forward_mode": "text",
            "b_prefix_segment_lens": p_seglens,
            "b_initial_growing_len": len(prefix_tokens),
        }


def build_primed_eval_prompts(chains, vocab, n_priming=4, name_set_label="test",
                              model_type="B"):
    """Build primed evaluation prompts with two modes:

    same_rel: 1 true + n_priming distractors of same relation, different entities.
        Model must match the given entity to the correct KG fact.
    same_entities: 1 true + n_priming distractors with same entities, different relations.
        Model must match the role name in text to the correct KG relation.

    Uses multi-target segment scoring (same as unprimed).
    """
    prompts = []

    for domain_name in CHAIN_ROLES.keys():
        rel_facts = _collect_facts_by_rel(chains, domain_name)

        # --- same_rel: same relation, different entities ---
        for rel, facts_list in rel_facts.items():
            if len(facts_list) < n_priming + 1:
                continue

            for true_idx, (true_entities, true_pos) in enumerate(facts_list):
                candidates = [(ents, p) for i, (ents, p) in enumerate(facts_list)
                              if i != true_idx]
                random.shuffle(candidates)
                distractors = candidates[:n_priming]
                if len(distractors) < n_priming:
                    continue

                distractor_pairs = [(rel, ents) for ents, _ in distractors]
                random.shuffle(distractor_pairs)
                priming_pairs = _place_true_fact((rel, true_entities), distractor_pairs)

                variant = TEMPLATE_VARIANTS[rel][0]
                prompts.append(_build_primed_prompt(
                    rel, true_entities, priming_pairs, variant, vocab,
                    model_type, name_set_label, "same_rel"))

    # --- same_entities: same entities, different relations (cross-domain) ---
    # For each true fact, construct distractors by pairing the same entities
    # with relations of the same arity from OTHER domains.
    # Group all relations by arity for distractor selection
    rels_by_arity = defaultdict(list)
    for rel, schema in RELATION_SCHEMA.items():
        rels_by_arity[schema["arity"]].append(rel)

    for domain_name in CHAIN_ROLES.keys():
        rel_facts = _collect_facts_by_rel(chains, domain_name)
        for rel, facts_list in rel_facts.items():
            arity = RELATION_SCHEMA[rel]["arity"]
            # Get all relations of same arity from OTHER domains as distractors
            other_rels = [r for r in rels_by_arity[arity]
                          if _get_domain(r) != domain_name]
            if len(other_rels) < n_priming:
                continue

            for true_entities, true_pos in facts_list:
                random.shuffle(other_rels)
                distractor_rels = other_rels[:n_priming]

                # Build distractor KG facts: same entities, different relations
                # Map entities to distractor relation's roles
                distractor_pairs = [(d_rel, true_entities) for d_rel in distractor_rels]
                random.shuffle(distractor_pairs)
                priming_pairs = _place_true_fact((rel, true_entities), distractor_pairs)

                variant = TEMPLATE_VARIANTS[rel][0]
                prompts.append(_build_primed_prompt(
                    rel, true_entities, priming_pairs, variant, vocab,
                    model_type, name_set_label, "same_entities"))

    return prompts


# ============================================================================
# Zero-Shot Evaluation
# ============================================================================

def _build_kg_tokens_B(entities, roles, vocab):
    """Build linearized KG tokens with shuffled entity order."""
    arity = len(entities)
    perm = list(range(arity))
    random.shuffle(perm)
    parts = []
    for orig_idx in perm:
        parts.append(f"<{roles[orig_idx]}> {entities[orig_idx]}")
    kg_str = " ".join(parts)
    return vocab.encode_sentence(kg_str), kg_str


def _build_kg_tokens_J(entities, roles, rel, vocab):
    """Build native KG tokens with shuffled entity order for J."""
    arity = len(entities)
    perm = list(range(arity))
    random.shuffle(perm)
    space_id = vocab.char2idx[" "]
    kg_tokens = []
    kg_lens = []
    for orig_idx in perm:
        ent_tokens = vocab.encode_entity(entities[orig_idx])
        full_tokens = [space_id] + ent_tokens + [space_id]
        kg_tokens.extend(full_tokens)
        kg_lens.append(len(full_tokens))
    return kg_tokens, kg_lens, perm


def _build_eval_prompt(rel, entities, fact_type, variant, vocab, model_type,
                       name_set_label, priming_mode="none"):
    """Build one eval prompt with multi-target segments."""
    roles = RELATION_SCHEMA[rel]["roles"]
    prefix_text, segments = build_template_segments(entities, roles, variant, vocab, rel)
    sep_and_prefix = " | " + prefix_text
    prefix_tokens = vocab.encode_sentence(sep_and_prefix)

    # Total scored chars
    target_len = sum(len(s["tokens"]) for s in segments if s["score"])
    target_entities = [entities[p] for p in variant["pred_indices"]]

    if model_type == "J":
        kg_tokens, kg_lens, perm = _build_kg_tokens_J(entities, roles, rel, vocab)
        prompt_tokens = kg_tokens + prefix_tokens
        return {
            "direction": "kg_to_text",
            "fact_type": fact_type,
            "name_set": name_set_label,
            "priming_mode": priming_mode,
            "relation": rel,
            "target": "/".join(target_entities),
            "prompt_tokens": prompt_tokens,
            "segments": segments,
            "target_len": target_len,
            "forward_mode": "paired",
            "j_prefix_entity_lens": list(kg_lens),
            "j_prefix_role_slots": list(perm),
            "j_growing_slot": MAX_ARITY,
            "j_initial_growing_len": len(prefix_tokens),
            "j_prefix_rel_names": [rel] * len(kg_lens),
            "j_growing_rel": rel,
            "j_prefix_fact_ids": [0] * len(kg_lens),
        }
    else:
        kg_tokens, kg_str = _build_kg_tokens_B(entities, roles, vocab)
        prompt_tokens = kg_tokens + prefix_tokens
        return {
            "direction": "kg_to_text",
            "fact_type": fact_type,
            "name_set": name_set_label,
            "priming_mode": priming_mode,
            "relation": rel,
            "target": "/".join(target_entities),
            "prompt_tokens": prompt_tokens,
            "segments": segments,
            "target_len": target_len,
            "forward_mode": "text",
            "b_prefix_segment_lens": [len(kg_tokens)],
            "b_initial_growing_len": len(prefix_tokens),
        }


def build_zero_shot_eval_prompts(chains, vocab, model_type, name_set_label="test",
                                 n_eval_skeletons=1):
    """Build evaluation prompts for zero-shot testing (unprimed).

    Samples n_eval_skeletons random skeletons per fact (with all given/pred perms
    for each skeleton). Set n_eval_skeletons=0 to use all skeletons.

    Uses a fixed seed for skeleton sampling so B and J get the same skeletons.
    """
    skel_rng = random.Random(42)
    prompts = []

    for domain_name in CHAIN_ROLES.keys():
        chain = chains[domain_name]
        n = len(chain)

        # chain2 facts
        for pos in range(n - 1):
            A, B = chain[pos], chain[pos + 1]
            for pi in range(5):
                rel = f"chain2_{domain_name}_p{pi}"
                if rel not in RELATION_SCHEMA:
                    continue
                variants = TEMPLATE_VARIANTS[rel]
                if n_eval_skeletons > 0:
                    n_skels = len(SKELETONS_2ARY)
                    chosen = skel_rng.sample(range(n_skels), min(n_eval_skeletons, n_skels))
                    variants = [v for v in variants if v["skeleton_idx"] in chosen]
                for variant in variants:
                    prompts.append(_build_eval_prompt(
                        rel, [A, B], "chain2", variant, vocab, model_type,
                        name_set_label))

        # chain3 facts
        for pos in range(n - 2):
            A, B, C = chain[pos], chain[pos + 1], chain[pos + 2]
            for wi in range(len(CHAIN3_WINDOWS)):
                rel = f"chain3_{domain_name}_w{wi}"
                if rel not in RELATION_SCHEMA:
                    continue
                variants = TEMPLATE_VARIANTS[rel]
                if n_eval_skeletons > 0:
                    n_skels = len(SKELETONS_3ARY)
                    chosen = skel_rng.sample(range(n_skels), min(n_eval_skeletons, n_skels))
                    variants = [v for v in variants if v["skeleton_idx"] in chosen]
                for variant in variants:
                    prompts.append(_build_eval_prompt(
                        rel, [A, B, C], "chain3", variant, vocab, model_type,
                        name_set_label))

    return prompts


def evaluate_zero_shot(model, eval_prompts, vocab, config, model_name="?", model_type="B"):
    """Evaluate with segment-aware multi-target scoring.

    Each prompt has a 'segments' list: alternating scored (entity) and unscored (template) parts.
    For each step, we teacher-force the correct token and score only entity tokens.

    h@1: all scored characters predicted exactly (argmax)
    h@5: all scored characters in top-5
    PPL: exp(-total_scored_log_probs / total_scored_chars)

    Returns summary dict: {(direction, fact_type, name_set, priming_mode): {h@1, h@5, ppl, n}}
    """
    model.eval()
    model.to(config.device)

    results = defaultdict(list)

    # Flatten all segment tokens into a single target sequence per prompt,
    # with a parallel mask indicating which positions to score.
    processed = []
    for p in eval_prompts:
        prompt_tokens = p["prompt_tokens"]
        segments = p["segments"]

        # Build flat target sequence and score mask
        flat_targets = []
        score_mask = []
        for seg in segments:
            flat_targets.extend(seg["tokens"])
            score_mask.extend([seg["score"]] * len(seg["tokens"]))

        total_target_len = len(flat_targets)
        scored_len = sum(score_mask)

        if len(prompt_tokens) + total_target_len > config.block_size:
            # Truncate prompt from left if needed
            max_prompt = config.block_size - total_target_len
            if max_prompt < 1:
                continue  # skip if even truncation can't fit
            prompt_tokens = prompt_tokens[-max_prompt:]

        fmode = p.get("forward_mode", "text")
        if model_type == "Bs" and "b_prefix_segment_lens" in p:
            fmode = "segmented"
        elif model_type == "Bl" and "b_prefix_segment_lens" in p:
            fmode = "content_addressed"

        item = {
            "prompt_tokens": prompt_tokens,
            "flat_targets": flat_targets,
            "score_mask": score_mask,
            "prompt_len": len(prompt_tokens),
            "total_target_len": total_target_len,
            "scored_len": scored_len,
            "direction": p["direction"],
            "fact_type": p["fact_type"],
            "name_set": p["name_set"],
            "priming_mode": p.get("priming_mode", "none"),
            "relation": p["relation"],
            "target": p["target"],
            "forward_mode": fmode,
        }
        if model_type == "J" and "j_prefix_entity_lens" in p:
            item["j_prefix_entity_lens"] = p["j_prefix_entity_lens"]
            item["j_prefix_role_slots"] = p["j_prefix_role_slots"]
            item["j_growing_slot"] = p["j_growing_slot"]
            item["j_initial_growing_len"] = p["j_initial_growing_len"]
            item["j_prefix_rel_names"] = p["j_prefix_rel_names"]
            item["j_growing_rel"] = p["j_growing_rel"]
            item["j_prefix_fact_ids"] = p.get("j_prefix_fact_ids")
            item["forward_mode"] = "paired"
        elif model_type == "Bs" and "b_prefix_segment_lens" in p:
            item["b_prefix_segment_lens"] = p["b_prefix_segment_lens"]
            item["b_initial_growing_len"] = p["b_initial_growing_len"]
            item["forward_mode"] = "segmented"
        elif model_type == "Bl" and "b_prefix_segment_lens" in p:
            item["b_prefix_segment_lens"] = p["b_prefix_segment_lens"]
            item["b_initial_growing_len"] = p["b_initial_growing_len"]
            item["forward_mode"] = "content_addressed"

        processed.append(item)

    # Group by (prompt_len, total_target_len, forward_mode) for batching
    groups = defaultdict(list)
    for idx, pp in enumerate(processed):
        groups[(pp["prompt_len"], pp["total_target_len"], pp["forward_mode"])].append(idx)

    n = len(processed)
    total_log_probs = [0.0] * n
    all_correct_top1 = [True] * n
    all_correct_top5 = [True] * n
    batch_size = config.batch_size

    with torch.no_grad():
        for (prompt_len, total_target_len, _fmode), prompt_indices in groups.items():
            if total_target_len == 0:
                continue
            current_tokens_map = {}
            for pi in prompt_indices:
                current_tokens_map[pi] = processed[pi]["prompt_tokens"].copy()

            for t_idx in range(total_target_len):
                input_len = min(prompt_len + t_idx, config.block_size)
                for sb_start in range(0, len(prompt_indices), batch_size):
                    sb_indices = prompt_indices[sb_start:sb_start + batch_size]
                    B = len(sb_indices)
                    x = torch.zeros(B, input_len, dtype=torch.long, device=config.device)
                    for bi, pi in enumerate(sb_indices):
                        toks = current_tokens_map[pi][-config.block_size:]
                        x[bi] = torch.tensor(toks, dtype=torch.long)

                    # Dispatch based on forward mode
                    first_pp = processed[sb_indices[0]]
                    fmode = first_pp["forward_mode"]
                    if fmode == "paired":
                        entity_lens_batch = []
                        role_slots_batch = []
                        rel_names_batch = []
                        fact_ids_batch = []
                        for bi, pi in enumerate(sb_indices):
                            pp = processed[pi]
                            growing_len = pp["j_initial_growing_len"] + t_idx
                            entity_lens_batch.append(pp["j_prefix_entity_lens"] + [growing_len])
                            role_slots_batch.append(pp["j_prefix_role_slots"] + [pp["j_growing_slot"]])
                            rel_names_batch.append(pp["j_prefix_rel_names"] + [pp["j_growing_rel"]])
                            pfids = pp.get("j_prefix_fact_ids")
                            if pfids is not None:
                                fact_ids_batch.append(pfids + [-1])
                            else:
                                fact_ids_batch.append(None)
                        if all(f is not None for f in fact_ids_batch):
                            pass_fact_ids = fact_ids_batch
                        else:
                            pass_fact_ids = None
                        logits, _ = model.forward_paired(
                            x, None, entity_lens_batch, rel_names_batch, role_slots_batch,
                            fact_ids=pass_fact_ids)
                    elif fmode == "segmented":
                        segment_lens_batch = []
                        for bi, pi in enumerate(sb_indices):
                            pp = processed[pi]
                            growing_len = pp["b_initial_growing_len"] + t_idx
                            segment_lens_batch.append(pp["b_prefix_segment_lens"] + [growing_len])
                        logits, _ = model.forward_segmented(
                            x, None, segment_lens_batch)
                    elif fmode == "content_addressed":
                        segment_lens_batch = []
                        for bi, pi in enumerate(sb_indices):
                            pp = processed[pi]
                            growing_len = pp["b_initial_growing_len"] + t_idx
                            segment_lens_batch.append(pp["b_prefix_segment_lens"] + [growing_len])
                        logits, _ = model.forward_content_addressed(
                            x, None, segment_lens_batch)
                    else:
                        logits = model.predict_text(x)

                    for bi, pi in enumerate(sb_indices):
                        pp = processed[pi]
                        t_tok = pp["flat_targets"][t_idx]
                        is_scored = pp["score_mask"][t_idx]
                        step_logits = logits[bi, -1, :]

                        if is_scored:
                            log_probs = F.log_softmax(step_logits, dim=0)
                            total_log_probs[pi] += log_probs[t_tok].item()

                            if all_correct_top1[pi]:
                                pred = torch.argmax(step_logits).item()
                                if pred != t_tok:
                                    all_correct_top1[pi] = False

                            if all_correct_top5[pi]:
                                top5 = torch.topk(step_logits, k=min(5, step_logits.shape[0])).indices.tolist()
                                if t_tok not in top5:
                                    all_correct_top5[pi] = False

                        # Teacher-force correct token regardless of scoring
                        current_tokens_map[pi].append(t_tok)

    # Aggregate results
    for pi, pp in enumerate(processed):
        scored_len = pp["scored_len"]
        if scored_len > 0:
            ppl = np.exp(-total_log_probs[pi] / scored_len)
        else:
            ppl = 1.0
        hit1 = 1 if all_correct_top1[pi] else 0
        hit5 = 1 if all_correct_top5[pi] else 0

        key = (pp["direction"], pp["fact_type"], pp["name_set"], pp["priming_mode"])
        results[key].append({
            "hit1": hit1,
            "hit5": hit5,
            "ppl": ppl,
            "target": pp["target"],
            "relation": pp["relation"],
        })

    # Summarize
    summary = {}
    for key, items in results.items():
        summary[key] = {
            "hit1": np.mean([r["hit1"] for r in items]),
            "hit5": np.mean([r["hit5"] for r in items]),
            "ppl": np.exp(np.mean([np.log(r["ppl"]) for r in items])),
            "n": len(items),
        }

    # Print grouped by priming mode
    priming_modes = sorted(set(k[3] for k in summary.keys()))
    for pm in priming_modes:
        pm_keys = sorted(k for k in summary.keys() if k[3] == pm)
        if not pm_keys:
            continue
        print(f"\n{'='*80}")
        if pm == "none":
            print(f"  Zero-Shot Evaluation: {model_name} (Unprimed)")
        else:
            print(f"  Zero-Shot Evaluation: {model_name} (Primed: {pm})")
        print(f"{'='*80}")
        print(f"  {'Direction':<15} {'FactType':<12} {'NameSet':<8} {'h@1':>6} {'h@5':>6} {'PPL':>8} {'n':>5}")
        print(f"  {'-'*70}")
        for key in pm_keys:
            direction, fact_type, name_set, _ = key
            s = summary[key]
            print(f"  {direction:<15} {fact_type:<12} {name_set:<8} "
                  f"{s['hit1']:>6.3f} {s['hit5']:>6.3f} {s['ppl']:>8.2f} {s['n']:>5d}")

    model.train()
    return summary, results


# ============================================================================
# Model Factory
# ============================================================================

MODEL_NAMES = ["B", "B'", "Bs", "Bs'", "Bl", "Bl'", "J", "J'", "Jr", "Jr'", "J0", "J0'"]


def create_model(name, vocab_size, config):
    n_e = config.n_embed
    n_l = config.n_layers
    bs = config.block_size
    d = config.dropout
    sm = config.use_softmax
    if name in ("B", "Bs"):
        return ModelB(vocab_size, n_e, n_l, bs, d, rotate_v=False, use_softmax=sm)
    elif name in ("B'", "Bs'"):
        return ModelB(vocab_size, n_e, n_l, bs, d, rotate_v=True, use_softmax=sm)
    elif name == "Ba":
        return ModelBa(vocab_size, n_e, n_l, bs, d, rotate_v=False, use_softmax=sm, slope=0.004)
    elif name == "Bas":
        return ModelBa(vocab_size, n_e, n_l, bs, d, rotate_v=False, use_softmax=sm, slope=0.25)
    elif name == "Bl":
        return ModelBl(vocab_size, n_e, n_l, bs, d, rotate_v=False, use_softmax=sm)
    elif name == "Bl'":
        return ModelBl(vocab_size, n_e, n_l, bs, d, rotate_v=True, use_softmax=sm)
    elif name == "J":
        return ModelJ(vocab_size, n_e, n_l, bs, n_relations=len(KG_RELATIONS),
                      dropout=d, rotate_v=False, use_softmax=sm)
    elif name == "J'":
        return ModelJ(vocab_size, n_e, n_l, bs, n_relations=len(KG_RELATIONS),
                      dropout=d, rotate_v=True, use_softmax=sm)
    elif name in ("Jr", "Jr'"):
        return ModelJ(vocab_size, n_e, n_l, bs, n_relations=len(KG_RELATIONS),
                      dropout=d, rotate_v=name.endswith("'"), use_softmax=sm, address_mode="random")
    elif name in ("J0", "J0'"):
        return ModelJ(vocab_size, n_e, n_l, bs, n_relations=len(KG_RELATIONS),
                      dropout=d, rotate_v=name.endswith("'"), use_softmax=sm, address_mode="none")
    else:
        raise ValueError(f"Unknown model: {name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_type(name):
    if name in ("B", "B'"):
        return "B"
    elif name in ("Bs", "Bs'"):
        return "Bs"
    elif name in ("Ba", "Bas"):
        return "Ba"
    elif name in ("Bl", "Bl'"):
        return "Bl"
    elif name in ("J", "J'", "Jr", "Jr'", "J0", "J0'"):
        return "J"
    raise ValueError(f"Unknown model: {name}")


def get_prompt_type(model_type):
    """Map model type to prompt type (Bs/Bl/Ba use B-style linearized prompts)."""
    if model_type in ("Bs", "Bl", "Ba"):
        return "B"
    return model_type


# ============================================================================
# Printing
# ============================================================================

def print_comparison_table(all_summaries, models):
    """Print comparison tables grouped by priming mode."""
    all_keys = set()
    for model_name in models:
        if model_name in all_summaries:
            all_keys.update(all_summaries[model_name].keys())

    priming_modes = sorted(set(k[3] for k in all_keys))

    for pm in priming_modes:
        pm_keys = sorted(k for k in all_keys if k[3] == pm)
        if not pm_keys:
            continue

        print(f"\n{'='*120}")
        if pm == "none":
            print(f"  COMPARISON TABLE — Unprimed")
        else:
            print(f"  COMPARISON TABLE — Primed: {pm}")
        print(f"{'='*120}")

        header = f"  {'Direction':<15} {'FactType':<12} {'NameSet':<8} "
        for m in models:
            header += f"{'h@1':>7} {'h@5':>7} {'PPL':>8}  "
        print(header)

        sub = f"  {'':<35} "
        for m in models:
            sub += f"{'--- ' + m + ' ---':^24}"
        print(sub)
        print(f"  {'-'*115}")

        for key in pm_keys:
            direction, fact_type, name_set, _ = key
            line = f"  {direction:<15} {fact_type:<12} {name_set:<8} "
            for m in models:
                if m in all_summaries and key in all_summaries[m]:
                    s = all_summaries[m][key]
                    line += f"{s['hit1']:>7.3f} {s['hit5']:>7.3f} {s['ppl']:>8.2f}  "
                else:
                    line += f"{'---':>7} {'---':>7} {'---':>8}  "
            print(line)


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(seed=42, models_to_run=None, n_train=500, n_test=100,
                   min_name_len=2, max_name_len=8, n_priming=3,
                   checkpoint_dir=None, load_checkpoints=False,
                   n_eval_skeletons=1):
    """Run one experiment with a given seed."""
    if models_to_run is None:
        models_to_run = MODEL_NAMES

    print(f"\n{'#'*70}")
    print(f"# Primed Zero-Shot, n_train={n_train}, n_test={n_test}, "
          f"n_priming={n_priming}, seed={seed}")
    print(f"{'#'*70}\n")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Generate data
    (train_names, test_names, train_chains, test_chains,
     train_facts, train_text_sentences, train_linearized) = generate_train_test_data(
        n_train=n_train, n_test=n_test, seed=seed,
        min_name_len=min_name_len, max_name_len=max_name_len)

    # Verify disjointness
    overlap = set(train_names) & set(test_names)
    assert len(overlap) == 0, f"Train/test name overlap: {overlap}"

    # Build vocabulary from all data + all names
    vocab = Vocabulary()
    vocab.build_from_sentences(train_text_sentences + train_linearized)
    vocab.build_from_sentences(train_names + test_names)

    print(f"Vocabulary size: {vocab.size}")
    print(f"Train names: {len(train_names)}, Test names: {len(test_names)}")
    print(f"Train facts: {len(train_facts)}")
    print(f"Train text sentences: {len(train_text_sentences)}")
    print(f"Train linearized KG: {len(train_linearized)}")

    # Build paired dataset
    paired_dataset = PairedDataset(train_facts, vocab, cfg.block_size)

    # Build eval prompts — subsample for large name sets
    max_eval_positions = 50
    train_eval_chains = {}
    test_eval_chains = {}
    for domain in CHAIN_ROLES.keys():
        tc = train_chains[domain]
        if len(tc) > max_eval_positions + 3:
            train_eval_chains[domain] = tc[:max_eval_positions + 3]
        else:
            train_eval_chains[domain] = tc
        tc2 = test_chains[domain]
        if len(tc2) > max_eval_positions + 3:
            test_eval_chains[domain] = tc2[:max_eval_positions + 3]
        else:
            test_eval_chains[domain] = tc2

    # Build per-model-type unprimed eval prompts
    # (J gets native KG + slot angle metadata, B gets linearized KG)
    unprimed_prompts_by_type = {}
    for mt in ["B", "J"]:
        train_ep = build_zero_shot_eval_prompts(train_eval_chains, vocab, mt, "train",
                                                n_eval_skeletons=n_eval_skeletons)
        test_ep = build_zero_shot_eval_prompts(test_eval_chains, vocab, mt, "test",
                                               n_eval_skeletons=n_eval_skeletons)
        unprimed_prompts_by_type[mt] = train_ep + test_ep
        print(f"Unprimed eval prompts ({mt}): {len(unprimed_prompts_by_type[mt])} "
              f"(train={len(train_ep)}, test={len(test_ep)})")

    # Primed eval prompts — per model type (J gets native KG, B gets linearized)
    primed_prompts_by_type = {}
    if n_priming > 0:
        for mt in ["B", "J"]:
            primed_train = build_primed_eval_prompts(
                train_eval_chains, vocab, n_priming, "train", model_type=mt)
            primed_test = build_primed_eval_prompts(
                test_eval_chains, vocab, n_priming, "test", model_type=mt)
            primed_prompts_by_type[mt] = primed_train + primed_test
            print(f"Primed eval prompts ({mt}): train={len(primed_train)} test={len(primed_test)}")

        # Verify block_size is large enough
        all_primed = []
        for pp_list in primed_prompts_by_type.values():
            all_primed.extend(pp_list)
        if all_primed:
            max_primed_len = max(
                len(p["prompt_tokens"]) + sum(len(s["tokens"]) for s in p["segments"])
                for p in all_primed
            )
            print(f"Max primed prompt+target length: {max_primed_len} "
                  f"(block_size={cfg.block_size})")
            if max_primed_len > cfg.block_size:
                print(f"  WARNING: Some primed sequences exceed block_size! "
                      f"They will be truncated.")

        # Print example primed prompts
        for mt_label in ["B", "J"]:
            for pm in ["same_rel", "same_entities"]:
                examples = [p for p in primed_prompts_by_type.get(mt_label, [])
                            if p["name_set"] == "test" and p["priming_mode"] == pm][:2]
                print(f"\n--- Example primed prompts {mt_label} ({pm}) ---")
                for p in examples:
                    decoded = vocab.decode(p["prompt_tokens"])
                    print(f"  prompt: {decoded[:200]}...")
                    print(f"  target: {p['target']}  rel: {p['relation']}")
        print()

    # Print example paired sequences
    print(f"--- Example paired sequences ---")
    for model_type in ["B", "J"]:
        print(f"\nModel {model_type}:")
        if model_type == "B":
            x, y, sl = paired_dataset.get_paired_batch_B(3, "cpu")
            for i in range(min(3, x.shape[0])):
                decoded = vocab.decode(x[i].tolist())
                print(f"  x[{i}]: {decoded[:120]}...")
                print(f"    segment_lens={sl[i]}")
        else:
            x, y, el, rn, rs, fi = paired_dataset.get_paired_batch_J(3, "cpu")
            for i in range(min(3, x.shape[0])):
                decoded = vocab.decode(x[i].tolist())
                print(f"  x[{i}]: {decoded[:120]}...")
                print(f"    entity_lens={el[i]}, role_slots={rs[i]}, rel={rn[i]}, fact_ids={fi[i]}")
    print()

    # Checkpoint directory
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Train and evaluate each model
    all_summaries = {}

    # Re-seed before training so eval prompt construction doesn't affect training
    torch.manual_seed(seed + 1000)
    random.seed(seed + 1000)
    np.random.seed(seed + 1000)

    for name in models_to_run:
        model = create_model(name, vocab.size, cfg)
        print(f"\nModel {name}: {count_parameters(model):,} params")

        ckpt_path = None
        if checkpoint_dir is not None:
            safe_name = name.replace("'", "p")
            ckpt_path = os.path.join(checkpoint_dir, f"model_{safe_name}_seed{seed}.pt")

        if load_checkpoints and ckpt_path and os.path.exists(ckpt_path):
            print(f"--- Loading checkpoint: {ckpt_path} ---")
            model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
            model.to(cfg.device)
        else:
            # Train
            print(f"--- Training Model {name} ---")
            model_type = get_model_type(name)
            train_model_paired(model, paired_dataset, cfg, model_type=model_type,
                               name=name, checkpoint_path=ckpt_path)

        # Evaluate (model-specific unprimed + model-specific primed)
        mt = get_model_type(name)
        pt = get_prompt_type(mt)  # Bs uses B-style prompts
        model_primed = primed_prompts_by_type.get(pt, [])
        model_eval_prompts = unprimed_prompts_by_type[pt] + model_primed
        summary, _ = evaluate_zero_shot(model, model_eval_prompts, vocab, cfg, name, model_type=mt)
        all_summaries[name] = summary

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_summaries


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Primed Zero-Shot Paired KG+Text Experiment")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to run (B, B', J, J')")
    parser.add_argument("--n_train", type=int, default=500,
                        help="Number of training names (default 500)")
    parser.add_argument("--n_test", type=int, default=100,
                        help="Number of test names (default 100)")
    parser.add_argument("--n_priming", type=int, default=4,
                        help="Number of distractor KG facts (default 4, 0=no priming)")
    parser.add_argument("--min_name_len", type=int, default=2)
    parser.add_argument("--max_name_len", type=int, default=8)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="first seed to run (seeds run are seed_offset .. seed_offset+seeds-1)")
    parser.add_argument("--n_embed", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test: 50 iters, 1 seed")
    parser.add_argument("--softmax", action="store_true",
                        help="Use softmax attention")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to save/load model checkpoints")
    parser.add_argument("--load_checkpoints", action="store_true",
                        help="Load checkpoints and skip training")
    parser.add_argument("--eval_true_pos", default="far",
                        help="position of the true fact among the priming facts at evaluation "
                             "(training always puts it adjacent to the text): far = furthest "
                             "(published protocol), near = adjacent, random = uniform, "
                             "posK = fixed index K (pos0 = far, pos4 = near with 4 distractors)")
    parser.add_argument("--n_eval_skeletons", type=int, default=1,
                        help="Skeletons per fact for eval (0=all, default=1)")
    args = parser.parse_args()

    if args.n_embed is not None:
        cfg.n_embed = args.n_embed
    if args.n_layers is not None:
        cfg.n_layers = args.n_layers
    if args.softmax:
        cfg.use_softmax = True
    global EVAL_TRUE_POS
    EVAL_TRUE_POS = args.eval_true_pos

    if args.smoke:
        cfg.max_iters = 50
        cfg.n_seeds = 1
        cfg.eval_interval = 25
        if args.n_train == 500:
            args.n_train = 20
        if args.n_test == 100:
            args.n_test = 10
        models = args.models or MODEL_NAMES
    else:
        if args.iters is not None:
            cfg.max_iters = args.iters
        if args.seeds is not None:
            cfg.n_seeds = args.seeds
        models = args.models or MODEL_NAMES

    # Normalize model names: trailing 'p' -> "'"
    normalized = []
    for m in models:
        if m.endswith("p") and m[:-1] + "'" in MODEL_NAMES:
            normalized.append(m[:-1] + "'")
        else:
            normalized.append(m)
    models = normalized

    print("=" * 70)
    print("  Primed Zero-Shot Paired KG+Text Experiment")
    print("=" * 70)
    attn_type = "softmax" if cfg.use_softmax else "log(exp+1)"
    print(f"\nConfig: n_train={args.n_train}, n_test={args.n_test}, "
          f"n_priming={args.n_priming}, "
          f"name_len={args.min_name_len}-{args.max_name_len}, n_embed={cfg.n_embed}, "
          f"n_layers={cfg.n_layers}, max_iters={cfg.max_iters}, "
          f"batch_size={cfg.batch_size}, lr={cfg.lr}, device={cfg.device}, "
          f"attention={attn_type}")
    print(f"Models: {models}")

    all_seed_summaries = []

    for seed in range(args.seed_offset, args.seed_offset + cfg.n_seeds):
        summaries = run_experiment(
            seed=seed, models_to_run=models,
            n_train=args.n_train, n_test=args.n_test,
            min_name_len=args.min_name_len, max_name_len=args.max_name_len,
            n_priming=args.n_priming,
            checkpoint_dir=args.checkpoint_dir,
            load_checkpoints=args.load_checkpoints,
            n_eval_skeletons=args.n_eval_skeletons)
        all_seed_summaries.append(summaries)

    # Average across seeds
    avg_summaries = {}
    for model_name in models:
        avg_summaries[model_name] = {}
        all_keys = set()
        for ss in all_seed_summaries:
            if model_name in ss:
                all_keys.update(ss[model_name].keys())
        for key in all_keys:
            metrics = {}
            for metric in ["hit1", "hit5", "ppl"]:
                values = [ss[model_name][key][metric]
                          for ss in all_seed_summaries
                          if model_name in ss and key in ss[model_name]]
                if values:
                    metrics[metric] = np.mean(values)
                    metrics[f"{metric}_std"] = np.std(values)
            metrics["n"] = (all_seed_summaries[0][model_name][key]["n"]
                           if model_name in all_seed_summaries[0]
                           and key in all_seed_summaries[0][model_name] else 0)
            avg_summaries[model_name][key] = metrics

    # Print comparison tables
    print_comparison_table(avg_summaries, models)

    # Aggregated summary grouped by priming mode
    all_priming_modes = set()
    for model_name in models:
        for key in avg_summaries[model_name]:
            all_priming_modes.add(key[3])
    priming_modes = sorted(all_priming_modes)

    print(f"\n{'='*80}")
    print(f"  AGGREGATED SUMMARY")
    print(f"{'='*80}")

    for pm in priming_modes:
        if pm == "none":
            print(f"\n  --- Unprimed ---")
        else:
            print(f"\n  --- Primed: {pm} ---")

        for name_set in ["train", "test"]:
            directions = ["kg_to_text"]
            for direction in directions:
                print(f"\n  {direction} / {name_set}:")
                for model_name in models:
                    h1s, h5s, ppls = [], [], []
                    for key, s in avg_summaries[model_name].items():
                        if key[0] == direction and key[2] == name_set and key[3] == pm:
                            h1s.append(s["hit1"])
                            h5s.append(s["hit5"])
                            ppls.append(s["ppl"])
                    if h1s:
                        print(f"    {model_name:>4s}: h@1={np.mean(h1s):.3f}  "
                              f"h@5={np.mean(h5s):.3f}  "
                              f"ppl={np.exp(np.mean(np.log(ppls))):.2f}")

    # Save results
    results_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"lis_results_{timestamp}.json")

    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(i) for i in obj]
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    save_data = {
        "config": {
            "n_train": args.n_train,
            "n_test": args.n_test,
            "n_priming": args.n_priming,
            "n_embed": cfg.n_embed,
            "n_layers": cfg.n_layers,
            "max_iters": cfg.max_iters,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "n_seeds": cfg.n_seeds,
            "models": models,
            "eval_true_pos": EVAL_TRUE_POS,
            "loaded_checkpoints": bool(args.load_checkpoints),
            "note": ("models loaded from checkpoints; max_iters above is the value passed to this "
                     "invocation, not necessarily the training length of the checkpoints"
                     if args.load_checkpoints else "models trained in this invocation"),
        },
        "results": to_serializable(avg_summaries),
        "per_seed": [to_serializable({m: {str(k): v for k, v in ss[m].items()} for m in ss})
                     for ss in all_seed_summaries],
        "seeds": list(range(args.seed_offset, args.seed_offset + cfg.n_seeds)),
        "timestamp": timestamp,
    }

    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    latest_file = os.path.join(results_dir, "lis_results_latest.json")
    with open(latest_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Latest results: {latest_file}")

    print("\n\nDone.")


if __name__ == "__main__":
    main()
