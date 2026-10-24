"""
Reproducible, headless runner for the Aioli synthetic-vs-real data-mixing study.

This is the single source of truth for the reported (multi-seed) results. It is the
GPU-ready version of `aioli-synthetic-comprehensive-analysis.ipynb` with the three
pre-submission fixes baked in (honest baseline names; disjoint train/val/test splits
with a non-empty assert; NaN backstop for empty shards) plus three additions the drafts
need: multiple seeds with error bands, held-out test-loss, and a FLOP-budget table.

Run on a free Colab/Kaggle GPU (see README_RUN.md). Examples:

    # quick CPU/GPU smoke test (tiny, ~minutes)
    python run_experiment.py --config SUPER_LIGHT --seeds 0 --scenario S2_9R_9S

    # reported run: 3 seeds, held-out test loss, the headline scenario
    python run_experiment.py --config HEAVY --seeds 0 1 2 --scenario S2_9R_9S --eval-test

    # add the rigorous "is the chosen mixture better" static-retrain eval (expensive)
    python run_experiment.py --config HEAVY --seeds 0 1 2 --scenario S2_9R_9S --eval-test --static-eval

Outputs (under --out-dir, default ./aioli_runs):
    results_<CONFIG>_<SCENARIO>.npy      per-seed histories + test losses
    agg_<...>_synth_share.csv            mean/std synthetic share per method per round
    agg_<...>_final_weights.csv          mean/std final per-domain weights
    agg_<...>_test_loss.csv              mean/std held-out test loss per method (if --eval-test)
    flops_<CONFIG>.csv / flops_<CONFIG>.md   FLOP budget table
    fig_<...>_synth_share.png            comparative synthetic share with error bands
    fig_<...>_divergence.png             L2-from-uniform with error bands
"""

import os, math, random, json, itertools, gc, time, logging, argparse
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_experiment")

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# transformers / datasets are imported lazily inside main() so that --help and the
# pure helper functions (FLOPs, aggregation) work without the heavy deps installed.


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
COMMON = dict(
    tokenizer_name="EleutherAI/pythia-160m",
    weight_decay=0.0,
    lr=5e-4,
    save_dir="./aioli_runs",
)

CONFIGS = {
    "SUPER_LIGHT": dict(
        model_name="EleutherAI/pythia-70m", context_length=2048,
        batch_size=2, max_steps=144, update_steps=22, lp_rounds=3, lp_steps=1,
        warmup_steps=20, eta=0.2, one_hot_factor=0.625, aioli_normalize_A=True, ema=None,
        # token budgets must be >> context_length so blocks form (see make_shards assert)
        train_tokens_per_domain=40_000, val_tokens_per_domain=8_000, test_tokens_per_domain=8_000,
        # DoGE: micro-batches per domain when estimating gradient alignment.
        # Skill-It: training steps per single-skill probe when LEARNING the skills graph.
        doge_est_batches=1, skillit_graph_probe_steps=5,
    ),
    "LIGHT": dict(
        model_name="EleutherAI/pythia-70m", context_length=1024,
        batch_size=4, max_steps=7200, update_steps=1100, lp_rounds=6, lp_steps=5,
        warmup_steps=200, eta=0.2, one_hot_factor=0.625, aioli_normalize_A=True, ema=None,
        train_tokens_per_domain=200_000, val_tokens_per_domain=20_000, test_tokens_per_domain=20_000,
        doge_est_batches=2, skillit_graph_probe_steps=50,
    ),
    "HEAVY": dict(
        model_name="EleutherAI/pythia-160m", context_length=2048,
        batch_size=8, max_steps=14400, update_steps=2200, lp_rounds=6, lp_steps=10,
        warmup_steps=500, eta=0.2, one_hot_factor=0.625, aioli_normalize_A=True, ema=0.1,
        train_tokens_per_domain=1_000_000, val_tokens_per_domain=100_000, test_tokens_per_domain=100_000,
        doge_est_batches=2, skillit_graph_probe_steps=200,
    ),
}

REAL_DOMAINS = {
    "wikipedia_en":     ("ola13/small-the_pile", "train", "Wikipedia (en)"),
    "pile_cc":          ("ola13/small-the_pile", "train", "Pile-CC"),
    "openwebtext2":     ("ola13/small-the_pile", "train", "OpenWebText2"),
    "github":           ("ola13/small-the_pile", "train", "Github"),
    "pubmed_abstracts": ("ola13/small-the_pile", "train", "PubMed Abstracts"),
    "stackexchange":    ("ola13/small-the_pile", "train", "StackExchange"),
    "freelaw":          ("ola13/small-the_pile", "train", "FreeLaw"),
    "uspto":            ("ola13/small-the_pile", "train", "USPTO Backgrounds"),
    "arxiv":            ("ola13/small-the_pile", "train", "ArXiv"),
}
SYN_DOMAINS = {
    "ultrachat":     ("HuggingFaceH4/ultrachat_200k",            "train_sft", "UltraChat"),
    "self_instruct": ("databricks/databricks-dolly-15k",          "train",     "Self-Instruct"),
    "cosmopedia":    ("HuggingFaceTB/cosmopedia",                "train",     "Cosmopedia"),
    "codealpaca":    ("sahil2801/CodeAlpaca-20k",                "train",     "Code-Alpaca"),
    "gpteacher":     ("teknium/GPTeacher-General-Instruct",      "train",     "GPTeacher"),
    "wizard70k":     ("WizardLMTeam/WizardLM_evol_instruct_70k", "train",     "WizardLM-70k"),
    "alpaca":        ("tatsu-lab/alpaca",                        "train",     "Alpaca-52k"),
    "sharegpt":      ("RyokoAI/ShareGPT52K",                     "train",     "ShareGPT"),
    "open_platypus": ("garage-bAInd/Open-Platypus",             "train",     "Open-Platypus"),
}
SCENARIOS = {
    "S1_1R_9S": {"real": ["wikipedia_en"], "synth": list(SYN_DOMAINS.keys())},
    "S2_9R_9S": {"real": list(REAL_DOMAINS.keys()), "synth": list(SYN_DOMAINS.keys())},
    "S3_9R_1S": {"real": list(REAL_DOMAINS.keys()), "synth": ["self_instruct"]},
}

# DoGE and Skill-It are faithful implementations (gradient alignment / learned skills graph).
# Gen-Delta and Improvement-Window are the simpler val-loss-improvement heuristics, kept as extra
# reference points behind --extra-heuristics (they are NOT DoGE / Skill-It).
METHOD_ORDER = ["Aioli", "Stratified", "DoReMi-style", "Group-DRO", "DoGE", "Skill-It"]
EXTRA_HEURISTICS = ["Gen-Delta", "Improvement-Window"]


# --------------------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------
# Data loading (FIX: disjoint shuffled splits + non-empty assert)
# --------------------------------------------------------------------------------------
def make_dataset_factory(tokenizer):
    from datasets import load_dataset

    def extract_text_real(example, target_domain_name):
        if "meta" in example and isinstance(example["meta"], dict):
            subset = example["meta"].get("pile_set_name")
            if subset is not None and subset != target_domain_name:
                return None
        for key in ("text", "content", "document", "body"):
            if key in example and isinstance(example[key], str):
                return example[key]
        if "meta" in example and isinstance(example["meta"], dict):
            for k, v in example["meta"].items():
                if isinstance(v, str) and len(v.split()) > 3:
                    return v
        return None

    def extract_text_synth(example, ds_pretty):
        prompt = response = None
        for pkey in ("instruction", "prompt", "question", "input", "src", "query"):
            if pkey in example and isinstance(example[pkey], str) and example[pkey].strip():
                prompt = example[pkey].strip(); break
        for rkey in ("output", "response", "answer", "target", "text", "completion"):
            if rkey in example and isinstance(example[rkey], str) and example[rkey].strip():
                response = example[rkey].strip(); break
        if response is None and isinstance(example.get("messages"), list):
            msgs = example["messages"]
            prompt = "\n".join(m.get("content", "") for m in msgs if m.get("role") in ("user", "system")).strip() or prompt
            response = "\n".join(m.get("content", "") for m in msgs if m.get("role") == "assistant").strip() or response
        if response is None and isinstance(example.get("conversations"), list):
            import json as _json
            def _as_dict(t):
                if isinstance(t, dict): return t
                try: return _json.loads(t)
                except Exception: return {}
            conv = [_as_dict(t) for t in example["conversations"]]
            prompt = "\n".join(t.get("value", "") for t in conv if t.get("from") in ("human", "system")).strip() or prompt
            response = "\n".join(t.get("value", "") for t in conv if t.get("from") in ("gpt", "assistant")).strip() or response
        if prompt and response:
            return f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
        if isinstance(example.get("text"), str):
            return example["text"]
        return None

    def load_domain_dataset(hub_id, split, pretty_name, is_synth, category="stories"):
        if hub_id == "HuggingFaceTB/cosmopedia":
            ds = load_dataset(hub_id, category, split=split, streaming=True)
        else:
            ds = load_dataset(hub_id, split=split, streaming=True)
        def map_fn(ex):
            txt = extract_text_synth(ex, pretty_name) if is_synth else extract_text_real(ex, pretty_name)
            return {"text": txt}
        ds = ds.map(map_fn)
        ds = ds.filter(lambda ex: ex["text"] is not None and len(ex["text"]) > 0)
        return ds

    return load_domain_dataset


def stream_tokens(tokenizer, ds_stream, target_tokens, block_size):
    from tqdm.auto import tqdm
    buffer_ids, produced = [], 0
    for ex in tqdm(ds_stream, desc=f"stream {target_tokens // 1000}k tok", leave=False):
        ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
        buffer_ids.extend(ids + [tokenizer.eos_token_id])
        while len(buffer_ids) >= block_size:
            yield buffer_ids[:block_size]
            produced += block_size
            buffer_ids = buffer_ids[block_size:]
            if produced >= target_tokens:
                return


class DomainShard(Dataset):
    def __init__(self, id_arrays):
        self.arrays = id_arrays
    def __len__(self):
        return len(self.arrays)
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.arrays[idx], dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def make_shards_for_domain(tokenizer, load_domain_dataset, name, hub_id, split, pretty, is_synth,
                           train_tok, val_tok, test_tok, block_size, split_seed=1234):
    # FIX (leakage): stream ONE pool per domain, shuffle deterministically, then slice into
    # DISJOINT train/val/test. Previously each split re-streamed from row 0, so val/test
    # overlapped the train prefix and validation loss was measured on training data.
    ds = load_domain_dataset(hub_id, split, pretty, is_synth)
    all_blocks = list(stream_tokens(tokenizer, ds, train_tok + val_tok + test_tok, block_size))
    np.random.default_rng(split_seed).shuffle(all_blocks)

    n_train = train_tok // block_size
    n_val = val_tok // block_size
    n_test = test_tok // block_size
    train_ids = all_blocks[:n_train]
    val_ids = all_blocks[n_train:n_train + n_val]
    test_ids = all_blocks[n_train + n_val:n_train + n_val + n_test]

    # FIX (empty shards): fail loudly instead of silently yielding a 0.0-loss domain.
    assert len(train_ids) > 0 and len(val_ids) > 0 and len(test_ids) > 0, (
        f"[{name}] empty shard: only {len(all_blocks)} blocks for "
        f"{train_tok + val_tok + test_tok} tokens at block_size={block_size} "
        f"(need train/val/test = {n_train}/{n_val}/{n_test}). Increase *_tokens_per_domain "
        f"(must be >> block_size) or check pile_set_name filtering for this domain."
    )
    return {"train": DomainShard(train_ids), "val": DomainShard(val_ids), "test": DomainShard(test_ids),
            "name": name, "pretty": pretty, "is_synth": is_synth}


def build_scenario_domains(tokenizer, load_domain_dataset, scenario, cfg, split_seed=1234):
    from tqdm.auto import tqdm
    keys = [(k, False) for k in scenario["real"]] + [(k, True) for k in scenario["synth"]]
    domains = []
    for key, is_synth in tqdm(keys, desc="building domains", leave=False):
        hub_id, split, pretty = (SYN_DOMAINS if is_synth else REAL_DOMAINS)[key]
        domains.append(make_shards_for_domain(
            tokenizer, load_domain_dataset, key, hub_id, split, pretty, is_synth,
            cfg["train_tokens_per_domain"], cfg["val_tokens_per_domain"], cfg["test_tokens_per_domain"],
            cfg["context_length"], split_seed=split_seed))
    return domains


# --------------------------------------------------------------------------------------
# Sampling + evaluation
# --------------------------------------------------------------------------------------
def attach_train_cursors(domains):
    for d in domains:
        d["_train_cursor_pool"] = np.arange(len(d["train"]))


def mixture_indices_per_batch(domains, weights, batch_size):
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    counts = np.random.multinomial(batch_size, weights)
    picks = []
    for di, c in enumerate(counts):
        if c == 0:
            continue
        pool = domains[di]["_train_cursor_pool"]
        if len(pool) == 0:
            continue
        idxs = np.random.randint(0, len(pool), size=c)
        picks.extend([(di, int(pool[i])) for i in idxs])
    random.shuffle(picks)
    return picks


def batchify(tokenizer, domains, picks, device):
    input_ids = [domains[di]["train"][si]["input_ids"] for (di, si) in picks]
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    # FIX: mask pad positions so loss is not taken on padding
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {"input_ids": input_ids.to(device), "labels": labels.to(device)}


def eval_collate_fn(tokenizer, device, batch_list):
    input_ids = [item["input_ids"] for item in batch_list]
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {"input_ids": input_ids.to(device), "labels": labels.to(device)}


def eval_domain_losses(model, tokenizer, domains, device, split="val", max_eval_batches=25, eval_bsz=16):
    model.eval()
    out = []
    with torch.no_grad():
        for di, d in enumerate(domains):
            ds = d[split]
            if len(ds) == 0:
                # FIX: never report a fake 0.0 for an empty shard; NaN is a loud backstop.
                logger.warning("Empty '%s' shard for domain %s; recording NaN.", split, d["name"])
                out.append(np.nan)
                continue
            loader = DataLoader(ds, batch_size=eval_bsz, shuffle=False,
                                collate_fn=lambda b: eval_collate_fn(tokenizer, device, b))
            total, n = 0.0, 0
            for i, batch in enumerate(loader):
                if i >= max_eval_batches:
                    break
                total += model(**batch).loss.detach().float().item()
                n += 1
            out.append(total / max(n, 1))
    return np.array(out, dtype=float)


# --------------------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------------------
@dataclass
class AioliArgs:
    eta: float
    lp_rounds: int
    lp_steps: int
    update_steps: int
    one_hot_factor: float
    aioli_normalize_A: bool = True
    ema: float = None


class CompactAioli:
    def __init__(self, model, tokenizer, domains, args, cfg, device, sched_total=None, sched_warmup=None):
        # sched_total/sched_warmup let short probes (e.g. the Skill-It graph) size their own LR schedule
        # instead of inheriting the full-run warmup, which would keep the probe stuck near LR 0.
        self.model, self.tok, self.domains, self.args = model, tokenizer, domains, args
        self.cfg, self.device = cfg, device
        self.k = len(domains)
        self.history = {"round": [], "p": [], "A": [], "val_losses": []}
        from transformers import get_linear_schedule_with_warmup
        total = sched_total if sched_total is not None else cfg["max_steps"]
        warm = sched_warmup if sched_warmup is not None else cfg["warmup_steps"]
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warm, num_training_steps=total)

    def _train_steps(self, n_steps, weights):
        attach_train_cursors(self.domains)
        bsz = self.cfg["batch_size"]
        for _ in range(n_steps):
            self.model.train()
            picks = mixture_indices_per_batch(self.domains, weights, bsz)
            if not picks:
                continue
            batch = batchify(self.tok, self.domains, picks, self.device)
            loss = self.model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step(); self.scheduler.step(); self.model.zero_grad()

    def _eval_val(self):
        return eval_domain_losses(self.model, self.tok, self.domains, self.device, split="val",
                                  eval_bsz=self.cfg["batch_size"] * 2)

    def _learn_params(self, prev_val_losses):
        k = self.k
        A_hat = np.zeros((k, k))
        indices = np.concatenate([np.random.permutation(k) for _ in range(self.args.lp_rounds)])
        eps = self.args.one_hot_factor
        weight_mat = np.full((k, k), (1 - eps) / (k - 1)); np.fill_diagonal(weight_mat, eps)
        cur = prev_val_losses.copy()
        for j in indices:
            self._train_steps(self.args.lp_steps, weight_mat[:, j])
            new = self._eval_val()
            A_hat[:, j] += (cur - new)
            cur = new
        A_hat /= self.args.lp_rounds
        if self.args.one_hot_factor not in (1.0, 0.0):
            try:
                A_hat = np.vstack([np.linalg.solve(weight_mat, A_hat[i, :]) for i in range(k)])
            except np.linalg.LinAlgError:
                logger.warning("LinAlgError solving for A; using A_hat directly.")
        if self.args.aioli_normalize_A:
            m = A_hat.min()
            if m < 0:
                A_hat = A_hat - m
            mc = A_hat.sum(axis=0).mean()
            if mc > 0:
                A_hat = A_hat / mc
        return A_hat

    def run(self):
        from tqdm.auto import tqdm
        k = self.k
        p = np.ones(k) / k
        val = self._eval_val()
        ema_A, step, t = None, 0, 0
        max_steps = self.cfg["max_steps"]
        pbar = tqdm(total=max_steps, desc="Aioli")
        while step < max_steps:
            self.history["round"].append(t); self.history["p"].append(p.copy()); self.history["val_losses"].append(val.copy())
            A = self._learn_params(val); self.history["A"].append(A.copy())
            probe = self.k * self.args.lp_rounds * self.args.lp_steps
            if step + probe > max_steps:
                pbar.update(max_steps - step); step += probe; break
            step += probe; pbar.update(probe)
            if self.args.ema is not None:
                ema_A = A if ema_A is None else (1 - self.args.ema) * A + self.args.ema * ema_A
                p = p * np.exp(self.args.eta * ema_A.sum(axis=0))
            else:
                p = p * np.exp(self.args.eta * A.sum(axis=0))
            p = p / p.sum()
            exploit = min(self.args.update_steps, max_steps - step)
            if exploit <= 0:
                break
            self._train_steps(exploit, p); step += exploit; pbar.update(exploit)
            val = self._eval_val(); t += 1
        pbar.close()
        self.history["round"].append(t); self.history["p"].append(p.copy()); self.history["val_losses"].append(val.copy())
        self.history["A"].append((A if "A" in dir() else np.zeros((k, k))).copy())
        return self.history


def _loss_reweight_loop(model, tokenizer, domains, cfg, device, steps, update_fn, desc):
    """Shared online-reweighting loop for the loss-based baselines."""
    from tqdm.auto import tqdm
    k = len(domains)
    w = np.ones(k) / k
    hist = {"round": [], "p": []}
    trainer = CompactAioli(model, tokenizer, domains, AioliArgs(0, 0, 0, 0, 1.0), cfg, device)
    state = {}
    step, t = 0, 0
    pbar = tqdm(total=steps, desc=desc)
    while step < steps:
        hist["round"].append(t); hist["p"].append(w.copy())
        run = min(cfg["update_steps"], steps - step)
        if run <= 0:
            break
        trainer._train_steps(run, w); step += run; pbar.update(run)
        w = update_fn(trainer, w, state, t)
        t += 1
    pbar.close()
    hist["round"].append(t); hist["p"].append(w.copy())
    return hist


def run_stratified(model, tokenizer, domains, cfg, device, steps):
    from tqdm.auto import tqdm
    k = len(domains)
    p = np.ones(k) / k
    trainer = CompactAioli(model, tokenizer, domains, AioliArgs(cfg["eta"], 1, 1, steps, 1.0, False, None), cfg, device)
    trainer._train_steps(steps, p)
    return {"round": [0, 1], "p": [p.copy(), p.copy()]}


def run_group_dro(model, tokenizer, domains, cfg, device, steps, gamma=0.2):
    def upd(trainer, w, state, t):
        loss = trainer._eval_val()
        loss = loss - np.nanmean(loss)
        w = w * np.exp(gamma * np.nan_to_num(loss, nan=0.0)); return w / w.sum()
    return _loss_reweight_loop(model, tokenizer, domains, cfg, device, steps, upd, "Group-DRO")


def run_doremi_full(ref_model, model, tokenizer, domains, cfg, device, steps, gamma=0.2):
    from tqdm.auto import tqdm
    k = len(domains)
    w = np.ones(k) / k
    hist = {"round": [], "p": []}
    loss_ref = eval_domain_losses(ref_model, tokenizer, domains, device, "val", eval_bsz=cfg["batch_size"] * 2)
    trainer = CompactAioli(model, tokenizer, domains, AioliArgs(0, 0, 0, 0, 1.0), cfg, device)
    p_hist, step, t = [], 0, 0
    pbar = tqdm(total=steps, desc="DoReMi-style")
    while step < steps:
        p_hist.append(w.copy()); hist["round"].append(t); hist["p"].append(np.mean(p_hist, axis=0))
        run = min(cfg["update_steps"], steps - step)
        if run <= 0:
            break
        trainer._train_steps(run, w); step += run; pbar.update(run)
        excess = trainer._eval_val() - loss_ref
        excess = excess - np.nanmean(excess)
        w = w * np.exp(gamma * np.nan_to_num(excess, nan=0.0)); w = w / w.sum(); t += 1
    pbar.close()
    hist["round"].append(t); hist["p"].append(np.mean(p_hist, axis=0) if p_hist else np.ones(k) / k)
    return hist


def run_gen_delta_heuristic(model, tokenizer, domains, cfg, device, steps, mu=0.1):
    # NOT DoGE. Val-loss-improvement heuristic (no gradient alignment). Reported as "Gen-Delta".
    state = {"prev": eval_domain_losses(model, tokenizer, domains, device, "val", eval_bsz=cfg["batch_size"] * 2)}
    def upd(trainer, w, st, t):
        cur = trainer._eval_val()
        delta = state["prev"] - cur
        w = w * np.exp(mu * np.nan_to_num(delta, nan=0.0) / (np.nanmean(np.abs(delta)) + 1e-6))
        state["prev"] = cur; return w / w.sum()
    return _loss_reweight_loop(model, tokenizer, domains, cfg, device, steps, upd, "Gen-Delta")


def run_imp_window_heuristic(model, tokenizer, domains, cfg, device, steps, eta=0.2, window=2):
    # NOT Skill-It. Trailing-window val-loss-improvement heuristic (no skills graph). "Improvement-Window".
    state = {"prev": eval_domain_losses(model, tokenizer, domains, device, "val", eval_bsz=cfg["batch_size"] * 2), "q": []}
    def upd(trainer, w, st, t):
        cur = trainer._eval_val()
        state["q"].append(np.nan_to_num(state["prev"] - cur, nan=0.0)); state["prev"] = cur
        if len(state["q"]) > window:
            state["q"].pop(0)
        imp = np.stack(state["q"], axis=0).sum(0)
        w = np.exp(eta * imp); return w / w.sum()
    return _loss_reweight_loop(model, tokenizer, domains, cfg, device, steps, upd, "Improvement-Window")


# ---- Faithful DoGE (Fan et al., 2024) ------------------------------------------------
def _grad_alignment_scores(model, tokenizer, domains, device, est_batches, bsz):
    """DoGE generalization estimation. Returns score_j = <g_j, g_sum> where g_j is the
    per-domain training gradient and g_sum = sum_i g_i is the universal-generalization
    gradient (the direction that lowers the average loss across all domains). A domain whose
    gradient aligns with g_sum is one whose training broadly helps; DoGE upweights it.
    Two backward passes over the domains: pass 1 accumulates g_sum into .grad (no zeroing
    between domains), pass 2 recomputes each g_j and dots it with the stored g_sum."""
    attach_train_cursors(domains)
    params = [p for p in model.parameters() if p.requires_grad]
    model.train()

    def domain_backward(di, scale):
        for _ in range(est_batches):
            picks = mixture_indices_per_batch(domains, np.eye(len(domains))[di], bsz)
            if not picks:
                continue
            loss = model(**batchify(tokenizer, domains, picks, device)).loss / (est_batches * scale)
            loss.backward()

    # pass 1: g_sum = sum_i grad(L_i) accumulated in .grad
    model.zero_grad(set_to_none=True)
    for di in range(len(domains)):
        domain_backward(di, 1.0)
    # store g_sum in fp32: summing k bf16 gradients then dotting would lose precision
    g_sum = [p.grad.detach().float().clone() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32)
             for p in params]

    # pass 2: score_j = <grad(L_j), g_sum>
    scores = np.zeros(len(domains))
    for di in range(len(domains)):
        model.zero_grad(set_to_none=True)
        domain_backward(di, 1.0)
        dot = 0.0
        for p, gs in zip(params, g_sum):
            if p.grad is not None:
                # accumulate in fp32: bf16 grads over ~1e8 elements would lose precision
                dot += torch.sum(p.grad.detach().float() * gs.float()).item()
        scores[di] = dot
    model.zero_grad(set_to_none=True)
    model.eval()
    del g_sum
    return scores


def run_doge(model, tokenizer, domains, cfg, device, steps, mu=0.2):
    """DoGE (Fan et al., 2024), universal-generalization variant. The reweighting signal is DoGE's
    Eq. (6): score_j = <grad(L_j), sum_i grad(L_i)> = <grad(L_j), grad of the average-loss direction>,
    which includes the ||grad(L_j)||^2 self-term by design. We apply the exponentiated-gradient update
    and report the time-averaged weights (DoGE's output a-bar). Two documented deviations, shared
    across all methods here for an apples-to-apples weight-trajectory comparison: (1) this is an ONLINE
    single-model run, not DoGE's proxy->base two-stage protocol; (2) the alignment is estimated per
    round (per update_steps block), not per step. Scores are normalized by their mean absolute value
    for scale-invariance (preserves the relative alignment magnitudes and ranking, not the absolute
    scale of Eq. (6), so mu is a unitless step size). The defining gradient-inner-product signal, absent
    in the older heuristic, is genuinely computed."""
    from tqdm.auto import tqdm
    k = len(domains)
    w = np.ones(k) / k
    hist = {"round": [], "p": []}
    trainer = CompactAioli(model, tokenizer, domains, AioliArgs(0, 0, 0, 0, 1.0), cfg, device)
    est_batches = cfg.get("doge_est_batches", 2)
    alpha_acc = np.zeros(k); n_acc = 0
    step, t = 0, 0
    pbar = tqdm(total=steps, desc="DoGE")
    while step < steps:
        scores = _grad_alignment_scores(model, tokenizer, domains, device, est_batches, cfg["batch_size"])
        s = scores / (np.abs(scores).mean() + 1e-8)  # scale-invariant EG step (keeps relative magnitude)
        w = w * np.exp(mu * s); w = w / w.sum()
        alpha_acc += w; n_acc += 1
        hist["round"].append(t); hist["p"].append((alpha_acc / n_acc).copy())  # DoGE reports averaged weights
        run = min(cfg["update_steps"], steps - step)
        if run <= 0:
            break
        trainer._train_steps(run, w); step += run; pbar.update(run); t += 1
    pbar.close()
    hist["round"].append(t); hist["p"].append((alpha_acc / max(n_acc, 1)).copy())
    return hist


# ---- Faithful Skill-It (Chen et al., 2023) -------------------------------------------
def learn_skill_graph(tokenizer, domains, cfg, device, probe_steps):
    """Learn the Skill-It skills-influence graph A in the paper's orientation (Chen et al., 2023,
    line 304): A[i, j] = strength of the edge from training skill i to eval skill j = the drop in
    validation loss on eval skill j after training a fresh proxy briefly on training skill i alone.
    One isolated single-skill proxy per row i (k fresh models), matching Skill-It's linear-time graph
    learning (their line 348). The probe gets its own short LR schedule (sized to probe_steps) so it
    is not stuck near LR 0 by the full-run warmup."""
    from tqdm.auto import tqdm
    k = len(domains)
    A = np.zeros((k, k))
    warm = min(cfg["warmup_steps"], max(1, probe_steps // 5))
    for i in tqdm(range(k), desc="Skill-It graph", leave=False):
        m = new_model(cfg, device)
        L0 = eval_domain_losses(m, tokenizer, domains, device, "val", eval_bsz=cfg["batch_size"] * 2)
        tr = CompactAioli(m, tokenizer, domains, AioliArgs(0, 0, 0, 0, 1.0), cfg, device,
                          sched_total=probe_steps, sched_warmup=warm)
        tr._train_steps(probe_steps, np.eye(k)[i])   # train on skill i only
        L1 = eval_domain_losses(m, tokenizer, domains, device, "val", eval_bsz=cfg["batch_size"] * 2)
        A[i, :] = np.nan_to_num(L0 - L1, nan=0.0)    # effect of training skill i on each eval skill j
        del m, tr; _free()
    return A


def run_skillit(model, tokenizer, domains, cfg, device, steps, eta=0.2, window=3):
    """Skill-It (Chen et al., 2023), continual-pretraining regime (S_eval = S_train, A is k x k).
    Learn the skills graph A once, then run the online multiplicative-weights update of Algorithm 1:
    w_i <- w_i * exp(eta * sum_{tau in last w rounds} sum_j A[i,j] * L_j(tau)), i.e. score_i = (A @ L)_i,
    where L_j is the current validation loss on eval skill j. This allocates training to skills that
    most help the currently-hardest skills, the defining Skill-It dynamic (the older heuristic had no
    graph at all). The graph is reused on the same val split (faithful in this regime). Two documented
    choices: weights start uniform (the paper inits from a softmax over A row-sums; we keep uniform so
    every method shares the 0.5 synthetic-share start), and scores are normalized by mean absolute
    value for scale-invariance."""
    from tqdm.auto import tqdm
    k = len(domains)
    A = learn_skill_graph(tokenizer, domains, cfg, device, cfg.get("skillit_graph_probe_steps", 50))
    w = np.ones(k) / k
    hist = {"round": [], "p": []}
    trainer = CompactAioli(model, tokenizer, domains, AioliArgs(0, 0, 0, 0, 1.0), cfg, device)
    recent = []  # trailing window of per-round (A @ L) scores
    step, t = 0, 0
    pbar = tqdm(total=steps, desc="Skill-It")
    while step < steps:
        hist["round"].append(t); hist["p"].append(w.copy())
        run = min(cfg["update_steps"], steps - step)
        if run <= 0:
            break
        trainer._train_steps(run, w); step += run; pbar.update(run)
        L = np.nan_to_num(trainer._eval_val(), nan=0.0)
        recent.append(A @ L)                          # score_i = sum_j A[i,j] * L_j
        if len(recent) > window:
            recent.pop(0)
        score = np.sum(recent, axis=0)                # trailing-window sum (Algorithm 1)
        s = score / (np.abs(score).mean() + 1e-8)     # scale-invariant
        w = w * np.exp(eta * s); w = w / w.sum(); t += 1
    pbar.close()
    hist["round"].append(t); hist["p"].append(w.copy())
    return hist


# --------------------------------------------------------------------------------------
# One scenario at one seed
# --------------------------------------------------------------------------------------
def new_model(cfg, device):
    from transformers import AutoModelForCausalLM
    # NOTE (dtype): every method, including the baselines and the DoReMi reference, loads in the SAME
    # precision (bf16 on CUDA, fp32 on CPU) so the per-domain loss signal is comparable across methods.
    # The original notebook loaded only Aioli in bf16 and the baselines in fp32; this uniform choice
    # supersedes that mixed-precision setup, so the multi-seed numbers replace the seed=42 [PRELIM] table
    # rather than extending it. See CODE_FIXES.md.
    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    return AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=dtype).to(device)


def run_scenario_one_seed(domains, tokenizer, cfg, device, seed, run_baselines=True, eval_test=False,
                          static_eval=False, extra_heuristics=False):
    set_all_seeds(seed)
    steps = cfg["max_steps"]
    result = {"histories": {}, "test_loss": {}, "static_test_loss": {}, "n_params": None}

    def finalize(name, model):
        if eval_test:
            result["test_loss"][name] = eval_domain_losses(model, tokenizer, domains, device, "test",
                                                           eval_bsz=cfg["batch_size"] * 2).tolist()
        if static_eval and name != "Stratified":
            # Save/restore RNG so the static retrain does not perturb the main per-seed stream
            # (otherwise enabling --static-eval would change the other methods' trajectories).
            py_state, np_state = random.getstate(), np.random.get_state()
            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            p_final = np.array(result["histories"][name]["p"][-1], dtype=float)
            sm = new_model(cfg, device)
            set_all_seeds(seed + 10_000)  # independent stream for the static retrain
            st = CompactAioli(sm, tokenizer, domains, AioliArgs(0, 0, 0, 0, 1.0), cfg, device)
            st._train_steps(steps, p_final / p_final.sum())
            result["static_test_loss"][name] = eval_domain_losses(sm, tokenizer, domains, device, "test",
                                                                  eval_bsz=cfg["batch_size"] * 2).tolist()
            del sm; _free()
            random.setstate(py_state); np.random.set_state(np_state)
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    # Aioli
    m = new_model(cfg, device)
    result["n_params"] = int(sum(p.numel() for p in m.parameters()))
    aioli = CompactAioli(m, tokenizer, domains, AioliArgs(
        cfg["eta"], cfg["lp_rounds"], cfg["lp_steps"], cfg["update_steps"],
        cfg["one_hot_factor"], cfg["aioli_normalize_A"], cfg["ema"]), cfg, device)
    result["histories"]["Aioli"] = aioli.run()
    finalize("Aioli", aioli.model)
    del aioli, m; _free()

    if run_baselines:
        m = new_model(cfg, device)
        result["histories"]["Stratified"] = run_stratified(m, tokenizer, domains, cfg, device, steps)
        finalize("Stratified", m); del m; _free()

        ref = new_model(cfg, device); m = new_model(cfg, device)
        result["histories"]["DoReMi-style"] = run_doremi_full(ref, m, tokenizer, domains, cfg, device, steps)
        finalize("DoReMi-style", m); del ref, m; _free()

        m = new_model(cfg, device)
        result["histories"]["Group-DRO"] = run_group_dro(m, tokenizer, domains, cfg, device, steps)
        finalize("Group-DRO", m); del m; _free()

        # Faithful DoGE (gradient alignment) and Skill-It (learned skills graph).
        m = new_model(cfg, device)
        result["histories"]["DoGE"] = run_doge(m, tokenizer, domains, cfg, device, steps)
        finalize("DoGE", m); del m; _free()

        m = new_model(cfg, device)
        result["histories"]["Skill-It"] = run_skillit(m, tokenizer, domains, cfg, device, steps)
        finalize("Skill-It", m); del m; _free()

        if extra_heuristics:
            m = new_model(cfg, device)
            result["histories"]["Gen-Delta"] = run_gen_delta_heuristic(m, tokenizer, domains, cfg, device, steps)
            finalize("Gen-Delta", m); del m; _free()

            m = new_model(cfg, device)
            result["histories"]["Improvement-Window"] = run_imp_window_heuristic(m, tokenizer, domains, cfg, device, steps)
            finalize("Improvement-Window", m); del m; _free()

    return result


def _free():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# --------------------------------------------------------------------------------------
# Aggregation + plotting + FLOPs (pure; unit-testable without GPU)
# --------------------------------------------------------------------------------------
def synth_share_series(history, is_synth):
    is_synth = np.asarray(is_synth, dtype=bool)
    rounds, shares = [], []
    for t, p in zip(history["round"], history["p"]):
        p = np.asarray(p, dtype=float)
        rounds.append(t); shares.append(float(p[is_synth].sum()))
    return np.array(rounds), np.array(shares)


def l2_from_uniform_series(history):
    rounds, l2 = [], []
    for t, p in zip(history["round"], history["p"]):
        p = np.asarray(p, dtype=float)
        k = len(p)
        l2.append(float(np.linalg.norm(p - 1.0 / k))); rounds.append(t)
    return np.array(rounds), np.array(l2)


def aggregate_over_seeds(per_seed_results, is_synth, method_order=METHOD_ORDER + EXTRA_HEURISTICS):
    """Return dict method -> {rounds, share_mean, share_std, l2_mean, l2_std} across seeds."""
    agg = {}
    for method in method_order:
        share_runs, l2_runs, rounds_ref = [], [], None
        for res in per_seed_results:
            h = res["histories"].get(method)
            if h is None:
                continue
            r, s = synth_share_series(h, is_synth)
            _, l2 = l2_from_uniform_series(h)
            if rounds_ref is None:
                rounds_ref = r
            n = min(len(rounds_ref), len(s))
            share_runs.append(s[:n]); l2_runs.append(l2[:n])
        if not share_runs:
            continue
        lengths = {len(x) for x in share_runs}
        if len(lengths) > 1:
            logger.warning("%s: seeds produced unequal round counts %s; truncating to the shortest.",
                           method, sorted(lengths))
        m = min(len(x) for x in share_runs)
        share = np.stack([x[:m] for x in share_runs]); l2 = np.stack([x[:m] for x in l2_runs])
        agg[method] = dict(rounds=rounds_ref[:m],
                           share_mean=share.mean(0), share_std=share.std(0),
                           l2_mean=l2.mean(0), l2_std=l2.std(0))
    return agg


def flop_table(cfg, n_params, config_name, k=None):
    """6*N*D approximation. D = max_steps*batch*ctx tokens per run.
    Aioli's probe steps are counted WITHIN max_steps (the run loop caps probe+exploit at max_steps),
    so the base figure is the full per-run training budget for most methods. Two methods add work on
    top of the base, accounted here when k (number of domains) is given:
      - DoGE: ~2*k*est_batches alignment micro-batches per round (rounds = max_steps/update_steps).
      - Skill-It: a one-time graph-learning cost of k single-skill proxies * probe_steps training steps.
    """
    N = n_params
    tok_per_step = cfg["batch_size"] * cfg["context_length"]
    D = cfg["max_steps"] * tok_per_step
    base = 6 * N * D
    rows = [dict(config=config_name, method="base (per run)", params=N, tokens=D,
                 flops=base, note="6*N*D; full per-run budget (incl. Aioli probes)")]
    if k:
        rounds = max(1, cfg["max_steps"] // cfg["update_steps"])
        doge_extra = 6 * N * (2 * k * cfg.get("doge_est_batches", 2) * rounds * tok_per_step)
        skillit_extra = 6 * N * (k * cfg.get("skillit_graph_probe_steps", 50) * tok_per_step)
        rows.append(dict(config=config_name, method="DoGE (total)", params=N, tokens=None,
                         flops=base + doge_extra,
                         note=f"base + {2*k*cfg.get('doge_est_batches',2)*rounds} alignment micro-batches"))
        rows.append(dict(config=config_name, method="Skill-It (total)", params=N, tokens=None,
                         flops=base + skillit_extra,
                         note=f"base + one-time graph learning: {k} proxies x {cfg.get('skillit_graph_probe_steps',50)} steps"))
    return pd.DataFrame(rows), base


def save_error_band_figures(agg, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # synthetic share with error bands
    plt.figure(figsize=(9, 5.5))
    for method, a in agg.items():
        plt.plot(a["rounds"], a["share_mean"], marker="o", markersize=3, label=method)
        if len(a["share_std"]) and np.any(a["share_std"] > 0):
            plt.fill_between(a["rounds"], a["share_mean"] - a["share_std"], a["share_mean"] + a["share_std"], alpha=0.15)
    plt.axhline(0.5, color="grey", lw=0.8, ls=":")
    plt.xlabel("Round"); plt.ylabel("Synthetic data share (sum p over synthetic)")
    plt.title("Synthetic share vs round (mean +/- std over seeds)")
    plt.ylim(0, 1.05); plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_synth_share.png", dpi=200, bbox_inches="tight"); plt.close()

    # L2 from uniform with error bands
    plt.figure(figsize=(9, 5.5))
    for method, a in agg.items():
        plt.plot(a["rounds"], a["l2_mean"], marker="d", markersize=3, label=method)
        if len(a["l2_std"]) and np.any(a["l2_std"] > 0):
            plt.fill_between(a["rounds"], a["l2_mean"] - a["l2_std"], a["l2_mean"] + a["l2_std"], alpha=0.15)
    plt.xlabel("Round"); plt.ylabel("L2 distance from uniform")
    plt.title("Divergence from uniform (mean +/- std over seeds)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_divergence.png", dpi=200, bbox_inches="tight"); plt.close()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Aioli synthetic-vs-real data-mixing runner")
    ap.add_argument("--config", choices=list(CONFIGS), default="HEAVY")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--scenario", choices=list(SCENARIOS) + ["ALL"], default="S2_9R_9S")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--eval-test", action="store_true", help="held-out test loss of each method's final model")
    ap.add_argument("--static-eval", action="store_true", help="retrain on each final static mixture and test (expensive)")
    ap.add_argument("--extra-heuristics", action="store_true",
                    help="also run the Gen-Delta / Improvement-Window val-loss heuristics (not DoGE/Skill-It)")
    ap.add_argument("--out-dir", default=COMMON["save_dir"])
    ap.add_argument("--split-seed", type=int, default=1234)
    args = ap.parse_args()

    cfg = dict(**COMMON, **CONFIGS[args.config])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    logger.info("config=%s device=%s seeds=%s scenario=%s", args.config, device, args.seeds, args.scenario)
    if device == "cpu":
        logger.warning("Running on CPU; use a GPU (Colab/Kaggle) for anything beyond SUPER_LIGHT.")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_domain_dataset = make_dataset_factory(tokenizer)

    scen_names = list(SCENARIOS) if args.scenario == "ALL" else [args.scenario]
    for scen_name in scen_names:
        spec = SCENARIOS[scen_name]
        logger.info("=== scenario %s ===", scen_name)
        # Domains are seed-independent (split_seed fixed); build once, reuse across seeds.
        domains = build_scenario_domains(tokenizer, load_domain_dataset, spec, cfg, split_seed=args.split_seed)
        is_synth = [d["is_synth"] for d in domains]
        pretty = [d["pretty"] for d in domains]

        per_seed = []
        for seed in args.seeds:
            logger.info("--- seed %d ---", seed)
            res = run_scenario_one_seed(domains, tokenizer, cfg, device, seed,
                                        run_baselines=not args.no_baselines,
                                        eval_test=args.eval_test, static_eval=args.static_eval,
                                        extra_heuristics=args.extra_heuristics)
            res["seed"] = seed
            per_seed.append(res)

        tag = f"{args.config}_{scen_name}"
        np.save(out / f"results_{tag}.npy", {"per_seed": per_seed, "is_synth": is_synth, "pretty": pretty,
                                             "cfg": cfg, "seeds": args.seeds}, allow_pickle=True)

        # aggregate + figures
        agg = aggregate_over_seeds(per_seed, is_synth)
        rows = []
        for method, a in agg.items():
            for i, r in enumerate(a["rounds"]):
                rows.append(dict(method=method, round=int(r),
                                 synth_share_mean=a["share_mean"][i], synth_share_std=a["share_std"][i],
                                 l2_mean=a["l2_mean"][i], l2_std=a["l2_std"][i]))
        pd.DataFrame(rows).to_csv(out / f"agg_{tag}_synth_share.csv", index=False)
        save_error_band_figures(agg, str(out / f"fig_{tag}"))

        # final per-domain weights mean/std
        fw = []
        for method in METHOD_ORDER + EXTRA_HEURISTICS:
            finals = [np.asarray(r["histories"][method]["p"][-1], dtype=float)
                      for r in per_seed if method in r["histories"]]
            if not finals:
                continue
            finals = np.stack(finals)
            for j, name in enumerate(pretty):
                fw.append(dict(method=method, domain=name, is_synth=bool(is_synth[j]),
                               weight_mean=float(finals[:, j].mean()), weight_std=float(finals[:, j].std())))
        pd.DataFrame(fw).to_csv(out / f"agg_{tag}_final_weights.csv", index=False)

        # per-seed per-round per-domain weight trajectories (for weight-evolution plots)
        wt = []
        for seed_res in per_seed:
            for method in METHOD_ORDER + EXTRA_HEURISTICS:
                h = seed_res["histories"].get(method)
                if h is None:
                    continue
                for r, p_vec in zip(h["round"], h["p"]):
                    p_arr = np.asarray(p_vec, dtype=float)
                    for j, (pname, synth) in enumerate(zip(pretty, is_synth)):
                        if j < len(p_arr):
                            wt.append(dict(method=method, seed=seed_res["seed"], round=int(r),
                                           pretty=pname, is_synth=bool(synth), weight=float(p_arr[j])))
        if wt:
            pd.DataFrame(wt).to_csv(out / f"agg_{tag}_weight_trajectories.csv", index=False)

        # per-seed per-round per-domain val-loss trajectories (only methods that log val_losses, e.g. Aioli)
        vlt = []
        for seed_res in per_seed:
            for method in METHOD_ORDER + EXTRA_HEURISTICS:
                h = seed_res["histories"].get(method)
                if h is None or "val_losses" not in h:
                    continue
                for r, losses in zip(h["round"], h["val_losses"]):
                    losses_arr = np.asarray(losses, dtype=float)
                    for j, (pname, synth) in enumerate(zip(pretty, is_synth)):
                        if j < len(losses_arr) and not np.isnan(losses_arr[j]):
                            vlt.append(dict(method=method, seed=seed_res["seed"], round=int(r),
                                            pretty=pname, is_synth=bool(synth), val_loss=float(losses_arr[j])))
        if vlt:
            pd.DataFrame(vlt).to_csv(out / f"agg_{tag}_val_loss_trajectories.csv", index=False)

        # held-out test loss aggregate
        if args.eval_test:
            tl = []
            for key in ("test_loss", "static_test_loss"):
                for method in METHOD_ORDER + EXTRA_HEURISTICS:
                    runs = [np.asarray(r[key][method], dtype=float) for r in per_seed if method in r.get(key, {})]
                    if not runs:
                        continue
                    runs = np.stack(runs)  # [seed, domain]
                    real_mask = ~np.asarray(is_synth, dtype=bool)
                    real_mean = float(np.nanmean(runs[:, real_mask])) if real_mask.any() else float("nan")
                    synth_mean = float(np.nanmean(runs[:, ~real_mask])) if (~real_mask).any() else float("nan")
                    tl.append(dict(eval=key, method=method,
                                   test_loss_mean=float(np.nanmean(runs)),
                                   test_loss_std=float(np.nanstd(np.nanmean(runs, axis=1))),
                                   real_mean=real_mean, synth_mean=synth_mean))
            if tl:
                pd.DataFrame(tl).to_csv(out / f"agg_{tag}_test_loss.csv", index=False)

        # FLOP table (uses params from first seed; k = number of domains for the per-method surcharges)
        n_params = per_seed[0].get("n_params") or 0
        df_flops, _ = flop_table(cfg, n_params, args.config, k=len(domains))
        df_flops.to_csv(out / f"flops_{args.config}.csv", index=False)
        try:
            md = df_flops.to_markdown(index=False)  # needs 'tabulate'
        except Exception:
            md = df_flops.to_string(index=False)
        (out / f"flops_{args.config}.md").write_text(md, encoding="utf-8")
        logger.info("saved artifacts for %s under %s", scen_name, out)

    logger.info("done.")


if __name__ == "__main__":
    main()
