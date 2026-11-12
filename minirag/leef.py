"""Opt-in evidence exchange after B2's A1 packing. No evaluation labels at inference."""

import hashlib
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .utils import encode_string_by_tiktoken

BUDGET = 4000
DIM = 384
FEATURES = 9 * DIM + 14
VERSION = "leef-v0.1"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODES = ("support_pilot", "full", "drop_pilot", "edit_pilot", "rank_pilot")
DROP_BONUS = 0.25
ADD_COST = 0.05


@dataclass(frozen=True)
class Swap:
    incoming: str
    outgoing: str

    def apply(self, sources):
        return [self.incoming if cid == self.outgoing else cid for cid in sources]


@dataclass(frozen=True)
class Drop:
    outgoing: str

    def apply(self, sources):
        return [cid for cid in sources if cid != self.outgoing]


@dataclass(frozen=True)
class Add:
    incoming: str

    def apply(self, sources):
        return [*sources, self.incoming]


def feasible_drops(sources):
    return [Drop(cid) for cid in sources] if len(sources) > 1 else []


def feasible_adds(ranked, sources, costs):
    if not sources:
        return []
    selected = set(sources)
    remaining = BUDGET - sum(costs[c] for c in sources)
    return [Add(c) for c in ranked if c not in selected and costs[c] <= remaining]


def vectors_from_storage(storage):
    """Read NanoVectorDB's loaded matrix without normalizing it in place."""
    ids = [row["__id__"] for row in storage["data"]]
    matrix = np.asarray(storage["matrix"], dtype=np.float32)
    if matrix.shape != (len(ids), DIM) or len(set(ids)) != len(ids):
        raise ValueError("LEEF requires a valid 384-dimensional chunk index")
    digest = hashlib.sha256(matrix.tobytes())
    digest.update(json.dumps(ids, separators=(",", ":")).encode())
    return dict(zip(ids, matrix)), digest.hexdigest()


@lru_cache(maxsize=1)
def _encoder():
    from huggingface_hub import snapshot_download
    from safetensors.torch import load
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    path = Path(snapshot_download(ENCODER, revision=REVISION, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_config(AutoConfig.from_pretrained(path, local_files_only=True))
    # Read into RAM; the cached model has previously failed with mmap on this host.
    weights = load((path / "model.safetensors").read_bytes())
    weights.pop("embeddings.position_ids", None)
    model.load_state_dict(weights)
    return tokenizer, model.eval()


def embed_queries(queries):
    """Match the existing hf_embed single-query pooling; never download or call an API."""
    import torch

    tokenizer, model = _encoder()
    embeddings = []
    with torch.inference_mode():
        for query in queries:
            ids = tokenizer([query], return_tensors="pt", truncation=True).input_ids
            embeddings.append(model(ids).last_hidden_state.mean(dim=1)[0].numpy())
    return np.asarray(embeddings, dtype=np.float32).reshape(-1, DIM)


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (DIM,) or not np.isfinite(vector).all():
        raise ValueError("LEEF received an invalid embedding")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("LEEF received a zero or invalid embedding norm")
    return vector / norm


def feasible_swaps(ranked, sources, costs):
    selected = set(sources)
    if (len(ranked) > 60 or len(set(ranked)) != len(ranked)
            or len(selected) != len(sources) or not selected <= set(ranked)):
        raise ValueError("LEEF received invalid candidate or Sources IDs")
    total = sum(costs[c] for c in sources)
    if total > BUDGET:
        raise ValueError("LEEF received Sources exceeding A1@4000")
    return [
        Swap(d, e) for d in ranked if d not in selected for e in sources
        if total - costs[e] + costs[d] <= BUDGET
    ]


def action_features(query_vector, actions, ranked, sources, bm25, dense, vectors, costs):
    """Same features for the offline trainer and the live selector."""
    if not actions:
        return np.empty((0, FEATURES), dtype=np.float32)
    bpos = {c: i for i, c in enumerate(bm25, 1)}
    vpos = {c: i for i, c in enumerate(dense, 1)}
    q = _unit(query_vector)
    needed = set(sources) | {a.incoming for a in actions if not isinstance(a, Drop)}
    z = {c: _unit(vectors[c]) for c in needed}
    total = sum(costs[c] for c in sources)
    summed = np.stack([z[c] for c in sources]).sum(axis=0)
    h = {c: (summed - z[c]) / (len(sources) - 1) if len(sources) > 1
         else np.zeros(DIM, dtype=np.float32) for c in sources}

    def provenance(c):
        return (1 / (60 + bpos[c]) if c in bpos else 0,
                1 / (60 + vpos[c]) if c in vpos else 0,
                int(c in bpos), int(c in vpos))

    prov = {c: provenance(c) for c in needed}
    no_incoming = np.zeros(DIM, dtype=np.float32)
    no_provenance = (0, 0, 0, 0)
    mean_context = summed / len(sources)
    features = np.empty((len(actions), FEATURES), dtype=np.float32)
    for i, action in enumerate(actions):
        d = action.incoming if not isinstance(action, Drop) else None
        e = action.outgoing if not isinstance(action, Add) else None
        zd, ze = z.get(d, no_incoming), z.get(e, no_incoming)
        rest = h.get(e, mean_context)
        incoming_provenance = prov.get(d, no_provenance)
        outgoing_provenance = prov.get(e, no_provenance)
        scalars = (
            *incoming_provenance, *outgoing_provenance, costs.get(d, 0) / BUDGET, costs.get(e, 0) / BUDGET,
            total / BUDGET, (BUDGET - total) / BUDGET, len(sources) / len(ranked),
            int(incoming_provenance[2:] != outgoing_provenance[2:]),
        )
        features[i] = np.concatenate((
            q, zd, ze, rest, q * zd, q * ze, zd * rest, ze * rest,
            np.abs(zd - ze), scalars,
        ))
    return features


def new_model(mode="support_pilot"):
    import torch
    from torch import nn

    layers = [
        nn.Linear(FEATURES, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(),
        nn.Linear(64, 3), nn.Softmax(dim=-1) if mode == "rank_pilot" else nn.Sigmoid(),
    ]
    if mode == "rank_pilot":
        class FrozenStandardize(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("mean", torch.zeros(FEATURES))
                self.register_buffer("scale", torch.ones(FEATURES))

            def forward(self, x):
                return (x - self.mean) / self.scale

        layers.insert(0, FrozenStandardize())
    return nn.Sequential(*layers)


def predict(model, features):
    import torch

    with torch.inference_mode():
        return model(torch.from_numpy(features)).numpy()


def choose(actions, predictions, thresholds, mode):
    if mode not in MODES:
        raise ValueError(f"Unknown LEEF mode: {mode!r}")
    if predictions.shape != (len(actions), 3) or not np.isfinite(predictions).all():
        raise ValueError("Invalid LEEF model predictions")
    if mode == "edit_pilot":
        return choose_edit(actions, predictions, thresholds)
    if mode == "rank_pilot":
        return choose_ranked(actions, predictions, thresholds)
    best = None
    best_utility = thresholds["tau"]
    drop = None
    best_irrelevance = thresholds.get("drop_min", 1.0)
    # Actions are ordered by RRF rank of incoming, then position of outgoing.
    for action, (gain, loss, risk) in zip(actions, predictions):
        if isinstance(action, Drop):
            if mode == "drop_pilot" and loss <= thresholds["drop_eps"] and risk > best_irrelevance:
                best_irrelevance = float(risk)
                drop = {"action": "DROP", "outgoing": action.outgoing, "gain": 0.0,
                        "loss": float(loss), "irrelevance": best_irrelevance}
            continue
        if loss > thresholds["eps"] or (mode == "full" and risk > thresholds["rho"]):
            continue
        risk = float(risk) if mode == "full" else None
        utility = float(gain) - 2 * float(loss) - (risk if risk is not None else 0)
        if utility > best_utility:
            best_utility = utility
            best = {
                "action": "SWAP", "incoming": action.incoming, "outgoing": action.outgoing,
                "gain": float(gain), "loss": float(loss), "risk": risk, "utility": utility,
            }
    return best or drop or {"action": "KEEP"}


def choose_ranked(actions, predictions, thresholds):
    best = {"action": "KEEP"}
    best_score = thresholds["min_utility"]
    for action, (helpful, harmful, neutral) in zip(actions, predictions):
        if isinstance(action, Add):
            raise ValueError("LEEF rank_pilot supports KEEP, DROP and SWAP")
        utility = float(helpful) - 2 * float(harmful) - 0.1 * float(neutral)
        if (helpful < thresholds["min_helpful"] or harmful > thresholds["max_harmful"]
                or utility <= best_score):
            continue
        best_score = utility
        best = {"action": "DROP" if isinstance(action, Drop) else "SWAP",
                "outgoing": action.outgoing, "helpful": float(helpful),
                "harmful": float(harmful), "neutral": float(neutral), "utility": utility}
        if isinstance(action, Swap):
            best["incoming"] = action.incoming
    return best


def choose_edit(actions, predictions, thresholds):
    best = {"action": "KEEP"}
    best_utility = 0.0
    for action, (gain, loss, non_gold) in zip(actions, predictions):
        if isinstance(action, Add):
            utility = float(gain) - ADD_COST
            if gain <= thresholds["add_min"] or utility <= best_utility:
                continue
            best = {"action": "ADD", "incoming": action.incoming, "gain": float(gain), "loss": 0.0}
        elif isinstance(action, Drop):
            utility = DROP_BONUS * float(non_gold) - 2 * float(loss)
            if loss > thresholds["drop_eps"] or non_gold <= thresholds["drop_min"] or utility <= best_utility:
                continue
            best = {"action": "DROP", "outgoing": action.outgoing, "gain": 0.0,
                    "loss": float(loss), "irrelevance": float(non_gold)}
        else:
            utility = float(gain) - 2 * float(loss)
            if loss > thresholds["eps"] or utility <= thresholds["tau"] or utility <= best_utility:
                continue
            best = {"action": "SWAP", "incoming": action.incoming, "outgoing": action.outgoing,
                    "gain": float(gain), "loss": float(loss)}
        best["utility"] = best_utility = utility
    return best


@lru_cache(maxsize=4)
def _checkpoint(path, mode, mtime_ns, size):
    from safetensors import safe_open
    from safetensors.torch import load

    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if (metadata.get("version") != VERSION or metadata.get("mode") != mode
            or metadata.get("encoder_revision") != REVISION):
        raise ValueError("LEEF checkpoint version, mode, or encoder does not match")
    if mode == "full" and metadata.get("risk_trained") != "true":
        raise ValueError("Full LEEF requires a trained risk head")
    if mode == "drop_pilot" and metadata.get("drop_trained") != "true":
        raise ValueError("LEEF DROP requires a trained irrelevance head")
    if mode == "edit_pilot" and metadata.get("edit_trained") != "true":
        raise ValueError("LEEF EDIT requires a trained four-action checkpoint")
    if mode == "rank_pilot" and metadata.get("objective") != "action_outcome_softmax_v2":
        raise ValueError("LEEF ranking requires an action-outcome checkpoint")
    thresholds = json.loads(metadata["thresholds"])
    extra_thresholds = {"full": ("rho",), "drop_pilot": ("drop_eps", "drop_min"),
                        "edit_pilot": ("drop_eps", "drop_min", "add_min")}
    keys = (("min_helpful", "max_harmful", "min_utility") if mode == "rank_pilot"
            else ("eps", "tau") + extra_thresholds.get(mode, ()))
    for key in keys:
        if not 0 <= float(thresholds[key]) <= 1:
            raise ValueError(f"Invalid LEEF threshold: {key}")
        thresholds[key] = float(thresholds[key])
    contents = Path(path).read_bytes()
    model = new_model(mode)
    model.load_state_dict(load(contents))
    return model.eval(), thresholds, metadata, hashlib.sha256(contents).hexdigest()


def load_checkpoint(path, mode):
    if mode not in MODES or not path:
        raise ValueError("LEEF requires a valid mode and MINIRAG_LEEF_CHECKPOINT")
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return _checkpoint(str(path), mode, stat.st_mtime_ns, stat.st_size)


def run_exchange(query, ranked, sources, bm25, dense, texts, storage, mode, checkpoint):
    started = time.perf_counter()
    model, thresholds, metadata, digest = load_checkpoint(checkpoint, mode)
    vectors, index_digest = vectors_from_storage(storage)
    if metadata.get("index_sha256") != index_digest:
        raise ValueError("LEEF checkpoint does not match the chunk vector index")
    costs = {c: len(encode_string_by_tiktoken(texts[c])) for c in ranked}
    swaps = feasible_swaps(ranked, sources, costs)
    drops = feasible_drops(sources) if mode in ("drop_pilot", "edit_pilot", "rank_pilot") else []
    adds = feasible_adds(ranked, sources, costs) if mode == "edit_pilot" else []
    actions = swaps + adds + drops
    decision = {"action": "KEEP"}
    if actions:
        features = action_features(
            embed_queries([query])[0], actions, ranked, sources, bm25, dense, vectors, costs,
        )
        decision = choose(actions, predict(model, features), thresholds, mode)
    chosen = list(sources)
    if decision["action"] == "SWAP":
        chosen = Swap(decision["incoming"], decision["outgoing"]).apply(sources)
    elif decision["action"] == "DROP":
        chosen = Drop(decision["outgoing"]).apply(sources)
    elif decision["action"] == "ADD":
        chosen = Add(decision["incoming"]).apply(sources)
    return chosen, {
        **decision, "mode": mode, "checkpoint_sha256": digest, "thresholds": thresholds,
        "baseline_ids": list(sources), "feasible_swaps": len(swaps),
        **({"feasible_drops": len(drops)} if mode in ("drop_pilot", "edit_pilot", "rank_pilot") else {}),
        **({"feasible_adds": len(adds)} if mode == "edit_pilot" else {}),
        "tokens_before": sum(costs[c] for c in sources),
        "tokens_after": sum(costs[c] for c in chosen),
        "exchange_ms": round(1000 * (time.perf_counter() - started), 3),
    }
