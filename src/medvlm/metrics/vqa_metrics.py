"""VQA correctness metrics.

- closed questions (yes/no): exact match after normalisation.
- open questions: token-level F1 (VQA-RAD / SLAKE convention) + optional semantic
  similarity via sentence-transformers (skipped gracefully if the model can't load,
  e.g. offline smoke test).
"""
from __future__ import annotations

import re
import string
from typing import Any, Dict, Optional

_ARTICLES = {"a", "an", "the"}
_PUNCT = str.maketrans("", "", string.punctuation)

# Lazy singleton for the sentence-transformer.
_ST_MODEL = None
_ST_FAILED = False


def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = text.translate(_PUNCT)
    tokens = [t for t in re.split(r"\s+", text) if t and t not in _ARTICLES]
    return " ".join(tokens)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common: Dict[str, int] = {}
    for tok in p:
        if tok in g:
            common[tok] = min(p.count(tok), g.count(tok))
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p)
    recall = n_same / len(g)
    return 2 * precision * recall / (precision + recall)


def _semantic_sim(pred: str, gold: str) -> Optional[float]:
    global _ST_MODEL, _ST_FAILED
    if _ST_FAILED:
        return None
    try:
        if _ST_MODEL is None:
            from sentence_transformers import SentenceTransformer

            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        emb = _ST_MODEL.encode([pred, gold], convert_to_numpy=True, normalize_embeddings=True)
        return float((emb[0] * emb[1]).sum())
    except Exception:
        _ST_FAILED = True  # don't retry every item once it fails (e.g. offline)
        return None


def score_vqa(pred: str, item, use_semantic: bool = True) -> Dict[str, Any]:
    gold = item.answer
    is_closed = item.answer_type == "closed"
    em = exact_match(pred, gold)
    f1 = token_f1(pred, gold)
    result: Dict[str, Any] = {
        "answer_type": item.answer_type,
        "exact_match": em,
        "token_f1": f1,
        # Primary correctness signal: EM for closed, token-F1 for open.
        "correct": em if is_closed else f1,
    }
    if use_semantic and not is_closed:
        sim = _semantic_sim(pred, gold)
        if sim is not None:
            result["semantic_sim"] = sim
    return result
