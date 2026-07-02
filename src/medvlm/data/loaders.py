"""Dataset loaders -> unified item schema.

Every dataset is mapped to either a VQAItem or a ReportItem so models and metrics
never touch dataset-specific field names. HF mirror schemas differ (especially for
SLAKE and IU-Xray), so column resolution is done by fuzzy name matching with a
clear error if nothing plausible is found.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image

from ..utils.logging_utils import get_logger

log = get_logger("medvlm.data")

CLOSED_ANSWERS = {"yes", "no"}


@dataclass
class VQAItem:
    id: str
    image: Image.Image
    question: str
    answer: str
    answer_type: str  # 'closed' | 'open'
    modality: Optional[str] = None
    anatomy: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportItem:
    id: str
    image: Image.Image
    reference: str
    meta: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------- helpers


def _pick_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Return the first dataset column whose name matches any candidate (case-insensitive,
    substring-friendly)."""
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for cand in candidates:
        for lc, orig in lower.items():
            if cand in lc:
                return orig
    return None


def _to_pil(value: Any) -> Optional[Image.Image]:
    """Coerce a datasets image field (PIL, dict with 'bytes'/'path', or path str) to PIL."""
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, list) and value:  # IU-Xray often stores [frontal, lateral]
        return _to_pil(value[0])
    if isinstance(value, dict):
        if value.get("bytes"):
            import io

            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])
    if isinstance(value, str):
        try:
            return Image.open(value)
        except Exception:
            return None
    return None


def _infer_answer_type(answer: str, explicit: Optional[str]) -> str:
    if explicit:
        e = explicit.lower()
        if e in ("closed", "close", "yes/no", "binary"):
            return "closed"
        if e in ("open", "open-ended"):
            return "open"
    return "closed" if str(answer).strip().lower() in CLOSED_ANSWERS else "open"


def clean_report(text: str) -> str:
    """Strip de-identification placeholders and normalise whitespace (IU-Xray)."""
    text = re.sub(r"X{2,}", "", str(text))          # XXXX de-id tokens
    text = re.sub(r"_{2,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------------------------------------------------- loaders


def _load_hf_split(hf_id: str, split: str):
    from datasets import load_dataset

    try:
        return load_dataset(hf_id, split=split)
    except Exception as e:
        log.warning("split '%s' failed for %s (%s); falling back to first split", split, hf_id, e)
        ds = load_dataset(hf_id)
        first = list(ds.keys())[0]
        return ds[first]


def load_vqa(hf_id: str, split: str, n: Optional[int], language: Optional[str]) -> List[VQAItem]:
    ds = _load_hf_split(hf_id, split)
    cols = ds.column_names
    img_c = _pick_column(cols, ["image", "img", "img_name"])
    q_c = _pick_column(cols, ["question", "query"])
    a_c = _pick_column(cols, ["answer", "answers", "label"])
    at_c = _pick_column(cols, ["answer_type", "answertype", "qtype"])
    mod_c = _pick_column(cols, ["modality"])
    ana_c = _pick_column(cols, ["location", "anatomy", "organ"])
    lang_c = _pick_column(cols, ["q_lang", "language", "lang"])
    if not (q_c and a_c):
        raise ValueError(f"Could not find question/answer columns in {hf_id}. Columns: {cols}")

    items: List[VQAItem] = []
    for i, row in enumerate(ds):
        if language and lang_c and str(row.get(lang_c, "")).lower() not in (language, language[:2]):
            continue
        img = _to_pil(row.get(img_c)) if img_c else None
        if img is None:
            continue
        answer = str(row.get(a_c, "")).strip()
        item = VQAItem(
            id=f"{i}",
            image=img,
            question=str(row.get(q_c, "")).strip(),
            answer=answer,
            answer_type=_infer_answer_type(answer, str(row.get(at_c)) if at_c else None),
            modality=str(row.get(mod_c)) if mod_c else None,
            anatomy=str(row.get(ana_c)) if ana_c else None,
        )
        items.append(item)
        if n and len(items) >= n:
            break
    log.info("Loaded %d VQA items from %s (split=%s)", len(items), hf_id, split)
    return items


def load_report(hf_id: str, split: str, n: Optional[int]) -> List[ReportItem]:
    ds = _load_hf_split(hf_id, split)
    cols = ds.column_names
    img_c = _pick_column(cols, ["image", "images", "img"])
    rep_c = _pick_column(cols, ["report", "text", "caption"])
    find_c = _pick_column(cols, ["findings", "finding"])
    imp_c = _pick_column(cols, ["impression", "impressions"])

    items: List[ReportItem] = []
    for i, row in enumerate(ds):
        img = _to_pil(row.get(img_c)) if img_c else None
        if img is None:
            continue
        if rep_c:
            ref = str(row.get(rep_c, ""))
        else:
            parts = [str(row.get(find_c, "")) if find_c else "", str(row.get(imp_c, "")) if imp_c else ""]
            ref = " ".join(p for p in parts if p)
        ref = clean_report(ref)
        if not ref:
            continue
        items.append(ReportItem(id=f"{i}", image=img, reference=ref))
        if n and len(items) >= n:
            break
    log.info("Loaded %d report items from %s (split=%s)", len(items), hf_id, split)
    return items


def load_dataset_items(dataset_cfg: Dict[str, Any], split: Optional[str], n: Optional[int]):
    """Dispatch on the dataset's declared task. Returns (task, items)."""
    task = dataset_cfg.get("task", "vqa")
    hf_id = dataset_cfg["hf_id"]
    split = split or dataset_cfg.get("default_split", "test")
    if task == "vqa":
        return task, load_vqa(hf_id, split, n, dataset_cfg.get("language"))
    if task == "report":
        return task, load_report(hf_id, split, n)
    raise ValueError(f"Unknown task '{task}' for dataset {hf_id}")


def load_synthetic_vqa(n: int = 6) -> List[VQAItem]:
    """Offline items (blank images) for the CPU smoke test — no network needed."""
    items: List[VQAItem] = []
    qas = [
        ("Is there a pneumothorax?", "no", "closed"),
        ("Is the heart size normal?", "yes", "closed"),
        ("What abnormality is seen in the lung?", "nodule", "open"),
        ("Which organ is highlighted?", "liver", "open"),
        ("Is there pleural effusion?", "no", "closed"),
        ("What is the imaging modality?", "x-ray", "open"),
    ]
    for i in range(n):
        q, a, t = qas[i % len(qas)]
        img = Image.new("RGB", (64, 64), color=(i * 20 % 255, 30, 60))
        items.append(VQAItem(id=f"syn{i}", image=img, question=q, answer=a, answer_type=t))
    return items
