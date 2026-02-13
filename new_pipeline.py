from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


HUMAN_SPEAKER = "user"
DEFAULT_CORPUS_CSV = "data/distortion_corpus.csv"

# Preferred order (your paper-friendly ordering)
CATEGORY_ORDER = [
    "catastrophizing",
    "dichotomous_reasoning",
    "disqualifying_the_positive",
    "emotional_reasoning",
    "fortune-telling",
    "labeling_&_mislabeling",
    "magnification_&_minimization",
    "mental-filtering",
    "mindreading",
    "overgeneralizing",
    "personalizing",
    "should_statement",
]



# Text normalization + matching
def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_feature(feat: str) -> str:
    feat = str(feat or "").strip()
    feat = feat.strip(" \t\r\n.,;:!?")
    return feat


def pretty_from_category(cat: str) -> str:
    """
    Convert corpus categories like 'dichotomous_reasoning' into display names like
    'Dichotomous Reasoning', matching your preferred capitalization.
    """
    c = str(cat or "").strip()
    if not c:
        return c

    # Normalize separators
    c = c.replace("__", "_")
    c = c.replace("_&_", " and ").replace("&", " and ")
    c = c.replace("_", " ").replace("-", " ")
    c = re.sub(r"\s+", " ", c).strip()

    # Special-case should statements label
    if c.casefold() in {"should statement", "should statements", "should"}:
        return "Should statements"

    # Title-case each token
    title = " ".join([w[:1].upper() + w[1:] for w in c.split(" ") if w])

    # Prefer lower-case articles/conjunctions like your list
    title = title.replace(" And ", " and ").replace(" The ", " the ")

    # Optional tweak: Fortune-telling style
    if title.casefold() == "fortune telling":
        return "Fortune-telling"

    return title


def _count_hits(sentence: str, feature_phrases: List[str]) -> Tuple[int, List[str]]:
    """
    Count occurrences of each feature phrase in the sentence (case-insensitive).
    Returns:
      total_hits, matched_features_unique
    """
    s = _normalize_text(sentence)
    total = 0
    matched: List[str] = []

    for feat in feature_phrases:
        f_raw = _clean_feature(feat)
        if not f_raw:
            continue

        f = _normalize_text(f_raw)

        pat = r"(?<!\w)" + re.escape(f) + r"(?!\w)"
        hits = re.findall(pat, s)
        if hits:
            total += len(hits)
            matched.append(f_raw)

    # Deduplicate but preserve order
    seen = set()
    matched_unique = []
    for m in matched:
        if m not in seen:
            seen.add(m)
            matched_unique.append(m)

    return int(total), matched_unique



# Load corpus
def load_corpus(
    corpus_csv: str = DEFAULT_CORPUS_CSV,
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, str]]:
    """
    Returns:
      categories_sorted,
      features_by_category (raw category keys),
      pretty_name_by_category
    """
    p = Path(corpus_csv)
    if not p.exists():
        raise FileNotFoundError(f"Corpus CSV not found: {p}")

    df = pd.read_csv(p)
    if "category" not in df.columns or "feature" not in df.columns:
        raise ValueError("Corpus CSV must contain columns: category, feature")

    cats_in_file = df["category"].dropna().astype(str).unique().tolist()

    # Respect preferred order where possible
    cats = [c for c in CATEGORY_ORDER if c in cats_in_file] + sorted([c for c in cats_in_file if c not in CATEGORY_ORDER])

    feats: Dict[str, List[str]] = {}
    pretty: Dict[str, str] = {}

    for c in cats:
        feats[c] = df.loc[df["category"].astype(str) == c, "feature"].dropna().astype(str).tolist()
        pretty[c] = pretty_from_category(c)

    return cats, feats, pretty


# -----------------------------
# Main extraction
# -----------------------------
def run_pipeline(
    conv_csv: str,
    corpus_csv: str = DEFAULT_CORPUS_CSV,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[str]]]]:
    """
    Feature-only extraction.

    Returns:
      df_wide: columns = human_turn, n_turn, sentence, <pretty distortion columns...>
      matches: dict keyed by human_turn (str) -> {pretty distortion: [matched features]}
    """
    conv_path = Path(conv_csv)
    if not conv_path.exists():
        raise FileNotFoundError(f"Conversation CSV not found: {conv_path}")

    df = pd.read_csv(conv_path)
    for col in ("n_turn", "speaker", "content"):
        if col not in df.columns:
            raise ValueError(f"Conversation CSV must contain column '{col}' (found: {list(df.columns)})")

    # Preserve original order among identical n_turn values
    df = df.sort_values("n_turn", kind="stable").reset_index(drop=True)

    # Filter human turns (user)
    df_h = df[df["speaker"].astype(str).str.casefold() == HUMAN_SPEAKER.casefold()].copy()
    df_h = df_h.reset_index(drop=True)
    df_h["human_turn"] = range(1, len(df_h) + 1)
    df_h["sentence"] = df_h["content"].astype(str)

    cats, feats_by_cat, pretty_by_cat = load_corpus(corpus_csv)

    out_rows: List[Dict[str, object]] = []
    matches: Dict[str, Dict[str, List[str]]] = {}

    for _, r in df_h.iterrows():
        human_turn = int(r["human_turn"])
        row: Dict[str, object] = {
            "human_turn": human_turn,
            "n_turn": int(r["n_turn"]),
            "sentence": str(r["sentence"]),
        }

        turn_matches: Dict[str, List[str]] = {}

        for c in cats:
            pretty = pretty_by_cat[c]
            hit_count, matched_feats = _count_hits(row["sentence"], feats_by_cat.get(c, []))
            row[pretty] = int(hit_count)
            if hit_count > 0:
                turn_matches[pretty] = matched_feats

        if turn_matches:
            matches[str(human_turn)] = turn_matches

        out_rows.append(row)

    df_wide = pd.DataFrame(out_rows)

    distortion_cols = [pretty_by_cat[c] for c in cats]
    df_wide = df_wide[["human_turn", "n_turn", "sentence"] + distortion_cols]

    return df_wide, matches


def save_outputs(
    df_wide: pd.DataFrame,
    matches: Dict[str, Dict[str, List[str]]],
    out_dir: str,
    conv_csv: str,
) -> Tuple[str, str]:
    """
    Saves:
      <stem>__featurehits.csv
      <stem>__matches.json
    Returns (csv_path, json_path)
    """
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    stem = Path(conv_csv).stem

    csv_path = out_p / f"{stem}__featurehits.csv"
    json_path = out_p / f"{stem}__matches.json"

    df_wide.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(matches, indent=2, ensure_ascii=False), encoding="utf-8")

    return str(csv_path), str(json_path)
