from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from groq import Groq


def load_env_upwards() -> None:
    if load_dotenv is None:
        return
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        envp = p / ".env"
        if envp.exists():
            load_dotenv(envp)
            return


def groq_reply(client, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 260) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages[-24:],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _resolve_repo_path(rel: str) -> Path:
    """
    Resolve a repo-relative path even if this page runs from dashboard/pages.
    """
    p = Path(rel)
    if p.exists():
        return p
    p2 = Path("dashboard") / rel
    if p2.exists():
        return p2
    here = Path(__file__).resolve()
    dash_root = here.parents[1] if len(here.parents) > 1 else here.parent
    p3 = dash_root / rel
    if p3.exists():
        return p3
    return p


def load_feature_pool(corpus_csv_rel: str = "data/distortion_corpus.csv", max_per_dist: int = 12) -> Dict[str, List[str]]:
    """
    Load distortion feature strings from the corpus CSV and return a dict:
    {distortion_name: [feature1, feature2, ...]}
    Supports common column names: distortion / distortion_name and feature / phrase / keyword / pattern.
    """
    corpus_path = _resolve_repo_path(corpus_csv_rel)
    df = pd.read_csv(corpus_path)
    dist_col = None
    for c in ["distortion", "distortion_name", "label", "category"]:
        if c in df.columns:
            dist_col = c
            break
    feat_col = None
    for c in ["feature", "phrase", "keyword", "pattern", "string"]:
        if c in df.columns:
            feat_col = c
            break
    if dist_col is None or feat_col is None:
        raise ValueError(f"Corpus CSV missing required columns. Found columns: {list(df.columns)}")
    df = df[[dist_col, feat_col]].dropna()
    df[dist_col] = df[dist_col].astype(str)
    df[feat_col] = df[feat_col].astype(str)
    pool: Dict[str, List[str]] = {}
    for d, sub in df.groupby(dist_col):
        feats = sub[feat_col].astype(str).tolist()
        seen = set()
        feats_uniq = []
        for f in feats:
            f2 = f.strip()
            if not f2:
                continue
            if f2.lower() in seen:
                continue
            seen.add(f2.lower())
            feats_uniq.append(f2)
        random.shuffle(feats_uniq)
        pool[str(d)] = feats_uniq[: max_per_dist]
    return pool


def build_feature_targets(feature_pool: Dict[str, List[str]], total_targets: int = 36) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for dist, feats in feature_pool.items():
        for f in feats:
            items.append((dist, f))
    random.shuffle(items)
    return items[:total_targets]


def simulate_conversation_fallback_groq(
    client,
    model: str,
    opening_assistant: str,
    debate_question: str,
    feature_targets: List[Tuple[str, str]],
    n_turns: int = 12,
) -> List[Dict[str, str]]:

    chat: List[Dict[str, str]] = [{"role": "assistant", "content": opening_assistant}]
    remaining = feature_targets.copy()
    last_asst = opening_assistant
    for _ in range(n_turns):
        take = remaining[:2] if remaining else []
        remaining = remaining[2:] if remaining else []
        user_messages = [
            {
                "role": "system",
                "content": (
                    "You are simulating a USER in a debate conversation for a research demo. "
                    "Persona: a sophomore Computer Science student at Binghamton University (Black male student). "
                    "Maintain LOW TRUST toward the assistant and push back on its persuasion attempts. "
                    "Write 1–3 natural sentences."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Debate question: {debate_question}\n\n"
                    f"Last assistant message: {last_asst}\n\n"
                    "Write the next USER message. Show low trust toward the assistant and push back on its persuasion attempt. "
                    "Do not reuse the same opening phrase as your previous USER message. "
                    "Include at least one of these phrases exactly (verbatim), and try to include both: "
                    + ", ".join([f'"{f}"' for _, f in take])
                ),
            },
        ]
        user_text = groq_reply(client, model, user_messages, temperature=0.8, max_tokens=220)
        chat.append({"role": "user", "content": user_text})
        asst_messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI debate partner trying to persuade the user toward a more nuanced perspective. "
                    "Respond in 2–3 concise sentences with one counterpoint and one focused question. "
                    "Do not summarize the user's argument. "
                    "Vary your persuasion strategy across turns."
                ),
            },
            {"role": "user", "content": f"User message: {user_text}\n\nReply in 2–3 sentences with one focused question."},
        ]
        asst_text = groq_reply(client, model, asst_messages, temperature=0.7, max_tokens=240)
        chat.append({"role": "assistant", "content": asst_text})
        last_asst = asst_text
    return chat


def export_turns_to_csv(chat_history: List[Dict[str, str]], path: str) -> str:
    turns = []
    for m in chat_history:
        turns.append({"speaker": m["role"], "content": m["content"]})
    df = pd.DataFrame(turns)
    df.insert(0, "n_turn", range(1, len(df) + 1))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return str(out)


if __name__ == "__main__":
    load_env_upwards()
    
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment")
        print("Create a .env file with: GROQ_API_KEY=your_key_here")
        exit(1)
    
    model = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
    
    DEBATE_QUESTION = (
        "Overall, has modern technology improved people's ability to think critically, "
        "or has it weakened it?"
    )
    
    OPENING_MESSAGE = (
        "Let's start with a brief debate question.\n\n"
        f"**{DEBATE_QUESTION}**\n\n"
        "Take whichever position you agree with more and explain your reasoning."
    )
    
    print("Generating conversation...")
    
    # Load corpus features
    pool = load_feature_pool("data/distortion_corpus.csv", max_per_dist=10)
    targets = build_feature_targets(pool, total_targets=max(30, 12 * 3))
    
    # Create Groq client
    client = Groq(api_key=api_key)
    
    # Generate conversation with 12 turns
    chat_hist = simulate_conversation_fallback_groq(
        client=client,
        model=model,
        opening_assistant=OPENING_MESSAGE,
        debate_question=DEBATE_QUESTION,
        feature_targets=targets,
        n_turns=12,
    )
    
    # Export to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"dashboard/data/generated_conversation_{timestamp}.csv"
    saved = export_turns_to_csv(chat_hist, output_path)
    
    print(f" Generated conversation saved to: {saved}")
    print(f"  {len(chat_hist)} messages total")