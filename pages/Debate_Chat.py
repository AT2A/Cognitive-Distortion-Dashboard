from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


USER_PROFILES_CSV = "dashboard/data/user_profiles.csv"
PROFILE_COLS = [
    "user_id", "last_updated", "sessions_count",
    "argument_style", "dominant_distortions",
    "effective_techniques", "confidence_notes",
]


def load_env_upwards() -> None:
    if load_dotenv is None:
        return
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        envp = p / ".env"
        if envp.exists():
            load_dotenv(envp)
            return


@st.cache_resource
def get_groq_client(api_key: str):
    try:
        from groq import Groq
    except Exception as e:
        return None, str(e)
    if not api_key:
        return None, "Missing GROQ_API_KEY"
    return Groq(api_key=api_key), None


def groq_reply(client, model: str, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 260) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages[-24:],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def groq_json(client, model: str, system: str, user: str, max_tokens: int = 350) -> dict:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return {}


# ── User Profile ──────────────────────────────────────────────────────────────

def _profile_path() -> Path:
    p = Path(USER_PROFILES_CSV)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_user_profile(user_id: str) -> dict:
    path = _profile_path()
    if path.exists():
        try:
            df = pd.read_csv(path)
            row = df[df["user_id"] == user_id]
            if not row.empty:
                return row.iloc[0].to_dict()
        except Exception:
            pass
    return {col: "" for col in PROFILE_COLS}


def save_user_profile(user_id: str, updates: dict) -> None:
    path = _profile_path()
    if path.exists():
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.DataFrame(columns=PROFILE_COLS)
    else:
        df = pd.DataFrame(columns=PROFILE_COLS)

    updates["user_id"] = user_id
    updates["last_updated"] = datetime.now().isoformat()

    if user_id in df["user_id"].values:
        for k, v in updates.items():
            if k in df.columns:
                df.loc[df["user_id"] == user_id, k] = v
    else:
        updates.setdefault("sessions_count", 1)
        new_row = {col: updates.get(col, "") for col in PROFILE_COLS}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(path, index=False)


def _profile_summary(profile: dict) -> str:
    if not any(profile.get(k) for k in ["argument_style", "dominant_distortions", "effective_techniques"]):
        return "No prior profile data for this user."
    parts = []
    if profile.get("argument_style"):
        parts.append(f"Argument style: {profile['argument_style']}")
    if profile.get("dominant_distortions"):
        parts.append(f"Common distortions: {profile['dominant_distortions']}")
    if profile.get("effective_techniques"):
        parts.append(f"Techniques that engaged them: {profile['effective_techniques']}")
    if profile.get("confidence_notes"):
        parts.append(f"Confidence pattern: {profile['confidence_notes']}")
    return ". ".join(parts)


# ── Agents ────────────────────────────────────────────────────────────────────

def run_stance_agent(client, model: str, user_message: str, recent_history: str, profile_sum: str) -> dict:
    system = (
        "You analyze debate messages to extract the user's stance. "
        "Return JSON with exactly these fields:\n"
        "- position: their stance in one sentence\n"
        "- argument_style: one of [emotional, logical, anecdotal, mixed]\n"
        "- distortions: list of cognitive distortions detected (e.g. catastrophizing, all-or-nothing). Empty list if none.\n"
        "- confidence: one of [high, medium, low]\n"
        "- weakness: the single weakest point in their argument (one sentence)\n"
        "- profile_update: one sentence about what this message reveals about how to persuade this user"
    )
    user = (
        f"User profile: {profile_sum}\n\n"
        f"Recent conversation:\n{recent_history}\n\n"
        f"Latest user message: {user_message}"
    )
    return groq_json(client, model, system, user, max_tokens=350)


def run_counter_agent(client, model: str, stance: dict, profile_sum: str) -> dict:
    system = (
        "You are a debate strategy expert. Given a stance analysis and user profile, "
        "choose the best counter-strategy. Return JSON with:\n"
        "- counter_argument: the specific argument to make (1-2 sentences)\n"
        "- technique: one of [socratic_questioning, counterexample, reframing, appeal_to_evidence, expose_contradiction]\n"
        "- target: the specific weakness or distortion to press on"
    )
    user = (
        f"User profile: {profile_sum}\n\n"
        f"Position: {stance.get('position', '')}\n"
        f"Argument style: {stance.get('argument_style', '')}\n"
        f"Weakness: {stance.get('weakness', '')}\n"
        f"Distortions: {stance.get('distortions', [])}"
    )
    return groq_json(client, model, system, user, max_tokens=300)


def run_orchestrator(client, model: str, stance: dict, counter: dict, turn: int, max_turns: int) -> str:
    system = (
        "Generate a focused tactical directive (2-3 sentences) for a debate AI. "
        "Tell it exactly what weakness to press on and which technique to use this turn. "
        "Plain text only — no JSON, no headers."
    )
    technique = counter.get("technique", "").replace("_", " ")
    user = (
        f"Turn {turn} of {max_turns}.\n"
        f"User's position: {stance.get('position', '')}\n"
        f"Their weakness: {stance.get('weakness', '')}\n"
        f"Counter argument: {counter.get('counter_argument', '')}\n"
        f"Technique: {technique}\n"
        f"Target: {counter.get('target', '')}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
            max_tokens=150,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def run_agent_pipeline(
    client, model: str, user_message: str,
    chat_history: List[dict], user_id: str,
    turn: int, max_turns: int,
) -> str:
    profile = load_user_profile(user_id)
    prof_sum = _profile_summary(profile)

    recent = "\n".join(
        f"{m['role'].title()}: {m['content']}"
        for m in chat_history[-6:]
    )

    stance = run_stance_agent(client, model, user_message, recent, prof_sum)
    if not stance:
        return ""

    counter = run_counter_agent(client, model, stance, prof_sum)

    # Update profile with what we learned this turn
    updates: dict = {}
    if stance.get("argument_style"):
        updates["argument_style"] = stance["argument_style"]
    if stance.get("distortions"):
        dist = stance["distortions"]
        updates["dominant_distortions"] = ", ".join(dist) if isinstance(dist, list) else str(dist)
    if stance.get("profile_update"):
        updates["confidence_notes"] = stance["profile_update"]
    if counter.get("technique"):
        updates["effective_techniques"] = counter["technique"].replace("_", " ")
    if updates:
        save_user_profile(user_id, updates)

    directive = run_orchestrator(client, model, stance, counter, turn, max_turns)
    return directive


# ── Existing helpers ──────────────────────────────────────────────────────────

DEMO_CONV_CANDIDATES = [
    "dashboard/data/Example_Convo.csv",
    "data/Example_Convo.csv",
]


def _resolve_path(rel_or_abs: str) -> str:
    p = Path(rel_or_abs)
    if p.exists():
        return str(p)
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        cand = base / rel_or_abs
        if cand.exists():
            return str(cand)
    cand2 = Path("dashboard") / rel_or_abs
    if cand2.exists():
        return str(cand2)
    return str(p)


def _resolve_demo_conv() -> str:
    for c in DEMO_CONV_CANDIDATES:
        p = Path(_resolve_path(c))
        if p.exists():
            return str(p)
    return _resolve_path(DEMO_CONV_CANDIDATES[0])


@st.cache_data(show_spinner=False)
def load_conversation_csv(conv_path: str) -> pd.DataFrame:
    df = pd.read_csv(conv_path)
    if "content" not in df.columns:
        raise ValueError(f"Conversation CSV missing 'content'. Found: {list(df.columns)}")
    if "speaker" not in df.columns and "role" in df.columns:
        df = df.rename(columns={"role": "speaker"})
    if "speaker" not in df.columns:
        raise ValueError(f"Conversation CSV missing 'speaker' or 'role'. Found: {list(df.columns)}")
    if "n_turn" not in df.columns:
        df.insert(0, "n_turn", range(1, len(df) + 1))
    df = df.sort_values("n_turn", kind="stable").reset_index(drop=True)
    return df


def append_turn(speaker: str, content: str) -> None:
    st.session_state.turns.append({"speaker": speaker, "content": content})


def export_turns_to_csv(path: str) -> str:
    df = pd.DataFrame(st.session_state.turns)
    df.insert(0, "n_turn", range(1, len(df) + 1))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return str(out)


# ── System prompt ─────────────────────────────────────────────────────────────

MAX_USER_TURNS = 12


def build_system_prompt(turn: int, directive: str = "") -> str:
    if turn <= 4:
        phase = "You are in the early phase (turn {turn} of {max}). Probe the user's reasoning — ask them to justify their assumptions and unpack vague claims.".format(turn=turn, max=MAX_USER_TURNS)
    elif turn <= 9:
        phase = "You are in the middle phase (turn {turn} of {max}). Directly challenge the weakest point in the user's argument with a concrete counterexample or contradicting fact.".format(turn=turn, max=MAX_USER_TURNS)
    else:
        phase = "You are in the final phase (turn {turn} of {max}). Press hard — expose contradictions in what the user has said across the debate and force them to defend their core position.".format(turn=turn, max=MAX_USER_TURNS)

    base = (
        "You are an AI debate partner. Your job is to challenge the user's position and maintain pressure throughout the debate.\n\n"
        "Rules:\n"
        "- Respond in exactly 2–3 sentences. No more.\n"
        "- End every response with one sharp, specific question.\n"
        "- Never agree with the user. Always find a flaw or counterexample in their argument.\n"
        "- If the user gives a vague or short answer, demand they be specific before moving on.\n"
        "- Never use pleasantries, summaries, or closing remarks.\n"
        "- Never break character or reference being an AI.\n\n"
        + phase
    )

    if directive:
        base += f"\n\n[AGENT DIRECTIVE — follow this for the current turn]\n{directive}"

    return base


DEBATE_QUESTION = (
    "Overall, has modern technology improved people's ability to think critically, "
    "or has it weakened it?"
)

OPENING_MESSAGE = (
    "Let's start with a brief debate question.\n\n"
    f"**{DEBATE_QUESTION}**\n\n"
    "Take whichever position you agree with more and explain your reasoning."
)


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Debate Chat", layout="wide")

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    .stDeployButton {display: none;}
    html, body, [class*="css"] { font-size: 20px !important; line-height: 1.55; }
    section[data-testid="stSidebar"] * { font-size: 18px !important; }
    input, textarea, select, button { font-size: 18px !important; }
    .stChatMessage { font-size: 19px !important; }
    .stCaption, small { font-size: 16px !important; opacity: 0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Debate Chat")

load_env_upwards()

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Demo"

with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "Choose mode",
        options=["Demo", "Research"],
        index=0 if st.session_state.app_mode == "Demo" else 1,
        key="chat_mode_radio",
    )
    st.session_state.app_mode = mode

    if mode == "Research":
        st.divider()
        st.header("Participant")
        user_id = st.text_input(
            "Your ID / name",
            value=st.session_state.get("participant_id", ""),
            key="participant_id_input",
            placeholder="e.g. participant_01",
        )
        st.session_state.participant_id = user_id
        if not user_id:
            st.caption("Enter an ID to enable the agent pipeline and user profile.")
    else:
        user_id = ""

    st.divider()
    st.header("Navigate")
    if st.button("Open Dashboard", key="nav_dashboard"):
        try:
            st.switch_page("Distortion_Dashboard.py")
        except Exception:
            try:
                st.switch_page("dashboard/Distortion_Dashboard.py")
            except Exception:
                st.info("Could not auto-navigate. Use the left multipage menu.")

    st.divider()
    st.header("Session")

    default_export = f"dashboard/data/chatbot_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    export_path = st.text_input("Export path", value=default_export, key="export_path_input")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset", key="reset_chat"):
            st.session_state.chat = []
            st.session_state.turns = []
            st.session_state.last_export_path = None
            st.rerun()
    with c2:
        if st.button("Export CSV", type="primary", key="export_csv"):
            if st.session_state.turns:
                saved = export_turns_to_csv(export_path)
                st.session_state.last_export_path = saved
                st.success("Exported.")

    if st.session_state.get("last_export_path"):
        st.caption("Use this path in the dashboard:")
        st.code(st.session_state.last_export_path)


# ── Session state init ────────────────────────────────────────────────────────

for k, default in [("chat", []), ("turns", []), ("last_export_path", None)]:
    if k not in st.session_state:
        st.session_state[k] = default


# ── Demo mode ─────────────────────────────────────────────────────────────────

if mode == "Demo":
    demo_path = _resolve_demo_conv()
    try:
        df = load_conversation_csv(demo_path)
        st.info(f"Demo transcript loaded from: {demo_path}")
        for _, r in df.iterrows():
            role = str(r["speaker"]).strip().casefold()
            role = "assistant" if role != "user" else "user"
            with st.chat_message(role):
                st.markdown(str(r["content"]))
    except Exception as e:
        st.error(f"Failed to load demo transcript: {e}")
    st.stop()


# ── Research mode ─────────────────────────────────────────────────────────────

api_key = os.getenv("GROQ_API_KEY", "")
model = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")

client, err = get_groq_client(api_key)
if client is None:
    st.warning(f"Groq not configured: {err}. Set GROQ_API_KEY in .env to enable live chat.")

if not st.session_state.chat:
    st.session_state.chat.append({"role": "assistant", "content": OPENING_MESSAGE})
    append_turn("assistant", OPENING_MESSAGE)

user_turn_count = sum(1 for t in st.session_state.turns if t["speaker"] == "user")
debate_complete = user_turn_count >= MAX_USER_TURNS

turns_remaining = MAX_USER_TURNS - user_turn_count
st.progress(min(user_turn_count / MAX_USER_TURNS, 1.0))
if debate_complete:
    st.caption(f"Debate complete — {MAX_USER_TURNS}/{MAX_USER_TURNS} turns used.")
else:
    st.caption(f"Turn {user_turn_count} / {MAX_USER_TURNS} — {turns_remaining} remaining")

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if debate_complete:
    st.success("You've reached the end of the debate. Export your conversation and open the dashboard to analyze it.")
    st.stop()

user_text = st.chat_input("Type your response…", disabled=debate_complete)

if user_text:
    st.session_state.chat.append({"role": "user", "content": user_text})
    append_turn("user", user_text)

    directive = ""
    if client is not None and user_id:
        with st.spinner("Analyzing stance…"):
            directive = run_agent_pipeline(
                client, model, user_text,
                st.session_state.chat[:-1],
                user_id,
                user_turn_count + 1,
                MAX_USER_TURNS,
            )

    messages = [{"role": "system", "content": build_system_prompt(user_turn_count, directive)}]
    messages += st.session_state.chat

    if client is None:
        assistant_text = "(Groq not configured.)"
    else:
        with st.spinner("Thinking…"):
            assistant_text = groq_reply(client, model, messages)

    st.session_state.chat.append({"role": "assistant", "content": assistant_text})
    append_turn("assistant", assistant_text)
    st.rerun()
