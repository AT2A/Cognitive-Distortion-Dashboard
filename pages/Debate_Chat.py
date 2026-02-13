from __future__ import annotations

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
    """Cache the Groq client to avoid recreating it on every rerun"""
    try:
        from groq import Groq
    except Exception as e:
        return None, str(e)
    if not api_key:
        return None, "Missing GROQ_API_KEY"
    return Groq(api_key=api_key), None


def groq_reply(client, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 260) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages[-24:],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


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
    """Cache conversation loading to avoid re-reading on every rerun"""
    df = pd.read_csv(conv_path)
    if "content" not in df.columns:
        raise ValueError(f"Conversation CSV missing 'content'. Found columns: {list(df.columns)}")
    if "speaker" not in df.columns and "role" in df.columns:
        df = df.rename(columns={"role": "speaker"})
    if "speaker" not in df.columns:
        raise ValueError(f"Conversation CSV missing 'speaker' or 'role'. Found columns: {list(df.columns)}")
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


st.set_page_config(page_title="Debate Chat", layout="wide")

st.markdown(
    """
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Your existing styles */
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
        key="chat_mode_radio"
    )
    st.session_state.app_mode = mode

    st.divider()
    st.header("Navigate")
    if st.button("Open Dashboard", key="nav_dashboard"):
        try:
            st.switch_page("Distortion_Dashboard.py")
        except Exception:
            try:
                st.switch_page("dashboard/Distortion_Dashboard.py")
            except Exception:
                st.info("Could not auto-navigate. Use the left multipage menu to open the Dashboard.")

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


if "chat" not in st.session_state:
    st.session_state.chat = []
if "turns" not in st.session_state:
    st.session_state.turns = []
if "last_export_path" not in st.session_state:
    st.session_state.last_export_path = None


SYSTEM_PROMPT = (
    "You are an AI debate partner trying to persuade the user toward a more nuanced perspective. "
    "Respond in 2–3 concise sentences with one counterpoint and one focused question. "
    "Do not summarize the user's argument. "
    "Vary your persuasion strategy across turns."
)

DEBATE_QUESTION = (
    "Overall, has modern technology improved people's ability to think critically, "
    "or has it weakened it?"
)

OPENING_MESSAGE = (
    "Let's start with a brief debate question.\n\n"
    f"**{DEBATE_QUESTION}**\n\n"
    "Take whichever position you agree with more and explain your reasoning."
)


if mode == "Demo":
    demo_path = _resolve_demo_conv()
    try:
        df = load_conversation_csv(demo_path)
        st.info(f"Demo transcript loaded from: {demo_path}")
        
        # Display conversation in chat format
        for _, r in df.iterrows():
            role = str(r["speaker"]).strip().casefold()
            role = "assistant" if role != "user" else "user"
            with st.chat_message(role):
                st.markdown(str(r["content"]))
    except Exception as e:
        st.error(f"Failed to load demo transcript: {e}")
    st.stop()


# Research mode - live chat
api_key = os.getenv("GROQ_API_KEY", "")
model = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")

client, err = get_groq_client(api_key)
if client is None:
    st.warning(f"Groq not configured: {err}. Set GROQ_API_KEY in .env to enable live chat.")


if not st.session_state.chat:
    st.session_state.chat.append({"role": "assistant", "content": OPENING_MESSAGE})
    append_turn("assistant", OPENING_MESSAGE)


# Display existing messages
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# Handle new user input
user_text = st.chat_input("Type your response…")

if user_text:
    # Add user message
    st.session_state.chat.append({"role": "user", "content": user_text})
    append_turn("user", user_text)

    # Prepare messages for API
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += st.session_state.chat

    # Get assistant response
    if client is None:
        assistant_text = "(Groq not configured.)"
    else:
        with st.spinner("Thinking..."):
            assistant_text = groq_reply(client, model, messages)

    # Add assistant message
    st.session_state.chat.append({"role": "assistant", "content": assistant_text})
    append_turn("assistant", assistant_text)
    st.rerun()