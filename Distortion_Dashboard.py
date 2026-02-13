from __future__ import annotations

import html
import os
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt

import new_pipeline

try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except Exception:
    PLOTLY_EVENTS_AVAILABLE = False

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False


# -----------------------------
# Constants
# -----------------------------
HUMAN_SPEAKER = "user"
CORPUS_CSV = "data/distortion_corpus.csv"
DEMO_CONV_CANDIDATES = [
    "dashboard/data/Example_Convo.csv",
    "data/Example_Convo.csv",
]


# -----------------------------
# Path helpers
# -----------------------------
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


# -----------------------------
# Groq LLM utilities
# -----------------------------
def _load_env_upwards() -> None:
    if load_dotenv is None:
        return
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        envp = p / ".env"
        if envp.exists():
            load_dotenv(envp)
            return


@st.cache_resource
def _get_groq_client():
    """Get cached Groq client"""
    if not GROQ_AVAILABLE:
        return None
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None
    return Groq(api_key=api_key)


@st.cache_data(show_spinner=False)
def _get_llm_explanation(sentence: str, distortion: str, features: List[str], _client) -> str:
    """
    Get LLM explanation of how the features relate to the distortion.
    Cached to avoid re-querying for the same content.
    """
    if _client is None or not features:
        return ""
    
    features_str = ", ".join(f'"{f}"' for f in features[:5])  # Limit to first 5
    
    prompt = f"""You are analyzing cognitive distortions in conversation transcripts.

User said: "{sentence}"

The following phrases were detected as potential indicators of {distortion}: {features_str}

Briefly explain (2-3 sentences max) how these specific phrases might relate to {distortion}. Focus on the connection between the detected phrases and the distortion pattern. Be concise and educational."""

    try:
        response = _client.chat.completions.create(
            model=os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant explaining cognitive distortion patterns. Be concise and educational."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating explanation: {str(e)}"


# -----------------------------
# Data loading utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def _load_conv(conv_csv: str) -> pd.DataFrame:
    df = pd.read_csv(conv_csv)
    for col in ("n_turn", "speaker", "content"):
        if col not in df.columns:
            raise ValueError(f"Conversation CSV must contain {col} (found: {list(df.columns)})")
    df = df.sort_values("n_turn", kind="stable").reset_index(drop=True)
    return df


def _assistant_context_for_human_row(conv_df: pd.DataFrame, human_row_idx: int) -> Tuple[Optional[str], Optional[str]]:
    prev_msg = None
    next_msg = None

    for j in range(human_row_idx - 1, -1, -1):
        if str(conv_df.loc[j, "speaker"]).casefold() != HUMAN_SPEAKER.casefold():
            prev_msg = str(conv_df.loc[j, "content"])
            break

    for j in range(human_row_idx + 1, len(conv_df)):
        if str(conv_df.loc[j, "speaker"]).casefold() != HUMAN_SPEAKER.casefold():
            next_msg = str(conv_df.loc[j, "content"])
            break

    return prev_msg, next_msg


def _minmax_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=float)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if mx - mn <= 1e-9:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)


def _highlight_features(sentence: str, features: List[str]) -> str:
    if not sentence:
        return ""

    feats = [f for f in (features or []) if f]
    feats = sorted(set(feats), key=len, reverse=True)
    if not feats:
        return html.escape(sentence)

    pat = re.compile("|".join(re.escape(f) for f in feats), flags=re.IGNORECASE)

    out = []
    last = 0
    for m in pat.finditer(sentence):
        out.append(html.escape(sentence[last : m.start()]))
        out.append(
            '<span style="background:#3b3b3b; color:#fff; padding:0.08rem 0.20rem; '
            'border-radius:0.28rem; text-decoration: underline; text-decoration-thickness: 2px; '
            'text-underline-offset: 2px;">'
            f"{html.escape(m.group(0))}"
            "</span>"
        )
        last = m.end()
    out.append(html.escape(sentence[last:]))

    return "".join(out)


def _pills(items: List[str]) -> str:
    if not items:
        return "<em>None</em>"
    chips = []
    for it in items:
        chips.append(
            "<span style='border:1px solid rgba(255,255,255,0.18); "
            "background:rgba(255,255,255,0.03); padding:0.28rem 0.55rem; "
            "border-radius:0.6rem; font-size:1.0rem;'>"
            + html.escape(it)
            + "</span>"
        )
    return (
        "<div style='display:flex; flex-wrap:wrap; gap:0.45rem; margin:0.2rem 0 0.6rem 0;'>"
        + "".join(chips)
        + "</div>"
    )


def _card(title: str, body_html: str):
    st.markdown(
        f"""
<div style="border:1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.03);
            padding: 1.0rem;
            border-radius: 0.85rem;
            margin-bottom: 0.95rem;">
  <div style="font-weight:750; margin-bottom:0.45rem; font-size:1.05rem;">{html.escape(title)}</div>
  <div>{body_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# -----------------------------
# Cached compute
# -----------------------------
@st.cache_data(show_spinner=False)
def _compute_outputs(conv_csv: str, corpus_csv: str):
    conv_df = _load_conv(conv_csv)

    user_rows = conv_df[conv_df["speaker"].astype(str).str.casefold() == HUMAN_SPEAKER.casefold()].copy()
    user_rows = user_rows.reset_index()
    human_row_idxs = user_rows["index"].astype(int).tolist()

    df_wide, matches = new_pipeline.run_pipeline(conv_csv=conv_csv, corpus_csv=corpus_csv)
    return conv_df, human_row_idxs, df_wide, matches


@st.cache_data(show_spinner=False)
def _build_instances_by_dist(
    df_wide: pd.DataFrame,
    matches: Dict[str, Dict[str, List[str]]],
    dist_cols: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Precompute the radar Instances tables for each distortion once per dataset.
    This makes the distortion dropdown fast.
    """
    out: Dict[str, pd.DataFrame] = {}
    base = df_wide[["human_turn", "n_turn", "sentence"]].copy()

    # Pre-cast human_turn to int for stable joins/indexing
    base["human_turn"] = base["human_turn"].astype(int)

    for d in dist_cols:
        inst = base.copy()
        inst["hit_count"] = df_wide[d].fillna(0).astype(float)
        inst = inst[inst["hit_count"] > 0].sort_values(["hit_count", "human_turn"], ascending=[False, True])

        if inst.empty:
            out[d] = inst
            continue

        # Attach matched features (string) using O(#rows) lookup
        ht_list = inst["human_turn"].tolist()
        inst["matched_features"] = [
            ", ".join(matches.get(str(ht), {}).get(d, [])) for ht in ht_list
        ]
        out[d] = inst

    return out


# -----------------------------
# Plots
# -----------------------------
def _wrap_dist_label(label: str, width: int = 16) -> str:
    replacements = {
        "Dichotomous Reasoning": "Dichotomous\nReasoning",
        "Disqualifying the Positive": "Disqualifying\nthe Positive",
        "Emotional Reasoning": "Emotional\nReasoning",
        "Labeling and Mislabeling": "Labeling\nand Mislabeling",
        "Magnification and Minimization": "Magnification\nand Minimization",
        "Mental Filtering": "Mental\nFiltering",
        "Should statements": "Should\nstatements",
    }
    if label in replacements:
        return replacements[label]
    chunks = textwrap.wrap(str(label), width=width, break_long_words=False, break_on_hyphens=False)
    return "\n".join(chunks) if chunks else str(label)


@st.cache_data(show_spinner=False)
def make_radar_mpl(df_wide: pd.DataFrame, dist_cols: List[str]) -> plt.Figure:
    fig = plt.figure(figsize=(10.5, 7.2), dpi=150)
    if df_wide.empty or not dist_cols:
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "No data.", ha="center", va="center")
        fig.tight_layout()
        return fig

    totals = df_wide[dist_cols].fillna(0).astype(float).sum(axis=0).to_numpy()
    mx = float(totals.max()) if float(totals.max()) > 0 else 0.0

    ax = fig.add_subplot(111, polar=True)

    if mx <= 1e-9:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No corpus feature hits detected.", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    values = (totals / mx).tolist()
    n = len(dist_cols)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    values += values[:1]

    labels = [_wrap_dist_label(d) for d in dist_cols]

    ax.plot(angles, values, linewidth=2.5)
    ax.fill(angles, values, alpha=0.25)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"], fontsize=9)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)

    ax.grid(True, alpha=0.35)
    ax.spines["polar"].set_alpha(0.35)

    fig.tight_layout(pad=2.0)
    return fig


@st.cache_data(show_spinner=False)
def make_heatmap_minmax(df_wide: pd.DataFrame, dist_cols: List[str]) -> go.Figure:
    if df_wide.empty or not dist_cols:
        return go.Figure()

    x = df_wide["human_turn"].astype(int).tolist()

    Z = []
    for d in dist_cols:
        raw = df_wide[d].fillna(0).astype(float).to_numpy()
        z = _minmax_01(raw)
        Z.append(z.tolist())

    fig = go.Figure(
        data=go.Heatmap(
            z=Z,
            x=x,
            y=dist_cols,
            colorscale="Greys",
            zmin=0,
            zmax=1,
            hovertemplate=(
                "Human turn: %{x}<br>"
                "Distortion: %{y}<br>"
                "Min-max value: %{z:.3f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Human turn (user utterance index)", font=dict(size=14)),
            tickmode="linear",
            tickfont=dict(size=12),
            automargin=True,
        ),
        yaxis=dict(
            title=dict(text="Distortion type", font=dict(size=14)),
            tickfont=dict(size=12),
            automargin=True,
        ),
        margin=dict(l=200, r=20, t=10, b=60),
        height=450,
    )
    fig.update_yaxes(title_standoff=80)
    return fig


# -----------------------------
# Streamlit app
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Cognitive Distortion Dashboard", layout="wide")
    
    # Load environment variables
    _load_env_upwards()

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
        .stCaption, small { font-size: 16px !important; opacity: 0.9; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Cognitive Distortion Dashboard")
   

    # ---- Mode toggle (shared across pages) ----
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "Demo"

    # ---- Initialize session state early ----
    for k in ("df_wide", "matches", "conv_df", "human_row_idxs", "sel_turn", "sel_dist", "radar_sel_dist", "instances_by_dist", "data_loaded", "show_explanation"):
        if k not in st.session_state:
            st.session_state[k] = None if k != "show_explanation" else False

    with st.sidebar:
        st.header("Mode")
        mode = st.radio(
            "Choose mode", 
            options=["Demo", "Research"], 
            index=0 if st.session_state.app_mode == "Demo" else 1,
            key="mode_radio"
        )
        st.session_state.app_mode = mode

        st.divider()
        st.header("Inputs")

        if mode == "Demo":
            conv_csv = _resolve_demo_conv()
            st.text_input("Conversation CSV path", value=conv_csv, disabled=True, key="demo_path")
            auto_run = True
        else:
            conv_csv = st.text_input("Conversation CSV path", value=_resolve_demo_conv(), key="research_path")
            conv_csv = _resolve_path(conv_csv)
            auto_run = False

        st.divider()
        st.header("Navigate")
        if st.button("Open Debate Chat", key="nav_chat"):
            try:
                st.switch_page("pages/Debate_Chat.py")
            except Exception:
                try:
                    st.switch_page("dashboard/pages/Debate_Chat.py")
                except Exception:
                    st.info("Could not auto-navigate. Use the left multipage menu to open Debate Chat.")

        run_btn = st.button("Run feature extraction", type="primary", disabled=auto_run, key="run_extract")

    # Auto-load in Demo, manual in Research
    should_run = (auto_run and not st.session_state.data_loaded) or run_btn
    
    if should_run:
        with st.spinner("Running feature extraction..."):
            try:
                conv_df, human_row_idxs, df_wide, matches = _compute_outputs(conv_csv, CORPUS_CSV)

                st.session_state.df_wide = df_wide
                st.session_state.matches = matches
                st.session_state.conv_df = conv_df
                st.session_state.human_row_idxs = human_row_idxs
                st.session_state.data_loaded = True

                dist_cols = [c for c in df_wide.columns if c not in ("human_turn", "n_turn", "sentence")]
                st.session_state.sel_turn = 1
                st.session_state.sel_dist = dist_cols[0] if dist_cols else None
                st.session_state.radar_sel_dist = dist_cols[0] if dist_cols else None

                # Precompute instances tables once
                st.session_state.instances_by_dist = _build_instances_by_dist(df_wide, matches, dist_cols)

                if mode == "Research":
                    st.success("Feature extraction finished (cached).")
            except Exception as e:
                st.error(f"Failed to run feature extraction: {e}")
                return

    df_wide: Optional[pd.DataFrame] = st.session_state.df_wide
    matches: Optional[Dict[str, Dict[str, List[str]]]] = st.session_state.matches
    conv_df: Optional[pd.DataFrame] = st.session_state.conv_df
    human_row_idxs: Optional[List[int]] = st.session_state.human_row_idxs

    if df_wide is None or matches is None or conv_df is None or human_row_idxs is None:
        st.info("Select a CSV (Research mode) or use Demo mode to auto-load.")
        return

    dist_cols = [c for c in df_wide.columns if c not in ("human_turn", "n_turn", "sentence")]
    if not dist_cols:
        st.warning("No distortion columns found (check corpus CSV).")
        return

    instances_by_dist: Dict[str, pd.DataFrame] = st.session_state.instances_by_dist or {}

    tab_heatmap, tab_radar = st.tabs(["Heatmap", "Radar"])

    # =============================
    # Heatmap tab + Inspector
    # =============================
    with tab_heatmap:
        st.subheader("Heatmap")
        
        if PLOTLY_EVENTS_AVAILABLE:
            st.caption("Click on any cell in the heatmap to inspect it.")
        else:
            st.caption("Install streamlit-plotly-events for click functionality.")

        fig = make_heatmap_minmax(df_wide, dist_cols)
        
        # Heatmap with click handling
        if PLOTLY_EVENTS_AVAILABLE:
            # plotly_events renders the chart itself and returns click data
            selected = plotly_events(
                fig,
                click_event=True,
                hover_event=False,
                select_event=False,
                override_height=450,
                key="heatmap_click"
            )
            
            # Process clicks
            if selected and len(selected) > 0:
                point = selected[0]
                x_val = point.get("x")
                y_val = point.get("y")
                
                if x_val is not None and y_val is not None:
                    try:
                        turn_clicked = int(x_val)
                        dist_clicked = str(y_val)
                        
                        if 1 <= turn_clicked <= int(df_wide["human_turn"].max()) and dist_clicked in dist_cols:
                            # Check if selection changed - if so, reset explanation
                            if st.session_state.sel_turn != turn_clicked or st.session_state.sel_dist != dist_clicked:
                                st.session_state.show_explanation = False
                            
                            st.session_state.sel_turn = turn_clicked
                            st.session_state.sel_dist = dist_clicked
                    except (ValueError, TypeError):
                        pass
        else:
            st.plotly_chart(fig, use_container_width=True, key="heatmap_static")

        st.divider()
        st.subheader("Inspector")

        # Use current selections from session state (updated by clicks)
        current_turn = st.session_state.sel_turn or 1
        current_dist = st.session_state.sel_dist or dist_cols[0]

        # Display details for current selection
        row = df_wide[df_wide["human_turn"] == int(current_turn)]
        
        if row.empty:
            st.warning("Selected human turn not found.")
        else:
            sent = str(row["sentence"].iloc[0])
            feats = matches.get(str(int(current_turn)), {}).get(current_dist, [])
            
            # Simple info header
            st.markdown(f"**Turn {int(current_turn)}** • {html.escape(current_dist)}", unsafe_allow_html=True)
            st.markdown("---")
            
            # Content in columns
            left, right = st.columns([0.5, 0.5], gap="large")
            
            with left:
                st.markdown("### User sentence")
                _card("Sentence", _highlight_features(sent, feats))

                st.markdown("### Matched corpus features")
                _card("Features", _pills(feats))

            with right:
                user_row_idx = human_row_idxs[int(current_turn) - 1] if int(current_turn) - 1 < len(human_row_idxs) else None
                prev_asst, next_asst = (None, None)
                if user_row_idx is not None:
                    prev_asst, next_asst = _assistant_context_for_human_row(conv_df, user_row_idx)

                st.markdown("### Adjacent assistant messages")
                if prev_asst:
                    _card("Previous assistant message", html.escape(prev_asst).replace("\n", "<br>"))
                if next_asst:
                    _card("Next assistant message", html.escape(next_asst).replace("\n", "<br>"))
                if not prev_asst and not next_asst:
                    st.info("No assistant context found")
            
            # LLM Explanation section (full width below) - button triggered
            if feats and GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
                st.markdown("---")
                st.markdown("### LLM Explanation")
                st.caption("Generate an AI explanation of how the detected features relate to this cognitive distortion.")
                
                if st.button("🤖 Generate Explanation", type="secondary", key="gen_llm_btn"):
                    st.session_state.show_explanation = True
                
                # Show explanation if button was clicked
                if st.session_state.get("show_explanation", False):
                    groq_client = _get_groq_client()
                    if groq_client:
                        with st.spinner("Generating explanation..."):
                            explanation = _get_llm_explanation(sent, current_dist, feats, groq_client)
                        
                        if explanation:
                            st.markdown(
                                f"""
                                <div style="border:1px solid rgba(100,150,255,0.3);
                                            background: rgba(100,150,255,0.05);
                                            padding: 1rem;
                                            border-radius: 0.85rem;
                                            margin-top: 0.5rem;">
                                    <div style="font-size:0.95rem; line-height:1.6;">{html.escape(explanation)}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    else:
                        st.info("Groq API not configured. Set GROQ_API_KEY in .env")
        
        # Raw data table at bottom of heatmap tab
        st.markdown("---")
        st.subheader("Raw Feature Hit Data")
        st.caption("Complete feature extraction results for all user turns and distortions.")
        st.dataframe(
            df_wide,
            use_container_width=True,
            hide_index=True,
        )

    # =============================
    # Radar tab
    # =============================
    with tab_radar:
        st.subheader("Radar")   

        radar_fig = make_radar_mpl(df_wide, dist_cols)
        _c1, _c2, _c3 = st.columns([1.2, 1.6, 1.2])
        with _c2:
            st.pyplot(radar_fig, use_container_width=True)

        # Dropdown with on_change callback to force update
        default_idx = 0
        if st.session_state.radar_sel_dist and st.session_state.radar_sel_dist in dist_cols:
            default_idx = dist_cols.index(st.session_state.radar_sel_dist)
        
        def update_radar_dist():
            # Force update session state from widget
            st.session_state.radar_sel_dist = st.session_state.radar_dropdown_widget
        
        sel_dist_radar = st.selectbox(
            "Selected distortion",
            options=dist_cols,
            index=default_idx,
            key="radar_dropdown_widget",
            on_change=update_radar_dist
        )
        
        # Use the value from session state (updated by callback)
        current_radar_dist = st.session_state.get("radar_sel_dist", sel_dist_radar)

        # Get precomputed instances for current selection
        inst = instances_by_dist.get(current_radar_dist)
        
        st.markdown("### Instances")
        if inst is None or inst.empty:
            st.info("No user sentences had feature hits for this distortion.")
        else:
            st.dataframe(
                inst[["human_turn", "n_turn", "hit_count", "sentence", "matched_features"]],
                use_container_width=True,
                hide_index=True,
            )
        
        # Raw data table at bottom of radar tab
        st.markdown("---")
        st.subheader("Raw Feature Hit Data")
        st.caption("Complete feature extraction results for all user turns and distortions.")
        st.dataframe(
            df_wide,
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()