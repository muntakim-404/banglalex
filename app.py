"""
BanglaLex — Prototype Demo Interface
======================================
Streamlit-based single-page prototype demonstrating the full
four-agent BanglaLex pipeline.

Run from the project root:
    streamlit run app.py

Requirements:
    streamlit>=1.35.0
"""

import sys
import time
from pathlib import Path

import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title = "BanglaLex · Legal AI for Bangladesh",
    page_icon  = "⚖",
    layout     = "centered",
)

# ── Session state init ─────────────────────────────────────────────────────────
if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""
if "result" not in st.session_state:
    st.session_state["result"] = None

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: system-ui, -apple-system, sans-serif;
    background-color: #F6F7F9 !important;
    color: #1E2535;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 760px; }

.banglalex-wordmark {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 0.2rem;
}
.banglalex-wordmark h1 {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: #1A3A5C;
    letter-spacing: -0.5px;
    margin: 0;
}
.banglalex-tagline {
    color: #5A6A80;
    font-size: 0.95rem;
    margin-bottom: 1.8rem;
    border-bottom: 2px solid #C8A951;
    padding-bottom: 1rem;
}
.pipeline-rail {
    display: flex;
    align-items: stretch;
    margin: 1.5rem 0 1rem 0;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #D8DDE8;
}
.pipeline-stage {
    flex: 1;
    padding: 10px 8px;
    text-align: center;
    font-size: 0.75rem;
    font-weight: 500;
    background: #ECEEF2;
    color: #8090A8;
    border-right: 1px solid #D8DDE8;
    line-height: 1.4;
}
.pipeline-stage:last-child { border-right: none; }
.pipeline-stage.active { background: #1A3A5C; color: #C8A951; font-weight: 600; }
.pipeline-stage.done   { background: #E8F2EC; color: #2A7A46; font-weight: 600; }
.pipeline-stage .stage-label {
    display: block; font-size: 0.65rem; font-weight: 400;
    opacity: 0.75; margin-top: 2px;
}
.outcome-badge {
    display: block; width: 100%; padding: 18px 24px; border-radius: 8px;
    text-align: center; font-family: Georgia, serif; font-size: 1.6rem;
    font-weight: 600; letter-spacing: 1px; margin: 1.5rem 0 1rem 0;
}
.outcome-favorable   { background:#E6F4EC; color:#1B6B3A; border:2px solid #2A7A46; }
.outcome-unfavorable { background:#FBEAEA; color:#8B1E1E; border:2px solid #B22222; }
.outcome-uncertain   { background:#FDF6E3; color:#7A5C00; border:2px solid #C8A951; }
.result-card {
    background: white; border-radius: 8px; padding: 20px 24px;
    margin-bottom: 1rem; border: 1px solid #D8DDE8;
}
.result-card-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; color: #8090A8; margin-bottom: 8px;
}
.result-card-content { font-size: 0.97rem; line-height: 1.65; color: #2C3447; }
.probability-statement {
    font-family: Georgia, serif; font-size: 1.05rem;
    color: #1A3A5C; font-style: italic; line-height: 1.7;
}
.statute-chip {
    display: inline-block; background: #EEF1F7; border: 1px solid #C8D0E0;
    border-radius: 4px; padding: 4px 10px; font-size: 0.78rem;
    color: #3A4A6A; margin: 3px; font-family: monospace;
}
.next-step {
    display: flex; gap: 12px; align-items: flex-start; padding: 8px 0;
    border-bottom: 1px solid #ECEEF2; font-size: 0.93rem; color: #2C3447;
}
.next-step:last-child { border-bottom: none; }
.next-step-num {
    background: #1A3A5C; color: #C8A951; border-radius: 50%;
    width: 22px; height: 22px; display: flex; align-items: center;
    justify-content: center; font-size: 0.7rem; font-weight: 700;
    flex-shrink: 0; margin-top: 1px;
}
.urgency-banner {
    background: #FDF3E4; border-left: 4px solid #C8A951;
    padding: 10px 16px; border-radius: 0 6px 6px 0;
    font-size: 0.88rem; color: #6B4A00; margin-top: 0.5rem;
}
.conf-bar-track {
    background: #ECEEF2; border-radius: 4px; height: 6px;
    margin-top: 6px; overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #1A3A5C, #C8A951);
}
.example-label {
    font-size: 0.75rem; color: #8090A8; font-weight: 500;
    letter-spacing: 0.5px; margin-bottom: 6px; text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ── Pipeline loader (cached — loads once, reused across reruns) ────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(model_name: str):
    from src.agents.pipeline import BanglaLexPipeline
    return BanglaLexPipeline(
        kb_dir       = str(REPO_ROOT / "data" / "knowledge_base"),
        cases_path   = str(REPO_ROOT / "data" / "annotated" / "cases_augmented.json"),
        model_name   = model_name,
        k_statutes   = 6,
        n_precedents = 4,
    )


# ── Pipeline rail renderer ─────────────────────────────────────────────────────
def render_rail(stages_done: int, active: int) -> str:
    stages = [
        ("Consultation", "Agent 1"),
        ("Knowledge",    "Agent 2"),
        ("Judgment",     "Agent 3"),
        ("Explanation",  "Agent 4"),
    ]
    html = '<div class="pipeline-rail">'
    for i, (name, label) in enumerate(stages):
        if i < stages_done:
            cls, icon = "done",   "✓ "
        elif i == active:
            cls, icon = "active", "● "
        else:
            cls, icon = "",       ""
        html += (
            f'<div class="pipeline-stage {cls}">'
            f'{icon}{name}'
            f'<span class="stage-label">{label}</span>'
            f'</div>'
        )
    return html + "</div>"


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banglalex-wordmark">
    <span style="font-size:1.8rem">⚖</span>
    <h1>BanglaLex</h1>
</div>
<div class="banglalex-tagline">
    Multi-Agent Legal AI for Bangladesh &nbsp;·&nbsp; আইনি সহায়তা সিস্টেম
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    model = st.selectbox(
        "LLM Backbone",
        options=[
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "gemini-3.1-flash-lite",
        ],
        index=0,
    )
    st.markdown("---")
    st.markdown(
        "<small>BanglaLex — B.Sc. Thesis Prototype<br>"
        "CUET, Dept. of CSE, 2026</small>",
        unsafe_allow_html=True,
    )


# ── Example queries ────────────────────────────────────────────────────────────
EXAMPLES = [
    "My landlord locked me out without notice and refused to return my security deposit. I had a written rental agreement.",
    "আমার মালিক আমার দুই মাসের বেতন দিচ্ছে না এবং কোনো কারণ ছাড়াই চাকরি থেকে বরখাস্ত করেছে।",
    "I signed a contract for land purchase and paid half the amount, but the seller is now refusing to complete the sale.",
]

st.markdown('<div class="example-label">Try an example</div>', unsafe_allow_html=True)
ex_cols = st.columns(3)
for i, (col, label) in enumerate(zip(ex_cols, ["Land (EN)", "Service (BN)", "Contract (EN)"])):
    with col:
        if st.button(label, use_container_width=True, key=f"ex_{i}"):
            # Fix: write directly to session state key, then rerun
            st.session_state["query_input"] = EXAMPLES[i]
            st.session_state["result"] = None  # clear old result
            st.rerun()


# ── Query input — key-based, no value= argument ────────────────────────────────
# Using key= lets Streamlit manage the widget state across reruns.
# Never pass value= here — that's what caused the can't-type bug.
query = st.text_area(
    "Describe your legal situation",
    key="query_input",
    height=120,
    placeholder=(
        "Describe your legal situation in English or Bangla…\n"
        "আপনার আইনি সমস্যা বাংলা বা ইংরেজিতে লিখুন…"
    ),
    label_visibility="collapsed",
)

analyse_btn = st.button(
    "⚖  Analyse Case",
    type="primary",
    use_container_width=True,
    disabled=not query.strip(),
)


# ── Run pipeline ───────────────────────────────────────────────────────────────
if analyse_btn and query.strip():
    st.session_state["result"] = None   # clear previous result
    rail_placeholder = st.empty()

    try:
        pipeline = load_pipeline(model)
    except Exception as e:
        st.error(
            f"Failed to load pipeline: {e}\n\n"
            "Run from the project root:  `streamlit run app.py`"
        )
        st.stop()

    with st.status("Running BanglaLex pipeline…", expanded=True) as status:

        rail_placeholder.markdown(render_rail(0, 0), unsafe_allow_html=True)
        st.write("**[1/4] Client Consultation Agent** — extracting case structure…")
        t0 = time.time()
        case_dict = pipeline.agent1.process(query)
        st.write(
            f"  ✓  Domain: **{case_dict.get('domain','?').upper()}** · "
            f"Language: **{case_dict.get('language','?').upper()}** · "
            f"Urgency: **{case_dict.get('urgency','?').upper()}**  "
            f"({time.time()-t0:.1f}s)"
        )

        rail_placeholder.markdown(render_rail(1, 1), unsafe_allow_html=True)
        st.write("**[2/4] Knowledge Integration Layer** — retrieving statutes…")
        t0 = time.time()
        enriched = pipeline.agent2.enrich(case_dict)
        n_ret = len(enriched.get("retrieved_statutes", []))
        n_app = len(enriched.get("applicable_statutes", []))
        st.write(
            f"  ✓  Retrieved **{n_ret}** sections · "
            f"Filtered to **{n_app}** applicable  ({time.time()-t0:.1f}s)"
        )

        rail_placeholder.markdown(render_rail(2, 2), unsafe_allow_html=True)
        st.write("**[3/4] Legal Judgment Agent** — predicting outcome…")
        t0 = time.time()
        analysis = pipeline.agent3.predict(enriched)
        outcome    = analysis.get("predicted_outcome", "uncertain")
        confidence = analysis.get("confidence", 0.5)
        st.write(
            f"  ✓  Prediction: **{outcome.upper()}** · "
            f"Confidence: **{confidence*100:.0f}%**  ({time.time()-t0:.1f}s)"
        )

        rail_placeholder.markdown(render_rail(3, 3), unsafe_allow_html=True)
        st.write("**[4/4] Output Agent** — generating plain-language explanation…")
        t0 = time.time()
        output = pipeline.agent4.generate(analysis)
        st.write(f"  ✓  Done  ({time.time()-t0:.1f}s)")

        rail_placeholder.markdown(render_rail(4, -1), unsafe_allow_html=True)
        status.update(label="Analysis complete ✓", state="complete", expanded=False)

    # Store results in session state so they survive reruns
    st.session_state["result"] = {
        "case_dict": case_dict,
        "enriched":  enriched,
        "analysis":  analysis,
        "output":    output,
        "outcome":   outcome,
        "confidence": confidence,
    }


# ── Display results (from session state, persists across reruns) ───────────────
if st.session_state["result"]:
    r          = st.session_state["result"]
    outcome    = r["outcome"]
    confidence = r["confidence"]
    output     = r["output"]
    analysis   = r["analysis"]
    enriched   = r["enriched"]
    case_dict  = r["case_dict"]

    badge_class = {"favorable": "outcome-favorable", "unfavorable": "outcome-unfavorable"}.get(
        outcome, "outcome-uncertain"
    )
    badge_text  = {
        "favorable":   "✓  FAVORABLE OUTCOME",
        "unfavorable": "✗  UNFAVORABLE OUTCOME",
    }.get(outcome, "?  UNCERTAIN — SEEK LEGAL ADVICE")

    st.markdown(
        f'<div class="outcome-badge {badge_class}">{badge_text}</div>',
        unsafe_allow_html=True,
    )

    conf_pct = int(confidence * 100)
    st.markdown(
        f'<div style="text-align:center;font-size:0.8rem;color:#8090A8;margin-bottom:4px;">'
        f'Model confidence: {conf_pct}%</div>'
        f'<div class="conf-bar-track">'
        f'<div class="conf-bar-fill" style="width:{conf_pct}%"></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    prob_stmt = output.get("probability_statement", "")
    if prob_stmt:
        st.markdown(
            f'<div class="result-card"><div class="result-card-label">Assessment</div>'
            f'<div class="probability-statement">{prob_stmt}</div></div>',
            unsafe_allow_html=True,
        )

    plain_reasoning = output.get("plain_reasoning", "")
    if plain_reasoning:
        st.markdown(
            f'<div class="result-card"><div class="result-card-label">Legal Reasoning</div>'
            f'<div class="result-card-content">{plain_reasoning}</div></div>',
            unsafe_allow_html=True,
        )

    next_steps = output.get("next_steps", [])
    if next_steps:
        steps_html = "".join(
            f'<div class="next-step">'
            f'<div class="next-step-num">{i+1}</div><div>{step}</div></div>'
            for i, step in enumerate(next_steps)
        )
        st.markdown(
            f'<div class="result-card"><div class="result-card-label">Recommended Next Steps</div>'
            f'{steps_html}</div>',
            unsafe_allow_html=True,
        )

    urgency_note = output.get("urgency_note")
    if urgency_note:
        st.markdown(
            f'<div class="urgency-banner">⚠ {urgency_note}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("View full pipeline details"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Domain**");     st.code(case_dict.get("domain", "—").upper())
            st.markdown("**Language**");   st.code(case_dict.get("language", "—").upper())
        with col2:
            st.markdown("**Precedents used**"); st.code(str(len(analysis.get("similar_cases", []))))
            st.markdown("**Urgency**");          st.code(case_dict.get("urgency", "—").upper())

        st.markdown("**Applicable Statutes**")
        chips = "".join(
            f'<span class="statute-chip">{s.get("act_name","?")} §{s.get("section_no","?")}</span>'
            for s in enriched.get("applicable_statutes", [])
        )
        st.markdown(chips or "<em>None identified</em>", unsafe_allow_html=True)

        st.markdown("**Reasoning Chain**")
        st.markdown(
            f'<div style="font-size:0.88rem;line-height:1.7;color:#3A4A6A;'
            f'background:#F6F7F9;padding:14px;border-radius:6px;">'
            f'{analysis.get("reasoning_chain", "—")}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Supporting Factors**")
        for f in analysis.get("supporting_factors", []):
            st.markdown(f"- {f}")

        st.markdown("**Risk Factors**")
        for f in analysis.get("risk_factors", []):
            st.markdown(f"- {f}")

    st.markdown(
        '<div style="margin-top:2rem;padding:12px 16px;background:#ECEEF2;'
        'border-radius:6px;font-size:0.78rem;color:#6A7A90;line-height:1.6;">'
        '<strong>Disclaimer:</strong> BanglaLex is a research prototype for academic '
        'evaluation only. It does not constitute legal advice. Consult a qualified '
        'Bangladesh advocate for your specific situation.</div>',
        unsafe_allow_html=True,
    )
