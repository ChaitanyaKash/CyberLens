"""CyberLens local web interface. Runs on this machine only: no uploads, logging or analytics.

Start with:  streamlit run app.py   (or .\\run_ui.ps1)
"""
import time
import pandas as pd
import streamlit as st
from cyberlens.data import ROOT
from cyberlens.predict import Predictor
from cyberlens.ui_helpers import (CALL_THRESHOLD, CONFIDENT_THRESHOLD, EXAMPLES, HELPLINE, THRESHOLD_SOURCE,
                                  friendly, parent_category, sparse_explanation, tips, verdict)

st.set_page_config(page_title="CyberLens", page_icon=":material/shield:", layout="centered",
                   initial_sidebar_state="collapsed")

st.html("""<style>
html, body, [class*="st-"], .stMarkdown, button, textarea {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif !important; }
header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], [data-testid="stDecoration"] { display: none !important; }
.block-container { max-width: 760px !important; padding-top: 3rem !important; padding-bottom: 4rem !important; }
h1 { font-size: 2.6rem !important; font-weight: 700 !important; letter-spacing: -0.02em; margin-bottom: 0 !important; }
h2 { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.01em; }
textarea { font-size: 1.15rem !important; line-height: 1.5 !important; }
[data-testid="stWidgetLabel"] p { font-size: 1.1rem !important; font-weight: 600; color: #1D1D1F; }
[data-testid="stFormSubmitButton"] button, .stButton button[kind="primary"] {
  min-height: 58px; font-size: 1.2rem !important; font-weight: 600; border-radius: 14px !important; }
.stButton button[kind="secondary"] { min-height: 48px; border-radius: 999px !important; padding: 0 1.1rem; }
[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > .result-card) {
  background: #FFFFFF; border: none !important; border-radius: 20px !important;
  box-shadow: 0 2px 14px rgba(0, 0, 0, 0.06); padding: 1.4rem !important; }
.subtitle { font-size: 1.3rem; color: #6E6E73; margin: 0.2rem 0 1.6rem 0; }
.dot { display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 50%; margin-right: 0.5rem; }
.small { color: #6E6E73; font-size: 0.95rem; }
</style>""")

MODELS = {"Champion (encoder + sparse fusion)": "fusion"} if (ROOT/"models/complaint_fusion.joblib").exists() else {}
MODELS["Fast mode (sparse model, CPU only)"] = "adjusted"


@st.cache_resource(show_spinner=False, max_entries=2)
def get_predictor(backend):
    return Predictor(backend=backend)


def encoder_device():
    import torch
    return f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"


def sparse_model(predictor, task):
    for (name, _), bundle in predictor.models.items():
        if name == task:
            model = bundle["model"] if isinstance(bundle, dict) else bundle
            return getattr(model, "sparse", None) or model
    return None


def classify(backend, text):
    predictor = get_predictor(backend)
    complaint = predictor.predict(text, task="complaint", threshold=CONFIDENT_THRESHOLD)
    call = predictor.predict(text, task="call_scam", threshold=CALL_THRESHOLD)
    return {"masked": complaint["masked_text"], "complaint": complaint["predictions"]["complaint"],
            "call_scam": call["predictions"]["call_scam"]}


def load_example(example):
    st.session_state.text = example["text"]
    st.session_state.pop("result", None)


st.session_state.setdefault("text", "")
st.session_state.setdefault("model", list(MODELS)[0])

st.title("CyberLens")
st.html('<p class="subtitle">Check if a message or call is a scam</p>')

with st.form("check", border=False):
    st.text_area("Paste the message or describe what happened", key="text", height=170,
                 placeholder="For example: “Someone called saying my bank card will be blocked and asked for the OTP…”")
    submitted = st.form_submit_button("Check message", type="primary", width="stretch")

st.caption("Or try an example:")
with st.container(horizontal=True):
    for i, example in enumerate(EXAMPLES):
        st.button(example["title"], key=f"example_{i}", on_click=load_example, args=(example,))

if submitted:
    if not st.session_state.text.strip():
        st.warning("Please paste a message or describe what happened first.")
    else:
        backend = MODELS[st.session_state.model]
        first = backend not in st.session_state.setdefault("loaded", set())
        with st.spinner("Getting ready, this takes a few seconds the first time…" if first else "Checking…"):
            if first:
                classify(backend, st.session_state.text)
                st.session_state.loaded.add(backend)
            start = time.perf_counter()
            result = classify(backend, st.session_state.text)
            result["ms"] = (time.perf_counter()-start)*1000
            result["backend"] = backend
        st.session_state.result = result

result = st.session_state.get("result")
if result:
    final = verdict(result["complaint"], result["call_scam"], CONFIDENT_THRESHOLD)
    with st.container(border=True):
        st.html('<span class="result-card"></span>')
        if final["benign"]:
            st.markdown("#### Result")
            st.header("No clear sign of a common scam")
            st.html('<p><span class="dot" style="background:#34C759"></span>'
                    'This does not look like a typical scam call or message.</p>')
            st.markdown("Still be careful: if anyone asks for an OTP, PIN, password or money, stop and check "
                        "with someone you trust.")
        else:
            st.markdown("#### This looks like" if final["confident"] else "#### Possibly")
            st.header(final["headline"])
            if final["confident"]:
                st.html('<p><span class="dot" style="background:#34C759"></span><b>Very likely</b></p>')
            else:
                st.html('<p><span class="dot" style="background:#AEAEB2"></span>Not sure. Please ask someone '
                        'you trust, or call <b>1930</b>.</p>')
            st.markdown("**What to do**")
            for tip in tips(final["label"]):
                st.markdown(f"- {tip}")
        st.markdown(HELPLINE)

st.write("")
with st.expander("Details for experts"):
    st.selectbox("Model", list(MODELS), key="model", help="Fast mode skips the transformer encoder.")
    if result:
        device = encoder_device() if result["backend"] == "fusion" else "CPU"
        st.caption(f"Inference time {result['ms']:.0f} ms (both models) · encoder on {device}")
        st.markdown("**Masked text that was classified**")
        st.code(result["masked"], language=None, wrap_lines=True)
        predictor = get_predictor(result["backend"])
        for task, title in (("complaint", "Complaint subtype"), ("call_scam", "Call/SMS scam type")):
            prediction = result[task]
            st.markdown(f"**{title}** · model `{prediction['model']}`"
                        + (" · *used for the verdict*" if final["task"] == task else ""))
            for item in prediction["top"]:
                label = item["label"]
                text = f"{friendly(label)} ({parent_category(label)})" if task == "complaint" else label
                st.progress(min(max(item["score"], 0.), 1.), text=f"{text} — {item['score']:.1%}")
            model = sparse_model(predictor, task)
            terms = sparse_explanation(model, result["masked"], prediction["top"][0]["label"], 6) \
                if model is not None and hasattr(model, "named_steps") else []
            if terms:
                st.caption("Why: n-grams that pushed the sparse model toward the top label (coefficient × tf-idf). "
                           "The encoder's contribution is not shown.")
                st.dataframe(pd.DataFrame(terms), hide_index=True, width="stretch",
                             column_config={"weight": st.column_config.ProgressColumn(
                                 "Contribution", format="%.3f", min_value=0., max_value=max(t["weight"] for t in terms))})
    st.markdown(
        f"""<p class="small">Complaint model: 39 subtypes, official test macro-F1 0.2745 and accuracy 0.56
(fast mode 0.264 / 0.55); most reliable on UPI, card and net-banking fraud. It has no "not a scam" class.
"Very likely" needs a complaint score of at least {CONFIDENT_THRESHOLD:.3f} ({THRESHOLD_SOURCE}; selective-prediction
rule on validation, measured on the sparse model) or a call-scam score of at least {CALL_THRESHOLD}. Scores are
uncalibrated. Text is masked by rules and never leaves this computer.</p>""", unsafe_allow_html=True)
