"""CyberLens local web interface, product-page layout. Runs on this machine only: no uploads, logging or analytics.

Start with:  streamlit run app_saas.py --server.port 8502   (or .\\run_ui_saas.ps1)
"""
import html
import time
import pandas as pd
import streamlit as st
from cyberlens.data import ROOT
from cyberlens.predict import Predictor
from cyberlens.ui_helpers import (CALL_THRESHOLD, CONFIDENT_THRESHOLD, EXAMPLES, THRESHOLD_SOURCE, classify_both,
                                  friendly, parent_category, sparse_explanation, sparse_model, tips, verdict)

st.set_page_config(page_title="CyberLens", page_icon=":material/shield:", layout="wide",
                   initial_sidebar_state="collapsed")

SHIELD = ('<svg width="30" height="30" viewBox="0 0 24 24" fill="none"><path d="M12 2l8 3v6c0 5-3.4 9.4-8 11'
          '-4.6-1.6-8-6-8-11V5l8-3z" fill="#4F46E5"/><path d="M8.5 12l2.3 2.3 4.7-4.8" stroke="#fff" stroke-width="2"'
          ' stroke-linecap="round" stroke-linejoin="round"/></svg>')

st.html("""<style>
html, body, p, li, label, h1, h2, h3, h4, textarea, input, button {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif !important; }
header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], [data-testid="stDecoration"] { display: none !important; }
.stApp { background: radial-gradient(1200px 420px at 50% -80px, #E0E7FF 0%, rgba(224,231,255,0) 70%), #F8FAFC; }
.block-container { max-width: 1180px !important; padding: 1.2rem 1.5rem 3rem !important; }
.nav { display: flex; align-items: center; justify-content: space-between; padding: 0.4rem 0 2.2rem; }
.brand { display: flex; align-items: center; gap: 0.6rem; font-weight: 700; font-size: 1.35rem; color: #0F172A; }
.pill { display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.4rem 0.9rem; border-radius: 999px;
  font-size: 0.9rem; font-weight: 600; background: #FFFFFF; border: 1px solid #E5E7EB; color: #334155; }
.pill .live { width: 8px; height: 8px; border-radius: 50%; background: #22C55E; }
.hero { text-align: center; margin: 0 auto 2.2rem; max-width: 760px; }
.hero h1 { font-size: clamp(2.2rem, 5vw, 3.3rem) !important; font-weight: 800 !important; letter-spacing: -0.03em;
  line-height: 1.1 !important; color: #0F172A; padding: 0 !important; }
.hero h1 span { background: linear-gradient(90deg, #4F46E5, #2563EB); -webkit-background-clip: text;
  background-clip: text; color: transparent; }
.hero p { font-size: 1.25rem; color: #475569; margin-top: 0.8rem; }
.label { font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #6366F1;
  margin-bottom: 0.6rem; }
[data-testid="stForm"], .card { background: #FFFFFF; border: 1px solid #E5E7EB !important; border-radius: 20px !important;
  box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 8px 24px rgba(15,23,42,0.06); padding: 1.6rem !important; }
[data-testid="stWidgetLabel"] p { font-size: 1.1rem !important; font-weight: 600; color: #0F172A; }
textarea { font-size: 1.15rem !important; line-height: 1.55 !important; }
[data-testid="stFormSubmitButton"] button { min-height: 56px; font-size: 1.15rem !important; font-weight: 600;
  border-radius: 12px !important; background: #4F46E5 !important; border-color: #4F46E5 !important; color: #FFF !important;
  transition: background .15s ease, transform .15s ease; }
[data-testid="stFormSubmitButton"] button:hover { background: #4338CA !important; transform: translateY(-1px); }
.stButton button { min-height: 44px; border-radius: 999px !important; padding: 0 1rem; background: #FFFFFF;
  border: 1px solid #E5E7EB; color: #334155; font-weight: 500; }
.stButton button:hover { border-color: #A5B4FC; color: #4338CA; background: #EEF2FF; }
.result { min-height: 430px; display: flex; flex-direction: column; }
.empty { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
  text-align: center; color: #64748B; gap: 0.8rem; font-size: 1.1rem; }
.empty .ring { width: 84px; height: 84px; border-radius: 50%; background: #EEF2FF; display: flex;
  align-items: center; justify-content: center; }
.kicker { color: #64748B; font-size: 1rem; font-weight: 600; }
.verdict { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; color: #0F172A; line-height: 1.2;
  margin: 0.2rem 0 0.9rem; }
.conf { align-self: flex-start; display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.4rem 0.85rem; border-radius: 999px;
  font-weight: 600; font-size: 0.95rem; }
.conf.sure { background: #DCFCE7; color: #166534; } .conf.maybe { background: #F1F5F9; color: #475569; }
.conf.safe { background: #DCFCE7; color: #166534; }
.conf i { width: 8px; height: 8px; border-radius: 50%; background: currentColor; display: inline-block; }
.note { color: #475569; font-size: 1.05rem; margin: 0.9rem 0 0; }
.tips { list-style: none; padding: 0; margin: 1.2rem 0 0; }
.tips li { display: flex; gap: 0.7rem; align-items: flex-start; font-size: 1.08rem; color: #1E293B; margin-bottom: 0.7rem; }
.tips li::before { content: "\\2713"; flex: none; width: 1.5rem; height: 1.5rem; border-radius: 50%; background: #EEF2FF;
  color: #4F46E5; font-weight: 800; font-size: 0.9rem; display: inline-flex; align-items: center; justify-content: center; margin-top: 1px; }
.help { margin-top: auto; padding: 1rem 1.1rem; border-radius: 14px; background: #EEF2FF; border: 1px solid #C7D2FE;
  color: #312E81; font-size: 1.05rem; }
.help b { font-size: 1.15rem; }
.features { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 1rem; margin: 2.4rem 0 1.4rem; }
.feature { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 16px; padding: 1.1rem 1.2rem; }
.feature b { display: block; color: #0F172A; font-size: 1.05rem; margin-bottom: 0.25rem; }
.feature span { color: #64748B; font-size: 0.95rem; }
[data-testid="stExpander"] details { background: #FFFFFF; border: 1px solid #E5E7EB !important; border-radius: 16px !important; }
.foot { text-align: center; color: #94A3B8; font-size: 0.9rem; margin-top: 2rem; }
.small { color: #64748B; font-size: 0.95rem; }
</style>""")

MODELS = {"Champion (encoder + sparse fusion)": "fusion"} if (ROOT/"models/complaint_fusion.joblib").exists() else {}
MODELS["Fast mode (sparse model, CPU only)"] = "adjusted"


@st.cache_resource(show_spinner=False, max_entries=2)
def get_predictor(backend):
    return Predictor(backend=backend)


def encoder_device():
    import torch
    return f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"


def load_example(example):
    st.session_state.text = example["text"]
    st.session_state.pop("result", None)


def result_card(final):
    if final is None:
        return (f'<div class="card result"><div class="label">Result</div><div class="empty">'
                f'<div class="ring">{SHIELD}</div><div>Your result will appear here</div>'
                f'<div class="small">Paste a message on the left and press <b>Check message</b>.</div></div></div>')
    if final["benign"]:
        body = ('<div class="kicker">Result</div><div class="verdict">No clear sign of a common scam</div>'
                '<span class="conf safe"><i></i>Looks safe</span>'
                '<p class="note">Still be careful: if anyone asks for an OTP, PIN, password or money, stop and check '
                'with someone you trust.</p>')
    else:
        pill = ('<span class="conf sure"><i></i>Very likely</span>' if final["confident"] else
                '<span class="conf maybe"><i></i>Possibly</span>')
        note = "" if final["confident"] else \
            '<p class="note">Not sure. Please ask someone you trust, or call <b>1930</b>.</p>'
        items = "".join(f"<li><span>{html.escape(t)}</span></li>" for t in tips(final["label"]))
        body = (f'<div class="kicker">{"This looks like" if final["confident"] else "This might be"}</div>'
                f'<div class="verdict">{html.escape(final["headline"])}</div>{pill}{note}'
                f'<ul class="tips">{items}</ul>')
    return (f'<div class="card result"><div class="label">Result</div>{body}<div class="help">'
            'Lost money or shared details? Call <b>1930</b> now or report at <b>cybercrime.gov.in</b></div></div>')


st.session_state.setdefault("text", "")
st.session_state.setdefault("model", list(MODELS)[0])

st.html(f'<div class="nav"><div class="brand">{SHIELD}CyberLens</div>'
        '<div class="pill"><span class="live"></span>Private · runs on your device</div></div>'
        '<div class="hero"><h1>Is this message <span>a scam?</span></h1>'
        '<p>Paste a message, or describe a call, and get a clear answer in seconds.</p></div>')

left, right = st.columns([1.15, 1], gap="large")
with left:
    with st.form("check", border=False):
        st.html('<div class="label">Your message</div>')
        st.text_area("Paste the message or describe what happened", key="text", height=210,
                     placeholder="For example: “Someone called saying my bank card will be blocked and asked for the OTP…”")
        submitted = st.form_submit_button("Check message", type="primary", width="stretch",
                                          icon=":material/verified_user:")
    st.caption("Try an example")
    with st.container(horizontal=True):
        for i, example in enumerate(EXAMPLES):
            st.button(example["title"], key=f"example_{i}", on_click=load_example, args=(example,))

if submitted:
    if not st.session_state.text.strip():
        left.warning("Please paste a message or describe what happened first.")
    else:
        backend = MODELS[st.session_state.model]
        first = backend not in st.session_state.setdefault("loaded", set())
        with right, st.spinner("Getting ready, this takes a few seconds the first time…" if first else "Checking…"):
            predictor = get_predictor(backend)
            if first:
                classify_both(predictor, st.session_state.text)
                st.session_state.loaded.add(backend)
            start = time.perf_counter()
            result = classify_both(predictor, st.session_state.text)
            result["ms"], result["backend"] = (time.perf_counter()-start)*1000, backend
        st.session_state.result = result

result = st.session_state.get("result")
final = verdict(result["complaint"], result["call_scam"], CONFIDENT_THRESHOLD, text=result["masked"]) if result else None
with right:
    st.html(result_card(final))

st.html('<div class="features">'
        '<div class="feature"><b>English &amp; Hinglish</b><span>Understands everyday writing, typos and mixed language.</span></div>'
        '<div class="feature"><b>Nothing leaves your device</b><span>No uploads, no accounts, nothing is saved.</span></div>'
        '<div class="feature"><b>Built from 93,000+ real complaint reports</b><span>79,000+ unique after cleaning · '
        '39 kinds of cybercrime</span></div>'
        '</div>')

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
    st.html(f"""<p class="small">Complaint model: 39 subtypes, official test macro-F1 0.2745 and accuracy 0.56
(fast mode 0.264 / 0.55); most reliable on UPI, card and net-banking fraud. It has no "not a scam" class.
"Very likely" needs a complaint score of at least {CONFIDENT_THRESHOLD:.3f} ({THRESHOLD_SOURCE}; selective-prediction
rule on validation, measured on the sparse model) or a strong police-scam call score. Scores are uncalibrated.</p>""")

st.html('<div class="foot">CyberLens is a research prototype and can be wrong. It is not legal or financial advice. '
        'For help, call the national cyber fraud helpline 1930.</div>')
