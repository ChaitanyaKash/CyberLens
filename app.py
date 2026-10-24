"""CyberLens local web interface. Runs on this machine only: no uploads, logging or analytics.

Start with:  streamlit run app.py   (or .\\run_ui.ps1)
"""
import time
import pandas as pd
import streamlit as st
from cyberlens.data import ROOT
from cyberlens.predict import Predictor
from cyberlens.ui_helpers import (CALL_THRESHOLD, EXAMPLES, complaint_threshold, parent_category,
                                  sparse_explanation, subtype)

st.set_page_config(page_title="CyberLens", page_icon=":material/shield:", layout="wide")

TASKS = {"Complaint subtype": ("complaint",), "Call/SMS scam type": ("call_scam",),
         "Both": ("complaint", "call_scam")}
HAS_FUSION = (ROOT/"models/complaint_fusion.joblib").exists()
MODELS = {"Champion (encoder + sparse fusion)": "fusion"} if HAS_FUSION else {}
MODELS["Fast mode (sparse model, CPU only)"] = "adjusted"
THRESHOLD, THRESHOLD_SOURCE = complaint_threshold()


@st.cache_resource(show_spinner=False, max_entries=2)
def get_predictor(backend):
    return Predictor(backend=backend)


def encoder_device():
    import torch
    return f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"


def sparse_model(predictor, task):
    """The linear sparse model behind a task, used for the explanation panel."""
    for (name, backend), bundle in predictor.models.items():
        if name == task:
            model = bundle["model"] if isinstance(bundle, dict) else bundle
            return getattr(model, "sparse", None) or model
    return None


def classify(predictor, text, tasks):
    out = {}
    for task in tasks:
        threshold = THRESHOLD if task == "complaint" else CALL_THRESHOLD
        out[task] = predictor.predict(text, task=task, threshold=threshold)
    return out


def review_reasons(prediction, threshold):
    reasons = []
    top = prediction["top"][0]
    if top["score"] < threshold:
        reasons.append(f"score {top['score']:.2f} is below the {threshold:.3f} confidence threshold")
    evidence = prediction["class_evidence"]
    if evidence.get("f1") is not None and (evidence["f1"] < .5 or (evidence.get("support") or 0) < 20):
        reasons.append(f"this class scores F1 {evidence['f1']:.2f} on {evidence['source']} (weak class)")
    if not prediction["has_known_features"]:
        reasons.append("no known words or character patterns in the text")
    if "needs review" in prediction.get("suggested_type", ""):
        reasons.append("the subtype and the most likely parent category disagree")
    if prediction["experimental"]:
        reasons.append("experimental model")
    return reasons


def render_prediction(task, prediction, predictor, cleaned):
    threshold = THRESHOLD if task == "complaint" else CALL_THRESHOLD
    top = prediction["top"]
    with st.container(border=True):
        heading = "Complaint subtype" if task == "complaint" else "Call/SMS scam type"
        st.caption(f"{heading} · model: {prediction['model']}")
        if task == "complaint":
            st.subheader(subtype(top[0]["label"]))
            st.markdown(f"Parent category: **{parent_category(top[0]['label'])}**")
        else:
            st.subheader(top[0]["label"])
        if prediction["needs_review"]:
            st.badge("Needs human review", icon=":material/visibility:", color="orange")
            for reason in review_reasons(prediction, threshold):
                st.caption(f"• {reason}")
        else:
            st.badge("Confident", icon=":material/check_circle:", color="green")
        st.markdown("**Top 3**")
        for item in top:
            label = subtype(item["label"]) if task == "complaint" else item["label"]
            st.progress(min(max(item["score"], 0.), 1.), text=f"{label} — {item['score']:.1%}")
        if task == "complaint" and "category_top" in prediction:
            with st.expander("Parent category scores"):
                for item in prediction["category_top"]:
                    st.progress(min(item["score"], 1.), text=f"{item['label']} — {item['score']:.1%}")
        model = sparse_model(predictor, task)
        if model is not None and hasattr(model, "named_steps"):
            with st.expander("Why this label (sparse model evidence)"):
                terms = sparse_explanation(model, cleaned, top[0]["label"])
                if terms:
                    st.caption("Word and character n-grams in the text that pushed the sparse model toward "
                               "this label (coefficient × tf-idf). In champion mode the encoder also "
                               "contributes and is not shown here.")
                    st.dataframe(pd.DataFrame(terms), hide_index=True, width="stretch",
                                 column_config={"weight": st.column_config.ProgressColumn(
                                     "Contribution", format="%.3f", min_value=0.,
                                     max_value=max(t["weight"] for t in terms))})
                else:
                    st.caption("No n-gram in the text pushed the sparse model toward this label.")


def load_example(example):
    st.session_state.text = example["text"]
    st.session_state.task = "Call/SMS scam type" if example["task"] == "call_scam" else \
        "Both" if example["task"] == "both" else "Complaint subtype"


st.session_state.setdefault("text", "")
st.session_state.setdefault("task", "Complaint subtype")

with st.sidebar:
    st.title(":material/shield: CyberLens")
    st.caption("Cybercrime complaint and scam-message classifier · English and Hinglish")
    model_name = st.selectbox("Model", list(MODELS), help="Fast mode skips the transformer encoder.")
    backend = MODELS[model_name]
    st.divider()
    st.markdown("**About the scores**")
    st.markdown(
        "- Complaint subtypes (39 classes): official test macro-F1 **0.2745**, accuracy **0.56** "
        "(champion). Fast mode: macro-F1 0.264, accuracy 0.55.\n"
        "- Most reliable on UPI, card and internet-banking fraud; rare subtypes are often confused.\n"
        "- The complaint model has **no 'not a scam' class**: every text gets a crime label.\n"
        f"- *Confident* needs a complaint score of at least **{THRESHOLD:.3f}** "
        f"({THRESHOLD_SOURCE}); call scores need {CALL_THRESHOLD}. Scores are uncalibrated.")
    st.markdown("**Privacy**")
    st.caption("Everything runs on this computer. Text is masked by rules before classification "
               "(names and addresses may remain). Nothing is saved or sent anywhere.")

single, batch = st.tabs([":material/edit_note: Single message", ":material/table: Batch CSV"])

with single:
    st.markdown("**Try an example**")
    with st.container(horizontal=True):
        for i, example in enumerate(EXAMPLES):
            st.button(example["title"], key=f"example_{i}", on_click=load_example, args=(example,), type="tertiary")
    with st.form("classify"):
        st.text_area("Complaint description or scam message", key="text", height=150,
                     placeholder="Describe what happened, or paste the SMS / call transcript…")
        st.segmented_control("Task", list(TASKS), key="task")
        submitted = st.form_submit_button("Classify", type="primary", icon=":material/search:")
    if submitted:
        text = st.session_state.text
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            first_load = backend not in st.session_state.setdefault("loaded", set())
            with st.spinner("Loading models (first run only)…" if first_load else "Classifying…"):
                predictor = get_predictor(backend)
                tasks = TASKS[st.session_state.task or "Complaint subtype"]
                if first_load:
                    classify(predictor, text, tasks)
                    st.session_state.loaded.add(backend)
                start = time.perf_counter()
                results = classify(predictor, text, tasks)
                elapsed = time.perf_counter()-start
            cleaned = next(iter(results.values()))["masked_text"]
            device = encoder_device() if backend == "fusion" and "complaint" in tasks else "CPU"
            st.caption(f"Inference time {elapsed*1000:.0f} ms · encoder on {device}")
            with st.expander("Masked text that was classified"):
                st.code(cleaned, language=None, wrap_lines=True)
            columns = st.columns(len(tasks))
            for column, task in zip(columns, tasks):
                with column:
                    render_prediction(task, results[task]["predictions"][task], predictor, cleaned)

with batch:
    st.markdown("Upload a CSV with a `text` column and, optionally, an `expected_label` column.")
    upload = st.file_uploader("CSV file", type="csv")
    batch_task = st.segmented_control("Task", ["Complaint subtype", "Call/SMS scam type"],
                                      default="Complaint subtype", key="batch_task")
    if upload is not None:
        frame = pd.read_csv(upload)
        if "text" not in frame.columns:
            st.error("The CSV needs a column named `text`.")
        elif st.button("Classify all rows", type="primary", icon=":material/play_arrow:"):
            task = TASKS[batch_task or "Complaint subtype"][0]
            threshold = THRESHOLD if task == "complaint" else CALL_THRESHOLD
            predictor = get_predictor(backend)
            rows, progress = [], st.progress(0., text="Classifying…")
            texts = frame["text"].fillna("").astype(str).tolist()
            for i, text in enumerate(texts):
                if not text.strip():
                    rows.append({"predicted_label": "", "score": None, "status": "empty text"})
                else:
                    prediction = predictor.predict(text, task=task, threshold=threshold)["predictions"][task]
                    top = prediction["top"][0]
                    rows.append({"predicted_label": top["label"], "score": top["score"],
                                 "status": "Needs human review" if prediction["needs_review"] else "Confident"})
                progress.progress((i+1)/len(texts), text=f"Classified {i+1} of {len(texts)}")
            progress.empty()
            out = pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
            if task == "complaint":
                out.insert(out.columns.get_loc("predicted_label")+1, "parent_category",
                           out["predicted_label"].map(lambda x: parent_category(x) if x else ""))
            if "expected_label" in out.columns:
                labelled = out["expected_label"].notna()
                out["correct"] = (out["predicted_label"] == out["expected_label"]).where(labelled)
                with st.container(horizontal=True):
                    st.metric("Accuracy", f"{out.loc[labelled, 'correct'].mean():.1%}", border=True)
                    st.metric("Labelled rows", int(labelled.sum()), border=True)
                    st.metric("Confident rows", int((out["status"] == "Confident").sum()), border=True)
            st.dataframe(out, hide_index=True, width="stretch",
                         column_config={"score": st.column_config.ProgressColumn("Score", format="%.2f",
                                                                                min_value=0., max_value=1.)})
            st.download_button("Download results CSV", out.to_csv(index=False).encode("utf-8"),
                               file_name="cyberlens_predictions.csv", mime="text/csv", icon=":material/download:")
