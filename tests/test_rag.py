import json
import numpy as np
import pytest
from cyberlens.rag.chunk import chunk_pages, sections
from cyberlens.rag.embed import HashEmbedder
from cyberlens.rag.generate import parse_answer
from cyberlens.rag.index import BM25, Index
from cyberlens.rag.load import clean_pages, is_garbled, load_pages
from cyberlens.rag.mobile import export_mobile
from cyberlens.rag.playbook import immediate_steps
from cyberlens.rag.retrieve import DocFilter

UPI = "Online Financial Fraud :: UPI Related Frauds"
DOCS = {
    "liability_2017": ("RBI", "6. Zero liability if the customer notifies the bank within three working days.\n"
                              "7. Limited liability of Rs 10,000 if reported within four to seven working days.", "", "2026-12-31"),
    "liability_2027": ("RBI", "76M. Zero liability if reported within five calendar days of the fraudulent transaction.", "2027-01-01", ""),
    "it_act": ("Parliament", "66D. Punishment for cheating by personation by using computer resource.\n"
                             "66C. Punishment for identity theft.", "", ""),
    "kyc_md": ("RBI", "Banks must keep customer identification records for five years.", "", ""),
}


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "text").mkdir()
    manifest = []
    for doc_id, (issuer, text, start, end) in DOCS.items():
        (tmp_path / "text" / f"{doc_id}.txt").write_text(text)
        manifest.append({"id": doc_id, "title": doc_id.replace("_", " "), "issuer": issuer, "date": "2020",
                         "effective_from": start, "effective_to": end, "role": "background" if doc_id == "kyc_md" else "rights",
                         "priority": "core", "subtypes": "ALL" if doc_id == "it_act" else [UPI], "url": "",
                         "file": f"text/{doc_id}.txt", "type": "txt"})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "subtype_to_docs.json").write_text(json.dumps(
        {"_applies_to_every_subtype": ["it_act"], UPI: ["liability_2017", "liability_2027", "kyc_md"]}))
    return tmp_path


@pytest.fixture
def index(corpus):
    return Index.build(chunk_pages(load_pages(corpus), max_words=40, overlap=5), HashEmbedder())


def test_sections_split_on_paragraph_numbers(corpus):
    labels = [label for _, label, _ in sections(load_pages(corpus)[:1])]
    assert labels == ["6", "7"]


def test_garbled_hindi_headers_and_repeated_footers_are_removed():
    assert is_garbled("���� RESERVE BANK")
    pages = [f"Footer line\nbody {i} text" for i in range(5)]
    assert all("Footer" not in p for p in clean_pages(pages))


def test_long_section_is_cut_with_overlap(corpus):
    pages = [{**load_pages(corpus)[0], "text": "1. " + " ".join(f"w{i}" for i in range(100))}]
    chunks = chunk_pages(pages, max_words=40, overlap=10)
    assert len(chunks) == 3 and chunks[1]["text"].split()[0] == "w29"


def test_bm25_finds_exact_section_numbers():
    bm = BM25(["section 66d cheating by personation", "section 66c identity theft", "zero liability"])
    assert int(np.argmax(bm.scores("punishment under 66D"))) == 0


def test_filter_switches_liability_rules_on_1_jan_2027(corpus):
    flt = DocFilter(corpus)
    before, after = flt.allowed(UPI, "2026-12-31"), flt.allowed(UPI, "2027-01-01")
    assert "liability_2017" in before and "liability_2027" not in before
    assert "liability_2027" in after and "liability_2017" not in after
    assert "it_act" in before and "kyc_md" not in before  # always-on law, background excluded


def test_parent_label_widens_to_all_its_subtypes(corpus):
    assert DocFilter(corpus).subtypes("Online Financial Fraud (subtype needs review)") == [UPI]


def test_hybrid_search_respects_filter(index, corpus):
    ids = DocFilter(corpus).allowed(UPI, "2026-11-01")
    hits = index.search("how many days to report to bank", HashEmbedder(), ids, k=5)
    assert hits and all(h["id"] in ids for h in hits)
    assert index.search("66D personation", HashEmbedder(), k=1, mode="bm25")[0]["id"] == "it_act"


def test_index_round_trip(index, tmp_path):
    index.save(tmp_path / "idx")
    loaded = Index.load(tmp_path / "idx")
    assert loaded.meta["embedder"] == "hash-char-3-5" and loaded.vectors.shape == index.vectors.shape


def test_search_refuses_mismatched_embedder(index):
    class Other(HashEmbedder):
        name = "other"
    with pytest.raises(ValueError):
        index.search("x", Other(), mode="vector")


def test_uncited_llm_points_are_dropped():
    raw = json.dumps({"what_to_do": ["Call 1930 [S1]", "Your money will surely come back", "See [S9]"], "your_rights": []})
    data, warnings = parse_answer("noise " + raw, n_sources=3)
    assert data["what_to_do"] == ["Call 1930 [S1]"] and len(warnings) == 2
    assert parse_answer("not json", 3)[0] is None


def test_playbook_uses_the_right_regime():
    old = " ".join(s["text"] for s in immediate_steps(UPI, "2017"))
    new = " ".join(s["text"] for s in immediate_steps(UPI, "2027"))
    assert "3 working days" in old and "5 calendar days" in new and "1930" in old


def test_agent_masks_text_and_falls_back_without_llm(index, corpus, tmp_path):
    from cyberlens.rag.agent import GuidanceAgent
    index.save(tmp_path / "idx")
    agent = GuidanceAgent(tmp_path / "idx", corpus)
    out = agent.answer("Paise kat gaye, fraudster number 9876543210", UPI, "2026-11-01")
    assert "9876543210" not in out["masked_text"] + out["complaint_draft"]
    assert out["guidance"] is None and out["immediate_steps"] and out["passages"]

    def fake_llm(prompt):
        assert "[S1]" in prompt and "9876543210" not in prompt
        return json.dumps({"summary": "x", "what_to_do": ["Report to bank [S1]"], "complaint_narrative": "I lost money."})
    fake_llm.name = "fake"
    agent.llm = fake_llm
    out = agent.answer("Paise kat gaye, fraudster number 9876543210", UPI, "2026-11-01")
    assert out["guidance"]["what_to_do"] == ["Report to bank [S1]"] and "I lost money." in out["complaint_draft"]


def test_mobile_export_is_loadable_without_python_objects(index, corpus, tmp_path):
    out = export_mobile(index, tmp_path / "m", corpus)
    cfg = json.loads((out / "config.json").read_text())
    vectors = np.fromfile(out / "embeddings_f16.bin", dtype="<f2").reshape(cfg["chunks"], cfg["dim"])
    assert np.allclose(vectors, index.vectors, atol=1e-2) and f"{UPI}|2027" in cfg["playbook"]


def test_tracing_records_retrieval_and_llm_spans(index, corpus, tmp_path, monkeypatch):
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from cyberlens.rag import tracing
    from cyberlens.rag.agent import GuidanceAgent
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("test"))
    index.save(tmp_path / "idx")
    llm = lambda prompt: json.dumps({"what_to_do": ["Report [S1]"]})
    llm.name = "fake"
    GuidanceAgent(tmp_path / "idx", corpus, llm=llm).answer("call from 9876543210", UPI, "2026-11-01")
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {"guidance", "retrieve", "llm"}
    assert spans["retrieve"].attributes["openinference.span.kind"] == "RETRIEVER"
    assert "retrieval.documents.0.document.id" in spans["retrieve"].attributes
    assert all("9876543210" not in str(v) for s in spans.values() for v in s.attributes.values())


def test_summary_and_narrative_tags_are_checked():
    raw = json.dumps({"summary": "Report in 3 days [S1, S76L]. Banks are usually kind. Refund comes in 10 days [Already Shown] [S2].",
                      "what_to_do": ["Tell the bank [S1], [S2]", "Wait [S7]"],
                      "complaint_narrative": "On [date] Rs [amount] left my account [S1, S2]. I received a compensation of 85%. "
                                             "The bank froze the money before it was moved. I request a refund."})
    data, warnings = parse_answer(raw, n_sources=3)
    assert data["summary"] == "Report in 3 days [S1]. Refund comes in 10 days [S2]."
    assert data["what_to_do"] == ["Tell the bank [S1][S2]"]
    assert data["complaint_narrative"] == "On [date] Rs [amount] left my account. I request a refund."
    assert "dropped uncited sentence in summary" in warnings
    assert warnings.count("dropped narrative sentence claiming money was returned") == 2
    assert parse_answer(json.dumps({"summary": "No source here."}), 3)[0]["summary"] == ""


def test_reply_language_is_detected_and_pinned_in_prompt():
    from cyberlens.rag.generate import build_prompt, detect_language
    assert detect_language("मेरे खाते से बिना अनुमति पैसे कट गए") == "hi"
    assert detect_language("UPI se mere account se 20000 nikal gaye bina OTP ke") == "hinglish"
    assert detect_language("Someone took money from my account through UPI") == "en"
    prompt = build_prompt("मेरे खाते से पैसे कट गए", UPI, "2026-11-01", [], [])
    assert "Reply in Hindi using Devanagari script." in prompt
    assert "Reply in Hinglish (Hindi in Latin script)." in build_prompt("paise kat gaye bina OTP ke", UPI, "2026-11-01", [], [])
    assert "[date], [amount]" in prompt


def test_agent_reports_language_and_falls_back_to_template_summary(index, corpus, tmp_path):
    from cyberlens.rag.agent import GuidanceAgent
    index.save(tmp_path / "idx")
    llm = lambda prompt: json.dumps({"summary": "Uncited claim [Already Shown].", "what_to_do": ["Report [S1]"]})
    llm.name = "fake"
    out = GuidanceAgent(tmp_path / "idx", corpus, llm=llm).answer("मेरे खाते से पैसे कट गए", UPI, "2026-11-01")
    assert out["language"] == "hi" and "UPI Related Frauds" in out["guidance"]["summary"]
    assert "है" in out["guidance"]["summary"]  # template in Devanagari


def test_playbook_states_2017_caps_per_account_type():
    text = " ".join(s["text"] for s in immediate_steps(UPI, "2017"))
    assert "Rs 5,000 for basic savings (BSBD)" in text and "Rs 10,000 for other savings" in text and "Rs 25,000" in text
