"""Agent 3: victim text + predicted subtype + date -> cited guidance and a complaint draft."""
import datetime as dt
from ..preprocessing import mask_sensitive
from .config import CORPUS, INDEX
from .embed import get_embedder
from .generate import build_prompt, detect_language, parse_answer
from .index import Index
from .playbook import immediate_steps
from .retrieve import DocFilter, parse_date
from .tracing import record, record_documents, span

LAW = {  # Sections suggested for the complaint, keyed by subtype keyword. Shown only if the Act is in the corpus.
    "it_act_2000": [("Identity Theft", "Section 66C"), ("Impersonat", "Section 66D"), ("CallVishing", "Section 66D"),
                    ("UPI", "Section 66D"), ("Card", "Section 66C"), ("Banking", "Section 66C"), ("Hacking", "Section 66"),
                    ("Sext", "Section 66E / 67"), ("Obscene", "Section 67"), ("Explicit", "Section 67A")],
    "bns_2023": [("Fraud", "Section 318 (cheating)"), ("Impersonat", "Section 319 (cheating by personation)"),
                 ("Blackmail", "Section 308 (extortion)"), ("Stalking", "Section 78 (stalking)")],
}


HINTS = {  # Victims describe events; rules are written as obligations. A short hint bridges the two.
    "Online Financial Fraud": "unauthorised electronic transaction customer liability report bank refund",
    "Online and Social Media Related Crime": "report cyber crime platform complaint",
    "Crime Against Women & Children": "blackmail extortion report remove content",
    "Cyber Attack/ Dependent Crimes": "cyber security incident report",
}


def expand_query(text, subtype):
    if not subtype:
        return text
    parent, _, sub = subtype.replace(" (subtype needs review)", "").partition(" :: ")
    return f"{text} ({sub or parent}; {HINTS.get(parent, '')})"


FALLBACK_SUMMARY = {  # Used when no summary sentence from the LLM had a valid citation.
    "en": "This looks like {fraud} on {date}. The steps and sources below follow the {rules}.",
    "hinglish": "Yeh {fraud} ka case lagta hai ({date}). Neeche diye steps aur sources {rules} ke hisaab se hain.",
    "hi": "यह {fraud} का मामला लगता है ({date})। "
          "नीचे दिए गए कदम और स्रोत {rules} के अनुसार हैं।",
}


class GuidanceAgent:
    def __init__(self, index_dir=INDEX, corpus=CORPUS, embedder=None, llm=None, k=6):
        self.index = Index.load(index_dir)
        self.embedder = embedder or get_embedder(self.index.meta["embedder"])
        self.filter = DocFilter(corpus)
        self.llm, self.k = llm, k

    def retrieve(self, query, subtype=None, incident_date=None, k=None, mode="hybrid"):
        ids = self.filter.allowed(subtype, incident_date)
        with span("retrieve", "RETRIEVER", input__value=query, metadata={"subtype": subtype, "date": str(incident_date),
                                                                        "mode": mode, "allowed_docs": sorted(ids)}) as s:
            hits = self.index.search(query, self.embedder, ids, k=k or self.k, mode=mode)
            record_documents(s, hits)
        return hits

    def sources(self, hits, steps):
        cited = []
        for h in hits:
            cited.append({k: h.get(k) for k in ("chunk_id", "id", "title", "issuer", "date", "page", "section", "url")})
        known = {c["id"] for c in cited}
        for s in steps:
            for doc_id in s["sources"]:
                d = self.filter.docs.get(doc_id)
                if d and doc_id not in known:
                    known.add(doc_id)
                    cited.append({"chunk_id": None, "id": doc_id, "title": d["title"], "issuer": d["issuer"],
                                  "date": d["date"], "page": None, "section": None, "url": d["url"]})
        return cited

    def complaint_draft(self, masked, subtype, date, narrative=None):
        sections = []
        for doc_id, rules in LAW.items():
            if doc_id in set(self.index.doc_of.tolist()):
                act = "IT Act, 2000" if doc_id == "it_act_2000" else "Bharatiya Nyaya Sanhita, 2023"
                sections += [f"{sec}, {act}" for key, sec in rules if key.lower() in (subtype or "").lower()]
        sections = list(dict.fromkeys(sections))
        return "\n".join([
            "To: The Officer, National Cyber Crime Reporting Portal / Cyber Police Station",
            f"Subject: Complaint regarding {(subtype or 'cyber fraud').split(' :: ')[-1]} on {date}",
            "",
            narrative or f"I wish to report the following incident: {masked}",
            "",
            "Details: date and time: ____ | amount lost: ____ | transaction/UTR IDs: ____ | "
            "fraudster's phone/UPI ID/website: ____ | my bank and complaint number: ____",
            "Evidence attached: screenshots, SMS/call logs, bank statement.",
            ("Possibly applicable provisions (for the police to verify): " + "; ".join(sections)) if sections else "",
            "I request that an FIR be registered, the fraudster's accounts be frozen, and the amount be recovered.",
            "",
            "Name: ____  Phone: ____  Address: ____",
        ]).strip()

    def answer(self, text, subtype=None, incident_date=None, query=None):
        date = parse_date(incident_date or dt.date.today())
        masked, _ = mask_sensitive(text)
        with span("guidance", "CHAIN", input__value=masked, metadata={"subtype": subtype, "date": date.isoformat()}) as root:
            result = self._answer(masked, subtype, date, query)
            record(root, output__value=result.get("guidance") or result["immediate_steps"])
        return result

    def _answer(self, masked, subtype, date, query):
        regime = self.filter.regime(date)
        steps = immediate_steps(subtype, regime)
        hits = self.retrieve(query or expand_query(masked, subtype), subtype, date)
        language = detect_language(masked)
        result = {"masked_text": masked, "subtype": subtype, "incident_date": date.isoformat(), "language": language,
                  "liability_rules": "RBI July 2017 circular" if regime == "2017" else "RBI 2026 amendment (from 1 Jan 2027)",
                  "immediate_steps": steps, "guidance": None, "llm": None, "warnings": [],
                  "passages": [{"n": i, "chunk_id": h["chunk_id"], "text": h["text"]} for i, h in enumerate(hits, 1)]}
        narrative = None
        if self.llm is not None:
            prompt = build_prompt(masked, subtype, date, hits, steps, language)
            try:
                with span("llm", "LLM", llm__model_name=self.llm.name, input__value=prompt) as s:
                    raw = self.llm(prompt)
                    record(s, output__value=raw)
                guidance, warnings = parse_answer(raw, len(hits))
                if guidance is not None and not guidance["summary"]:
                    guidance["summary"] = FALLBACK_SUMMARY[language].format(
                        fraud=(subtype or "cyber fraud").split(" :: ")[-1], date=date.isoformat(),
                        rules=result["liability_rules"])
                result.update(guidance=guidance, llm=self.llm.name)
                result["warnings"] += warnings
                narrative = (guidance or {}).get("complaint_narrative")
            except OSError as e:
                result["warnings"].append(f"LLM unavailable ({e}); showing retrieved rules only")
        result["sources"] = self.sources(hits, steps)
        result["complaint_draft"] = self.complaint_draft(masked, subtype, date, narrative)
        result["disclaimer"] = "Guidance from official documents, not legal advice. Check deadlines with your bank or police."
        return result
