"""Turn a predicted subtype and incident date into the set of documents worth searching."""
import datetime as dt
import json
from .config import CORPUS

# Call-scam labels from the Hinglish model, mapped onto the complaint taxonomy used by the corpus.
CALL_LABELS = {
    "police_digital_arrest": ["Online Financial Fraud :: Fraud CallVishing", "Online and Social Media Related Crime :: Cheating by Impersonation"],
    "police_blackmail": ["Crime Against Women & Children :: Cyber Blackmailing & Threatening", "Online and Social Media Related Crime :: Cheating by Impersonation"],
    "other_scam": ["Online Financial Fraud :: Fraud CallVishing"],
}


def parse_date(value):
    if not value:
        return dt.date.today()
    return value if isinstance(value, dt.date) else dt.date.fromisoformat(str(value)[:10])


class DocFilter:
    def __init__(self, corpus=CORPUS):
        mapping = json.loads((corpus / "subtype_to_docs.json").read_text(encoding="utf-8"))
        self.always = set(mapping.pop("_applies_to_every_subtype"))
        self.by_subtype = {k: set(v) for k, v in mapping.items()}
        manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
        self.docs = {d["id"]: d for d in manifest}

    def subtypes(self, label):
        """Accept a full 'Parent :: Sub' label, a bare parent, a call label, or the classifier's
        'Parent (subtype needs review)' fallback, which widens to every subtype under that parent."""
        if not label:
            return []
        label = label.replace(" (subtype needs review)", "").strip()
        if label in CALL_LABELS:
            return CALL_LABELS[label]
        if label in self.by_subtype:
            return [label]
        return [s for s in self.by_subtype if s.split(" :: ")[0] == label.split(" :: ")[0]]

    def in_force(self, doc_id, date):
        d = self.docs.get(doc_id, {})
        start, end = d.get("effective_from") or "", d.get("effective_to") or ""
        return (not start or date >= parse_date(start)) and (not end or date <= parse_date(end))

    def allowed(self, label=None, date=None, include_background=False):
        date = parse_date(date)
        names = self.subtypes(label)
        ids = set(self.always).union(*(self.by_subtype[s] for s in names)) if names else set(self.docs)
        return {i for i in ids if i in self.docs and self.in_force(i, date)
                and self.docs[i].get("priority") != "exclude"
                and (include_background or self.docs[i].get("role") != "background")}

    def regime(self, date):
        """Which customer-liability rules apply to a transaction on this date."""
        return "2027" if parse_date(date) >= dt.date(2027, 1, 1) else "2017"
