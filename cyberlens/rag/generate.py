"""Prompt construction and LLM backends. The model may only use the numbered sources it is given."""
import json
import re
import urllib.request
from .config import LLM_MODEL, OLLAMA_URL

SYSTEM = """You help victims of cyber fraud in India. Use ONLY the numbered sources below.
Rules:
- Every point must end with the source number(s) it comes from, like [S2].
- If the sources do not answer something, say so. Never guess deadlines, amounts or section numbers.
- Never promise a refund or that money will be recovered.
- Use simple words.
Return JSON only, with keys:
  "summary": one or two sentences on what happened, each ending with its source number(s),
  "your_rights": list of strings,
  "what_to_do": list of strings (in order),
  "deadlines": list of strings,
  "where_to_complain": list of strings,
  "complaint_narrative": a short first-person paragraph for the complaint. Use only facts from the victim's message.
    Write [date], [amount], [transaction ID] or [complaint number] for anything the victim did not say.
    Do not say that money was refunded, recovered or compensated."""

LANGUAGE = {"hi": "Reply in Hindi using Devanagari script.",
            "hinglish": "Reply in Hinglish (Hindi in Latin script).",
            "en": "Reply in English."}
# Common Hindi words in Latin script; two or more of these marks a message as Hinglish.
HINGLISH = set("""hai hain tha thi mera mere meri mujhe maine hum aap aapka kya kyu kaise kitna kitne kitni nahi nahin
bhi aur ya se ko ka ki ke me mein par pe kar karna karein kiya gaya gaye gayi hua hui ho raha rahi diya liya
paise paisa rupaye din baad pehle abhi turant bata batana chahiye milega milenge wapas kat kata nikal bina""".split())


def detect_language(text):
    letters = [c for c in text if c.isalpha()]
    if letters and sum("ऀ" <= c <= "ॿ" for c in letters) / len(letters) > 0.3:
        return "hi"
    words = re.findall(r"[a-z]+", text.lower())
    return "hinglish" if sum(w in HINGLISH for w in words) >= 2 else "en"


def format_sources(hits):
    lines = []
    for n, h in enumerate(hits, 1):
        where = f"para {h['section']}" if h.get("section") else f"page {h['page']}"
        lines.append(f"[S{n}] {h['issuer']}, {h['title']} ({h['date']}), {where}:\n{h['text']}")
    return "\n\n".join(lines)


def build_prompt(victim_text, subtype, incident_date, hits, steps, language=None):
    fixed = "\n".join(f"- {s['text']}" for s in steps)
    return (f"{SYSTEM}\n\nSOURCES:\n{format_sources(hits)}\n\n"
            f"ALREADY SHOWN TO THE VICTIM (do not repeat, add to it):\n{fixed}\n\n"
            f"Predicted fraud type: {subtype or 'unknown'}\nIncident date: {incident_date}\n"
            f"Victim's message (private details masked):\n{victim_text}\n\n"
            f"{LANGUAGE[language or detect_language(victim_text)]}\nJSON:")


class OllamaLLM:
    def __init__(self, model=LLM_MODEL, url=OLLAMA_URL, timeout=180):
        self.name, self.url, self.timeout = model, url, timeout

    def __call__(self, prompt):
        body = json.dumps({"model": self.name, "prompt": prompt, "stream": False, "format": "json",
                           "options": {"temperature": .2, "num_ctx": 8192}}).encode()
        req = urllib.request.Request(f"{self.url}/api/generate", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["response"]


TAG = re.compile(r"\[([^\[\]]*)\]")
CITE = re.compile(r"\[S(\d+)\]")
PLACEHOLDERS = {"date", "time", "amount", "transaction id", "complaint number", "bank", "phone number", "upi id"}
SENTENCE = re.compile(r"(?<=[.!?।])\s+(?!\[)")
# A narrative sentence that claims money came back is made up: the victim is still asking for help.
MONEY_BACK = re.compile(r"compensat|refund|reimburs|credited back|reversed|recovered|froze|frozen|wapas mil|"
                        "मुआवज|वापस", re.I)
RECEIVED = re.compile(r"\b(received|got|was|were|been)\b|mil gay|aa gay|"
                      "मिल गय|आ गय", re.I)


def clean_tags(text, n_sources, keep_placeholders=False):
    """Rewrite tags like "[S1, S76L]" or "[S2], [S4]" as valid [S#] tags and delete everything else."""
    def fix(m):
        inner = m.group(1).strip()
        if keep_placeholders and inner.lower() in PLACEHOLDERS:
            return m.group(0)
        valid = [c for c in (p.strip() for p in inner.split(","))
                 if re.fullmatch(r"S\d+", c) and 1 <= int(c[1:]) <= n_sources]
        return "".join(f"[{c}]" for c in valid)
    text = re.sub(r"\]\s*,\s*\[", "][", TAG.sub(fix, str(text)))
    return re.sub(r"\s+([.,!?।])", r"\1", re.sub(r"\s{2,}", " ", text)).strip()


def parse_answer(raw, n_sources):
    """Parse the model's JSON. Drop any point or summary sentence without a valid citation,
    and any narrative sentence that claims money was returned."""
    try:
        data = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except ValueError:
        return None, ["LLM output was not valid JSON"]
    warnings = []
    for key in ("your_rights", "what_to_do", "deadlines", "where_to_complain"):
        kept = []
        for item in data.get(key) or []:
            item = clean_tags(item, n_sources)
            if CITE.search(item):
                kept.append(item)
            else:
                warnings.append(f"dropped uncited or invalid point in {key}")
        data[key] = kept
    summary = []
    for sentence in SENTENCE.split(str(data.get("summary") or "")):
        sentence = clean_tags(sentence, n_sources)
        if CITE.search(sentence):
            summary.append(sentence)
        elif sentence:
            warnings.append("dropped uncited sentence in summary")
    data["summary"] = " ".join(summary)
    narrative = []  # The complaint goes to the police, who don't see the [S#] sources, so tags are removed.
    for sentence in SENTENCE.split(str(data.get("complaint_narrative") or "")):
        sentence = CITE.sub("", clean_tags(sentence, n_sources, keep_placeholders=True))
        sentence = clean_tags(sentence, n_sources, keep_placeholders=True)  # tidy the gaps left behind
        if MONEY_BACK.search(sentence) and RECEIVED.search(sentence):
            warnings.append("dropped narrative sentence claiming money was returned")
        elif sentence:
            narrative.append(sentence)
    data["complaint_narrative"] = " ".join(narrative)
    return data, warnings
