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
- Reply in the same language and style as the victim (English, Hindi or Hinglish), in simple words.
Return JSON only, with keys:
  "summary": one or two sentences on what happened,
  "your_rights": list of strings,
  "what_to_do": list of strings (in order),
  "deadlines": list of strings,
  "where_to_complain": list of strings,
  "complaint_narrative": a short first-person paragraph for the complaint (facts only, no legal advice)."""


def format_sources(hits):
    lines = []
    for n, h in enumerate(hits, 1):
        where = f"para {h['section']}" if h.get("section") else f"page {h['page']}"
        lines.append(f"[S{n}] {h['issuer']}, {h['title']} ({h['date']}), {where}:\n{h['text']}")
    return "\n\n".join(lines)


def build_prompt(victim_text, subtype, incident_date, hits, steps):
    fixed = "\n".join(f"- {s['text']}" for s in steps)
    return (f"{SYSTEM}\n\nSOURCES:\n{format_sources(hits)}\n\n"
            f"ALREADY SHOWN TO THE VICTIM (do not repeat, add to it):\n{fixed}\n\n"
            f"Predicted fraud type: {subtype or 'unknown'}\nIncident date: {incident_date}\n"
            f"Victim's message (private details masked):\n{victim_text}\n\nJSON:")


class OllamaLLM:
    def __init__(self, model=LLM_MODEL, url=OLLAMA_URL, timeout=180):
        self.name, self.url, self.timeout = model, url, timeout

    def __call__(self, prompt):
        body = json.dumps({"model": self.name, "prompt": prompt, "stream": False, "format": "json",
                           "options": {"temperature": .2, "num_ctx": 8192}}).encode()
        req = urllib.request.Request(f"{self.url}/api/generate", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["response"]


CITE = re.compile(r"\[S(\d+)\]")


def parse_answer(raw, n_sources):
    """Parse the model's JSON and drop any point that cites a source it was not given (or none at all)."""
    try:
        data = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except ValueError:
        return None, ["LLM output was not valid JSON"]
    warnings = []
    for key in ("your_rights", "what_to_do", "deadlines", "where_to_complain"):
        kept = []
        for item in data.get(key) or []:
            cites = [int(c) for c in CITE.findall(str(item))]
            if cites and all(1 <= c <= n_sources for c in cites):
                kept.append(str(item))
            else:
                warnings.append(f"dropped uncited or invalid point in {key}")
        data[key] = kept
    return data, warnings
