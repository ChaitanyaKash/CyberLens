"""Helpers for the local Streamlit interface: confidence tiers, plain-language labels, explanations."""
import json
import re
import numpy as np
from .data import ROOT

# Complaint "Very likely" threshold: derived on validation by the selective-prediction rule (largest
# coverage with answered macro-F1 >= 0.5), read from reports/complaint_selective.json; 0.975 otherwise.
# Measured on the sparse `adjusted` model, so it is an approximation for the fusion model.
FALLBACK_THRESHOLD = 0.975
# Call-scam scores are sharper (four labels including `none`); fixed heuristic, not validated.
CALL_THRESHOLD = 0.55
# The call model was trained on Hinglish police-scam calls and is overconfident elsewhere (bank OTP and
# hacked-account texts scored 0.84 and 0.58 for digital arrest), so it decides the verdict only for
# police labels at or above this score. Heuristic, set from a handful of examples.
CALL_OVERRIDE = 0.85
# Everyday words ("ke", "naam", "bank", "aaya", "card") also push that model to digital arrest: a bank
# reward-points OTP call scored 0.87 without any police wording. The override therefore also needs an
# authority or police-case word in the text. These words occur in 96% of the police training calls
# (the rest are Aadhaar/SIM misuse lines) and in 23% of the others.
POLICE_CUES = re.compile(r"\b(?:police|cbi|ncb|ed|acp|dcp|cyber ?cell|crime branch|cyber crime|investigation|"
                         r"arrest(?:ed)?|warrant|fir|case|criminal|officer|inspector|thana|station|court|customs|"
                         r"narcotics|enforcement|suspect|khilaaf|jail|giraftaar|drugs?|illegal|misuse|gande|asleel|"
                         r"morphed|pornographic|objectionable)\b", re.I)
# Below this complaint score, a confident `none` from the call model is shown as "no clear sign".
# Heuristic and deliberately low: wrongly reassuring a victim is worse than a vague answer.
BENIGN_CEILING = 0.25

# Scam/not-scam gate: TF-IDF v2 (cyberlens/scam_binary_v2.py), trained on SMS-style messages.
# Band edges on its scam probability; tuned on the v2 validation set (see reports/scam_gate_bands.json).
SCAM_GATE_PATH = ROOT / "models/scam_binary_v2_tfidf.joblib"
NOT_SCAM_BELOW = 0.10
PROBABLY_NOT_BELOW = 0.30
UNCLEAR_BELOW = 0.60
GATE_DISCLAIMER = ("This is an automated estimate from a model trained mostly on SMS-style messages. It can be wrong. "
                   "Never share OTPs, PINs or passwords, and if in doubt, verify through official channels or call "
                   "1930 / cybercrime.gov.in.")
# The gate only knows messages, not victims describing what happened ("I lost 5000 via UPI…"), and it scores
# most of those as harmless. A first-person or third-person account of an incident is therefore never told
# "not a scam"; it keeps the complaint-model result.
NARRATIVE_CUES = re.compile(
    r"\b(?:i|we)\s+(?:have\s+|had\s+|was\s+|were\s+|am\s+|just\s+)?(?:received|got|lost|paid|sent|transferred|"
    r"clicked|shared|gave|given|installed|downloaded|applied|invested|ordered|bought|registered|entered|"
    r"asked|told|cheated|scammed|duped|tricked|blackmailed|threatened|harassed|hacked|blocked|called|contacted)\b"
    r"|\b(?:my|our)\b(?:\s+\S+){0,4}\s+(?:was|were|has been|have been|got|is|are)\s+(?:hacked|blocked|debited|"
    r"deducted|stolen|misused|compromised|frozen|withdrawn|transferred|changed|taken|used|leaked)\b"
    r"|\b(?:someone|somebody|a person|a man|a woman|a lady|a guy|the caller|caller|fraudster|scammer|they|he|she|"
    r"unknown number|unknown person)\s+(?:has\s+|had\s+|then\s+)?(?:called|messaged|texted|asked|told|said|sent|"
    r"took|cheated|threatened|blackmailed|hacked|deducted|debited|withdrew|transferred|claimed|pretended)\b"
    r"|\b(?:asked|told|forced|convinced|tricked|cheated)\s+me\b|\bfrom my\s+(?:\S+\s+){0,2}(?:account|card|wallet)\b"
    r"|\b(?:mujhe|mujhko|humein|hume|hamein)\b.{0,50}?\b(?:call|phone|message|msg|sms|link|otp)\b"
    r"|\b(?:maine|mene|humne|hamne)\b"
    r"|\b(?:mere|mera|meri|hamare|hamara)\s+(?:\S+\s+){0,3}(?:account|khata|khate|card|paise|paisa|number|"
    r"id|whatsapp|instagram|facebook|upi)\b"
    r"|\bcall\s+(?:aaya|aya|aayi|ayi)\b|\b(?:ho gaye|ho gayi|kat gaye|kat gaya|nikal gaye|chale gaye)\b"
    # Naming the crime itself: short complaints ("TOTAL INTERNET BANKING FRAUD OF RS") have no pronoun.
    r"|\b(?:fraud\w*|scam(?:med|mer|mers)?|cheat\w*|hack(?:ed|er|ers|ing)?|stolen|victim|complain\w*|blackmail\w*|"
    r"harass\w*|abus\w*|extort\w*|fake|impersonat\w*|caller informed|dhokha\w*|thagi|fir|stole|theft|thief|"
    r"took away|money debited|amount debited)\b"
    # Introductions and help requests typical of complaints, not of the scam messages themselves.
    r"|\bmy name is\b|\bi am\s+\w+(?:\s+\w+)?\s+from\b|\b(?:contacted|approached|called|messaged|threatened)\s+me\b"
    r"|\b(?:please\s+)?help me\b", re.I)
# A message the gate calls harmless still reads "Unclear" when the call model is this sure of a scam type.
CALL_STRONG = 0.95
# Fragments ("OTP l", "Credit card no") are too short to call harmless; they read "Unclear" at best.
MIN_WORDS_NOT_SCAM = 4

HELPLINE = "If you lost money or shared details, call **1930** at once or report at **cybercrime.gov.in**."

FRIENDLY = {
    "UPI Related Frauds": "UPI payment fraud",
    "Internet Banking Related Fraud": "Net banking fraud",
    "DebitCredit Card FraudSim Swap Fraud": "Debit or credit card fraud",
    "EWallet Related Fraud": "Mobile wallet fraud",
    "Fraud CallVishing": "Fraud phone call",
    "DematDepository Fraud": "Investment or share trading fraud",
    "Business Email CompromiseEmail Takeover": "Fake business email",
    "Cryptocurrency Fraud": "Cryptocurrency fraud",
    "Online Job Fraud": "Fake job offer",
    "Online Matrimonial Fraud": "Marriage website fraud",
    "Cheating by Impersonation": "Someone pretending to be another person",
    "FakeImpersonating Profile": "Fake social media profile",
    "Profile Hacking Identity Theft": "Hacked account or identity theft",
    "Cyber Bullying  Stalking  Sexting": "Online harassment or blackmail",
    "EMail Phishing": "Fake email to steal details",
    "Impersonating Email": "Fake email from a known name",
    "Intimidating Email": "Threatening email",
    "Email Hacking": "Hacked email account",
    "Unauthorised AccessData Breach": "Unauthorised access to an account",
    "Online Gambling  Betting": "Online betting or gambling",
    "Ransomware": "Ransomware (files locked for money)",
    "police_digital_arrest": "Fake police or “digital arrest” call",
    "police_blackmail": "Fake police blackmail call",
    "other_scam": "Scam call",
    "none": "No clear sign of a scam call",
}

TIPS = {
    "money": "Never share your OTP, UPI PIN, card number or password. Your bank will never ask for them.",
    "collect": "To receive money you never need to enter your UPI PIN.",
    "job": "Real employers do not ask you to pay a fee to get a job.",
    "invest": "Do not trust apps or groups that promise sure profits. Check SEBI registration.",
    "account": "Change your password now, turn on two-step login, and tell your contacts not to send money.",
    "harass": "Do not pay. Save the messages as proof and block the person.",
    "police": "Real police never arrest anyone on a video call or ask for money. Hang up.",
    "general": "Do not click unknown links or install apps a stranger asks for.",
}

EXAMPLES = [
    {"title": "Bank OTP call", "text": "Bank ke naam se call aaya ki credit card ke reward points expire ho "
     "rahe hain, OTP bata do. OTP batate hi card se 25,000 ki shopping ho gayi.", "expected": "Card fraud"},
    {"title": "UPI refund", "text": "A buyer on OLX sent me a payment request on PhonePe saying I would "
     "receive money for my sofa. I entered my UPI PIN and Rs 18,500 was debited instead.", "expected": "UPI fraud"},
    {"title": "Job offer", "text": "I applied for a work from home data entry job on Telegram. They took "
     "Rs 3,000 as registration fee and then asked for more money. Now they have blocked me.", "expected": "Job fraud"},
    {"title": "Lottery SMS", "text": "I received a message that I won 25 lakh in a KBC lottery. To claim "
     "the prize they asked me to pay GST and processing charges of Rs 15,000.", "expected": "Lottery fraud"},
    {"title": "Account hacked", "text": "My Instagram account was hacked last night. The password and "
     "recovery email were changed and the hacker is asking my friends for money.", "expected": "Hacked account"},
    {"title": "Police call", "text": "Main CBI officer bol raha hoon, aapke naam ke parcel mein drugs mile "
     "hain. Video call pe bane rahiye, aap digital arrest mein hain.", "expected": "police_digital_arrest"},
]


def confident_threshold(root=ROOT):
    """Return (threshold, source) for the complaint "Very likely" verdict."""
    try:
        report = json.loads((root/"reports/complaint_selective.json").read_text())
        return float(report["selected"]["threshold"]), "reports/complaint_selective.json"
    except (OSError, KeyError, TypeError, ValueError):
        return FALLBACK_THRESHOLD, "fallback"


CONFIDENT_THRESHOLD, THRESHOLD_SOURCE = confident_threshold()


def parent_category(label):
    return str(label).split(" :: ")[0]


def subtype(label):
    """Readable short name; generic subtypes ("Unspecified", "Other") fall back to the parent category."""
    parts = str(label).split(" :: ", 1)
    return parts[0] if parts[-1] in ("Unspecified", "Other") else parts[-1]


def friendly(label):
    short = subtype(label)
    return FRIENDLY.get(short, " ".join(short.split()))


def tips(label):
    text = str(label)
    keys = (("police", ("police",)), ("collect", ("UPI",)), ("money", ("Fraud", "Card", "Banking", "Wallet", "UPI")),
            ("job", ("Job",)), ("invest", ("Demat", "Crypto")), ("account", ("Hacking", "Hacked", "Access", "Profile")),
            ("harass", ("Bullying", "Sexual", "Intimidating", "Blackmail", "Obscene")))
    out = [TIPS[key] for key, words in keys if any(w in text for w in words)]
    return (out or [TIPS["general"]])[:2]


def load_scam_gate(path=SCAM_GATE_PATH):
    """The TF-IDF scam/not-scam pipeline, or None when the model file is missing."""
    import joblib
    try:
        return joblib.load(path)["model"]
    except (OSError, KeyError, EOFError):
        return None


def scam_probability(gate, text):
    return float(gate.predict_proba([text])[0, 1])


def is_narrative(text):
    """True when the text reads like someone describing an incident rather than the message itself."""
    return bool(NARRATIVE_CUES.search(str(text)))


def gate_band(p):
    if p < NOT_SCAM_BELOW:
        return "not"
    if p < PROBABLY_NOT_BELOW:
        return "probably_not"
    return "unclear" if p < UNCLEAR_BELOW else "scam"


def verdict(complaint, call, threshold=FALLBACK_THRESHOLD, call_threshold=CALL_THRESHOLD, text=None,
            scam_p=None, raw_text=None):
    """Combine the models into one plain-language verdict.

    Without `scam_p` this is the two-model verdict below. With `scam_p` (the gate's scam probability for
    `raw_text`) a message can also read "Not a scam" / "Probably not a scam" / "Unclear"; see gated_verdict.
    """
    base = _two_model_verdict(complaint, call, threshold, call_threshold, text)
    if scam_p is None:
        return {**base, "tier": None, "scam_p": None}
    return gated_verdict(base, complaint, call, scam_p, raw_text if raw_text is not None else text,
                         threshold, call_threshold)


def gated_verdict(base, complaint, call, scam_p, raw_text, threshold=FALLBACK_THRESHOLD, call_threshold=CALL_THRESHOLD):
    """The gate alone decides scam vs not-scam; the other models only pick the scam type.

    - Victim narratives are never told "not a scam": they get the scam-type result, without the old
      "no clear sign" reassurance (the gate replaces that logic).
    - Gate >= UNCLEAR_BELOW: scam. The type is the police call label when the police override holds
      (confident police call + police wording), otherwise the complaint subtype.
    - UNCLEAR_BELOW > gate >= PROBABLY_NOT_BELOW: "Unclear".
    - Gate below PROBABLY_NOT_BELOW: "Not a scam" / "Probably not a scam", unless the call model is at
      least CALL_STRONG sure of a scam label or the text has fewer than MIN_WORDS_NOT_SCAM words ("Unclear").
    """
    complaint_top = complaint["top"][0] if complaint else None
    closest = {"label": complaint_top["label"], "score": complaint_top["score"]} if complaint_top else None
    if base["task"] == "call_scam" and not base["benign"]:
        scam_type = base
    else:
        scam_type = {"task": "complaint", "label": complaint_top["label"], "score": complaint_top["score"],
                     "confident": complaint_top["score"] >= threshold, "headline": friendly(complaint_top["label"]),
                     "benign": False}
    if is_narrative(raw_text):
        return {**scam_type, "tier": None, "scam_p": scam_p, "narrative": True}
    band = gate_band(scam_p)
    if band == "scam":
        return {**scam_type, "tier": None, "scam_p": scam_p}
    unclear = {"task": "scam_gate", "label": "unclear", "score": scam_p, "confident": False, "benign": False,
               "headline": "Unclear — please review carefully", "tier": "unclear", "scam_p": scam_p, "closest": closest}
    call_top = call["top"][0] if call else None
    too_short = len(str(raw_text).split()) < MIN_WORDS_NOT_SCAM
    if band == "unclear" or too_short or (call_top and call_top["label"] != "none" and call_top["score"] >= CALL_STRONG):
        return unclear
    return {"task": "scam_gate", "label": "not_scam", "score": 1 - scam_p, "confident": band == "not",
            "benign": True, "headline": "Not a scam" if band == "not" else "Probably not a scam",
            "tier": band, "scam_p": scam_p, "closest": closest}


def _two_model_verdict(complaint, call, threshold=FALLBACK_THRESHOLD, call_threshold=CALL_THRESHOLD, text=None):
    """Combine both models into one plain-language verdict.

    `complaint` and `call` are Predictor prediction dicts. The call model wins when it confidently
    finds a police scam call and `text` (the classified text, when given) contains a police cue; it
    also marks the text as benign when it says `none` and the complaint model is not confident.
    Returns {"task", "label", "score", "confident", "headline", "benign"}.
    """
    complaint_top = complaint["top"][0] if complaint else None
    if call is not None:
        top = call["top"][0]
        if top["score"] >= call_threshold:
            police_cue = text is None or bool(POLICE_CUES.search(text))
            if top["label"].startswith("police") and top["score"] >= CALL_OVERRIDE and police_cue:
                return {"task": "call_scam", "label": top["label"], "score": top["score"], "confident": True,
                        "headline": friendly(top["label"]), "benign": False}
            if top["label"] == "none" and (complaint_top is None or complaint_top["score"] < BENIGN_CEILING):
                return {"task": "call_scam", "label": "none", "score": top["score"], "confident": False,
                        "headline": "No clear sign of a common scam", "benign": True}
    return {"task": "complaint", "label": complaint_top["label"], "score": complaint_top["score"],
            "confident": complaint_top["score"] >= threshold, "headline": friendly(complaint_top["label"]),
            "benign": False}


def classify_both(predictor, text):
    """Run the complaint and call-scam models on one text."""
    complaint = predictor.predict(text, task="complaint", threshold=CONFIDENT_THRESHOLD)
    call = predictor.predict(text, task="call_scam", threshold=CALL_THRESHOLD)
    return {"masked": complaint["masked_text"], "complaint": complaint["predictions"]["complaint"],
            "call_scam": call["predictions"]["call_scam"]}


def sparse_model(predictor, task):
    """The loaded linear sparse model behind a task (the sparse member for fusion), or None."""
    for (name, _), bundle in predictor.models.items():
        if name == task:
            model = bundle["model"] if isinstance(bundle, dict) else bundle
            return getattr(model, "sparse", None) or model
    return None


def sparse_explanation(model, cleaned, label, top_k=8):
    """Word/char n-grams with the largest coefficient x tf-idf toward `label` for a linear sparse pipeline.

    `model` needs `named_steps` with `features` and a linear `classifier`, and `classes_`.
    Returns a list of {"feature", "kind", "weight"} sorted by weight, positive contributions only.
    """
    steps = model.named_steps
    features, classifier = steps["features"], steps["classifier"]
    row = features.transform([cleaned]).tocsr()
    if row.nnz == 0:
        return []
    index = list(model.classes_).index(label)
    coef = classifier.coef_
    # Binary linear models keep a single row of weights for the second class.
    weights = coef[index] if len(coef) > 1 else (coef[0] if index == 1 else -coef[0])
    contributions = row.data*weights[row.indices]
    names = features.get_feature_names_out()
    out = []
    for i in np.argsort(contributions)[::-1][:top_k]:
        if contributions[i] <= 0:
            break
        name = str(names[row.indices[i]])
        kind, _, feature = name.partition("__") if "__" in name else ("word", "", name)
        out.append({"feature": feature, "kind": "char n-gram" if kind == "char" else "word n-gram",
                    "weight": float(contributions[i])})
    return out
