"""Helpers for the local Streamlit interface: confidence thresholds, sparse explanations, examples."""
import json
import numpy as np
from .data import ROOT

# Complaint threshold from the selective-prediction experiment (validation-selected on the sparse
# `adjusted` scores, about 5% validation coverage). Used when reports/complaint_selective.json is absent.
DEFAULT_COMPLAINT_THRESHOLD = 0.975
# Call-scam scores are much more peaked (four labels including `none`); this matches Predictor's default.
CALL_THRESHOLD = 0.55

EXAMPLES = [
    {"title": "UPI collect request", "task": "complaint", "expected": "UPI Related Frauds",
     "text": "A buyer on OLX sent me a payment request on PhonePe saying I would receive the amount for my sofa. "
             "I entered my UPI PIN and Rs 18,500 was debited from my account instead."},
    {"title": "UPI refund (Hinglish)", "task": "complaint", "expected": "UPI Related Frauds",
     "text": "Kisi ne Google Pay pe collect request bheja aur bola refund aa raha hai. Maine PIN daala "
             "aur mere account se 12000 rupaye kat gaye."},
    {"title": "KYC update SMS", "task": "complaint", "expected": "Internet Banking / phishing",
     "text": "Got an SMS that my SBI account will be blocked today unless I update KYC. I opened the link, "
             "entered my net banking login and the OTP, and 40,000 was transferred out."},
    {"title": "Card OTP call (Hinglish)", "task": "complaint", "expected": "Debit/Credit Card Fraud",
     "text": "Bank ke naam se call aaya ki credit card ke reward points expire ho rahe hain, OTP bata do. "
             "OTP batate hi card se 25,000 ki shopping ho gayi."},
    {"title": "Fake work-from-home job", "task": "complaint", "expected": "Online Job Fraud",
     "text": "I applied for a work from home data entry job on Telegram. They took Rs 3,000 as registration "
             "fee and then asked for more money for tasks. Now they have blocked my number."},
    {"title": "Sextortion video call", "task": "complaint", "expected": "Sextortion / blackmail",
     "text": "A woman video called me on WhatsApp and recorded the call. Now they are threatening to send an "
             "edited obscene video to my family unless I pay Rs 50,000."},
    {"title": "Instagram account hacked", "task": "complaint", "expected": "Profile Hacking Identity Theft",
     "text": "My Instagram account was hacked last night. The password and recovery email were changed and "
             "the hacker is messaging my friends asking for money."},
    {"title": "Trading app investment", "task": "complaint", "expected": "Investment fraud",
     "text": "I joined a WhatsApp group for share market tips. They made me install a trading app and invest "
             "4 lakh. The app shows profit but does not allow any withdrawal."},
    {"title": "Lottery prize SMS", "task": "complaint", "expected": "Lottery / other fraud",
     "text": "I received a message that I won 25 lakh in a KBC lottery. To claim the prize they asked me to "
             "pay GST and processing charges of Rs 15,000, which I paid."},
    {"title": "Loan app harassment (Hinglish)", "task": "complaint", "expected": "Loan app harassment",
     "text": "Instant loan app se 5000 ka loan liya tha, ab woh 20000 maang rahe hain aur meri photo morph "
             "karke mere contacts ko bhej rahe hain."},
    {"title": "Digital arrest call (Hinglish)", "task": "call_scam", "expected": "police_digital_arrest",
     "text": "Main CBI officer bol raha hoon, aapke naam ke parcel mein drugs mile hain. Video call pe bane "
             "rahiye, aap digital arrest mein hain, warna police ghar aayegi."},
    {"title": "Police blackmail call (Hinglish)", "task": "call_scam", "expected": "police_blackmail",
     "text": "Hello, main police station se bol raha hoon. Aapke bete ko pakda hai, case band karwana hai to "
             "abhi 50 hazaar transfer karo."},
    {"title": "Harmless message", "task": "both", "expected": "Not a scam (no benign complaint class)",
     "text": "Hi Rahul, are we still meeting for lunch tomorrow at 1 pm? Let me know if the time works."},
]


def complaint_threshold(root=ROOT):
    """Return (threshold, source) for the complaint confidence badge."""
    path = root/"reports/complaint_selective.json"
    if path.exists():
        try:
            return float(json.loads(path.read_text())["selected"]["threshold"]), "reports/complaint_selective.json"
        except (KeyError, TypeError, ValueError):
            pass
    return DEFAULT_COMPLAINT_THRESHOLD, "default from the selective-prediction experiment"


def parent_category(label):
    return str(label).split(" :: ")[0]


def subtype(label):
    """Readable short name; generic subtypes ("Unspecified", "Other") fall back to the parent category."""
    parts = str(label).split(" :: ", 1)
    return parts[0] if parts[-1] in ("Unspecified", "Other") else parts[-1]


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
