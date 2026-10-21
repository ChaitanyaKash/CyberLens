"""Hand-authored smoke challenge. NOT representative or a substitute for held-out data."""
from .predict import Predictor
from .data import ROOT
from .metrics import save_report

EXAMPLES = [
    ("police_digital_arrest", "This is the police. You are under digital arrest. Transfer money now or we will arrest you."),
    ("police_digital_arrest", "mai police se bol rha hu tum digital arrest mein ho turant paise transfer kro"),
    ("police_digital_arrest", "CBI officer bol raha hun. Video call pe raho, digital arrest se bachne ko payment karo."),
    ("police_digital_arrest", "Your son has been arrested by police. Send money immediately to release him."),
    ("police_digital_arrest", "apka beta police custody me hai paise bhejo warna usko jail bhej denge"),
    ("police_digital_arrest", "tumhare bete ko drugs case me pakda hai. Abhi paise do toh chhod denge"),
    ("police_blackmail", "Your phone number has been used to share obscene videos. Pay us to avoid a police case."),
    ("police_blackmail", "aapke number se gande videos share hue hain turant paise bhejo warna case hoga"),
    ("other_scam", "Your bank account will be blocked. Share your OTP to update KYC immediately."),
    ("other_scam", "aapki bank kyc expire ho gayi otp batao warna account band ho jayega"),
    ("other_scam", "You won a lottery. Pay a processing fee to claim your prize."),
    ("other_scam", "amazon prize jeet gaye ho delivery fees pehle pay karo"),
    ("other_scam", "Your Aadhaar is blocked. Give me your OTP to reactivate it."),
    ("none", "Hello beta ghar aa gaya hoon darwaza khol do"),
    ("none", "Your order is out for delivery. I will reach your gate in ten minutes."),
    ("none", "kal class ke baad library mein milte hain"),
    ("none", "Never share your OTP with anyone. This is a bank security awareness message."),
    ("none", "Police issued a warning about digital arrest scams. Do not send money to callers."),
]


def run():
    results = {}
    backends = ["baseline", "word2vec", "subword"]
    if (ROOT/"reports/call_scam_transformer.json").exists():
        backends.append("transformer")
    if (ROOT/"models/champions.json").exists():
        backends.append("best")
    for backend in backends:
        predictor = Predictor(backend=backend)
        records = []
        for expected, text in EXAMPLES:
            p = predictor.predict(text, "call_scam")["predictions"]["call_scam"]
            records.append({"text": text, "expected": expected, "predicted": p["top"][0]["label"],
                            "score": p["top"][0]["score"], "needs_review": p["needs_review"]})
        results[backend] = {"notice": "Small manually authored diagnostic; not a benchmark or training source. Police/relative arrest threats use the broad source label police_digital_arrest; obscene-content threats use police_blackmail. Gold labels were revised after source-semantic inspection; this is development data.",
                            "correct": sum(x["expected"] == x["predicted"] for x in records),
                            "total": len(records), "records": records}
    save_report(ROOT / "reports/manual_challenge.json", results)


if __name__ == "__main__":
    run()
