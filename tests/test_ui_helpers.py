import json
import tempfile
import unittest
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from cyberlens.ui_helpers import (EXAMPLES, FALLBACK_THRESHOLD, SCAM_GATE_PATH, confident_threshold, friendly,
                                  is_narrative, load_scam_gate, parent_category, scam_probability, sparse_explanation,
                                  subtype, tips, verdict)


def prediction(label, score):
    return {"top": [{"label": label, "score": score}]}


class UiHelperTests(unittest.TestCase):
    def test_threshold_reads_report_and_falls_back(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(confident_threshold(root), (FALLBACK_THRESHOLD, "fallback"))
            (root/"reports").mkdir()
            (root/"reports/complaint_selective.json").write_text(json.dumps({"selected": {"threshold": .9}}))
            self.assertEqual(confident_threshold(root)[0], .9)

    def test_label_parts_and_friendly_names(self):
        label = "Online Financial Fraud :: UPI Related Frauds"
        self.assertEqual(parent_category(label), "Online Financial Fraud")
        self.assertEqual(subtype(label), "UPI Related Frauds")
        self.assertEqual(friendly(label), "UPI payment fraud")
        self.assertEqual(subtype("Sexually Explicit Act :: Unspecified"), "Sexually Explicit Act")
        self.assertEqual(friendly("Cyber Terrorism :: Cyber Terrorism"), "Cyber Terrorism")
        self.assertTrue(1 <= len(tips(label)) <= 2)
        self.assertIn("police", tips("police_digital_arrest")[0].lower())

    def test_verdict_combines_models(self):
        upi = prediction("Online Financial Fraud :: UPI Related Frauds", .93)
        call_none = prediction("none", .9)
        self.assertEqual(verdict(upi, call_none, .975)["task"], "complaint")
        self.assertFalse(verdict(upi, call_none, .975)["confident"])
        self.assertTrue(verdict(prediction(upi["top"][0]["label"], .99), call_none, .975)["confident"])
        police = verdict(upi, prediction("police_blackmail", .9), .975)
        self.assertEqual((police["task"], police["confident"]), ("call_scam", True))
        self.assertEqual(verdict(upi, prediction("police_digital_arrest", .84), .975)["task"], "complaint")
        self.assertEqual(verdict(upi, prediction("other_scam", .95), .975)["task"], "complaint")
        self.assertTrue(verdict(prediction("Any Other Cyber Crime :: Other", .2), call_none, .975)["benign"])
        self.assertFalse(verdict(prediction("Any Other Cyber Crime :: Other", .3), call_none, .975)["benign"])
        # Only a confident `none` reassures; a confident other_scam must not read as benign.
        self.assertFalse(verdict(prediction("Any Other Cyber Crime :: Other", .2), prediction("other_scam", .9), .975)["benign"])

    def test_police_override_needs_police_wording(self):
        card = prediction("Online Financial Fraud :: DebitCredit Card FraudSim Swap Fraud", .9)
        arrest = prediction("police_digital_arrest", .87)
        bank = "Bank ke naam se call aaya ki credit card ke reward points expire ho rahe hain, OTP bata do."
        self.assertEqual(verdict(card, arrest, .975, text=bank)["task"], "complaint")
        cbi = "Main CBI officer bol raha hoon, aap digital arrest mein hain."
        self.assertEqual(verdict(card, arrest, .975, text=cbi)["task"], "call_scam")
        self.assertEqual(verdict(card, arrest, .975)["task"], "call_scam")

    def test_gate_decides_scam_or_not(self):
        other = prediction("Any Other Cyber Crime :: Other", .2)
        call_none = prediction("none", .9)
        chat = "Kal office nahi aa paunga, tabiyat theek nahi hai."
        sure = verdict(other, call_none, .975, text=chat, scam_p=.02, raw_text=chat)
        self.assertEqual((sure["tier"], sure["headline"], sure["benign"]), ("not", "Not a scam", True))
        self.assertEqual(sure["closest"]["label"], other["top"][0]["label"])
        self.assertEqual(verdict(other, call_none, .975, scam_p=.2, raw_text=chat)["tier"], "probably_not")
        self.assertEqual(verdict(other, call_none, .975, scam_p=.45, raw_text=chat)["tier"], "unclear")
        # The gate replaces the call model's `none` reassurance: gate says scam, so no "no clear sign".
        scam = verdict(other, call_none, .975, scam_p=.9, raw_text=chat)
        self.assertEqual((scam["task"], scam["benign"], scam["tier"]), ("complaint", False, None))
        # A very sure scam call turns "not a scam" into "unclear"; a merely likely one does not.
        self.assertEqual(verdict(other, prediction("other_scam", .96), .975, scam_p=.02, raw_text=chat)["tier"], "unclear")
        self.assertEqual(verdict(other, prediction("other_scam", .9), .975, scam_p=.02, raw_text=chat)["tier"], "not")
        # Fragments are never called harmless.
        self.assertEqual(verdict(other, call_none, .975, scam_p=.02, raw_text="Credit card no")["tier"], "unclear")

    def test_gate_never_reassures_a_victim_narrative(self):
        upi = prediction("Online Financial Fraud :: UPI Related Frauds", .6)
        story = "I entered my UPI PIN on a payment request and Rs 18,500 was debited from my account."
        self.assertTrue(is_narrative(story))
        result = verdict(upi, prediction("none", .9), .975, scam_p=.05, raw_text=story)
        self.assertFalse(result["benign"])
        self.assertEqual((result["task"], result["tier"]), ("complaint", None))
        for text in ("Mujhe bank ke naam se call aaya, OTP bata diya", "My Instagram account was hacked last night",
                     "TOTAL INTERNET BANKING FRAUD OF RS"):
            self.assertTrue(is_narrative(text), text)
        for text in ("Mummy, main 8 baje tak ghar aa jaunga.", "Your OTP for Flipkart login is 482913.",
                     "HDFC Bank: Rs 500 debited from a/c XX1234. Avl bal Rs 12,300."):
            self.assertFalse(is_narrative(text), text)

    def test_police_type_only_after_gate_says_scam(self):
        card = prediction("Online Financial Fraud :: DebitCredit Card FraudSim Swap Fraud", .9)
        arrest = prediction("police_digital_arrest", .87)
        cbi = "Main CBI officer bol raha hoon, aap digital arrest mein hain, video call pe rahiye."
        self.assertEqual(verdict(card, arrest, .975, text=cbi, scam_p=.9, raw_text=cbi)["task"], "call_scam")
        otp = "Credit card ke reward points expire ho rahe hain, OTP bata do abhi."
        self.assertEqual(verdict(card, arrest, .975, text=otp, scam_p=.9, raw_text=otp)["task"], "complaint")

    @unittest.skipUnless(SCAM_GATE_PATH.exists(), "scam gate model not trained on this machine")
    def test_saved_gate_on_typical_messages(self):
        gate = load_scam_gate()
        other, call_none = prediction("Any Other Cyber Crime :: Other", .2), prediction("none", .9)
        def tier(text):
            return verdict(other, call_none, .975, scam_p=scam_probability(gate, text), raw_text=text)
        self.assertIn(tier("Kal office nahi aa paunga, tabiyat theek nahi hai.")["tier"], ("not", "probably_not"))
        self.assertIn(tier("HDFC Bank: Rs 500 debited from a/c XX1234 on 26-Oct. Avl bal Rs 12,300. "
                           "Not you? Call 18002586161.")["tier"], ("not", "probably_not"))
        kyc = tier("Dear customer your SBI KYC has expired, update now at sbi-kyc-update.in or account will be "
                   "blocked today")
        self.assertEqual((kyc["tier"], kyc["benign"]), (None, False))
        story = tier("A buyer on OLX sent me a payment request on PhonePe. I entered my UPI PIN and Rs 18,500 was "
                     "debited instead.")
        self.assertFalse(story["benign"])

    def test_explanation_ranks_class_specific_ngrams(self):
        texts = ["upi pin debited money", "upi request pin", "instagram hacked password", "account hacked login"] * 3
        labels = ["upi", "upi", "hack", "hack"] * 3
        features = FeatureUnion([("word", TfidfVectorizer()), ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4)))])
        model = Pipeline([("features", features), ("classifier", LogisticRegression(C=10))]).fit(texts, labels)
        terms = sparse_explanation(model, "my upi pin was used", "upi", top_k=5)
        self.assertTrue(terms and all(t["weight"] > 0 for t in terms))
        self.assertEqual([t["weight"] for t in terms], sorted((t["weight"] for t in terms), reverse=True))
        self.assertIn("upi", {t["feature"].strip() for t in terms})
        self.assertEqual(sparse_explanation(model, "zzzz qqqq", "upi"), [])

    def test_examples_are_well_formed(self):
        self.assertLessEqual(len(EXAMPLES), 6)
        for example in EXAMPLES:
            self.assertTrue(example["text"].strip() and example["title"].strip())


if __name__ == "__main__":
    unittest.main()
