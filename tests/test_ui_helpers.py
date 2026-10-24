import json
import tempfile
import unittest
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from cyberlens.ui_helpers import (EXAMPLES, FALLBACK_THRESHOLD, confident_threshold, friendly, parent_category,
                                  sparse_explanation, subtype, tips, verdict)


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
