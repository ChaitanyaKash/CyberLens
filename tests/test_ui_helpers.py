import json
import tempfile
import unittest
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from cyberlens.ui_helpers import (DEFAULT_COMPLAINT_THRESHOLD, EXAMPLES, complaint_threshold, parent_category,
                                  sparse_explanation, subtype)


class UiHelperTests(unittest.TestCase):
    def test_threshold_reads_report_and_falls_back(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(complaint_threshold(root)[0], DEFAULT_COMPLAINT_THRESHOLD)
            (root/"reports").mkdir()
            (root/"reports/complaint_selective.json").write_text(json.dumps({"selected": {"threshold": .9}}))
            self.assertEqual(complaint_threshold(root), (.9, "reports/complaint_selective.json"))

    def test_label_parts(self):
        label = "Online Financial Fraud :: UPI Related Frauds"
        self.assertEqual(parent_category(label), "Online Financial Fraud")
        self.assertEqual(subtype(label), "UPI Related Frauds")
        self.assertEqual(subtype("none"), "none")
        self.assertEqual(subtype("Sexually Explicit Act :: Unspecified"), "Sexually Explicit Act")

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
        self.assertGreaterEqual(len(EXAMPLES), 10)
        for example in EXAMPLES:
            self.assertIn(example["task"], ("complaint", "call_scam", "both"))
            self.assertTrue(example["text"].strip() and example["title"].strip())


if __name__ == "__main__":
    unittest.main()
