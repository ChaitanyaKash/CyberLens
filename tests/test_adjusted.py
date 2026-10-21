import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from cyberlens.adjusted import PriorAdjustedClassifier, log_prior
from cyberlens.data import ROOT, load


class AdjustedTests(unittest.TestCase):
    def test_zero_tau_is_identity_and_prior_shifts_toward_frequent_class(self):
        x = np.array([[0.], [1.], [2.], [3.]])
        y = np.array(["rare", "common", "common", "common"])
        pipeline = Pipeline([("classifier", LogisticRegression())]).fit(x, y)
        prior = log_prior(pd.Series(y), pipeline.classes_)
        raw = pipeline.predict_proba(x)
        np.testing.assert_allclose(PriorAdjustedClassifier(pipeline, prior, 0).predict_proba(x), raw, rtol=1e-6)
        shifted = PriorAdjustedClassifier(pipeline, prior, 1).predict_proba(x)
        common = list(pipeline.classes_).index("common")
        self.assertTrue(np.all(shifted[:, common] >= raw[:, common]))
        np.testing.assert_allclose(shifted.sum(axis=1), 1, rtol=1e-6)

    @unittest.skipUnless((ROOT/"data/processed/complaint_official.csv").exists(),
                         "Private processed datasets are not available (expected in public CI)")
    def test_official_protocol_keeps_train_validation_disjoint(self):
        df = load("complaint", protocol="official")
        development = df[df.split != "test"]
        self.assertEqual(development.groupby("group").split.nunique().max(), 1)
        self.assertFalse(development.group.duplicated().any())
        self.assertEqual(set(df[df.split == "train"].label), set(df[df.split == "validation"].label))
        test = df[df.split == "test"]
        self.assertEqual(len(test), 30771)
        self.assertTrue(test.seen_template.any() and not test.seen_template.all())


if __name__ == "__main__":
    unittest.main()
