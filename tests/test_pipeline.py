import unittest
import numpy as np
from cyberlens.preprocessing import normalize, mask_sensitive, fingerprint, corrupt
from cyberlens.data import ROOT, TASKS, load


class PreprocessingTests(unittest.TestCase):
    def test_language_not_spellcorrected(self):
        text = "is account mein pese nhi aaye; bank is closed"
        self.assertEqual(normalize(text), text)

    def test_specific_masks_and_clues(self):
        masked, counts = mask_sensitive("OTP 123456 bhejo; call +91 9876543210; send to abc@oksbi; a@b.com; https://x.test/a")
        for entity in ("OTP", "PHONE", "UPI_ID", "EMAIL", "URL"):
            self.assertIn(f"<{entity}>", masked)
        self.assertIn("OTP <OTP>", masked)
        self.assertNotIn("123456", masked)
        self.assertEqual(sum(counts.values()), 5)

    def test_names_not_claimed_masked(self):
        self.assertIn("Rahul", normalize("Rahul asked for OTP"))
        self.assertEqual(mask_sensitive("Rahul asked", [(0,5,"PERSON")])[0], "<PERSON> asked")

    def test_negation_and_money_preserved(self):
        self.assertEqual(normalize("OTP nahi diya, 5000 rupees nahi bheje"), "OTP nahi diya, 5000 rupees nahi bheje")

    def test_mask_idempotence(self):
        text = normalize("Call 9876543210 OTP 123456")
        self.assertEqual(normalize(text), text)

    def test_duplicate_templates(self):
        self.assertEqual(fingerprint("I lost 5000 rupees!"), fingerprint("I lost 9000 rupees."))

    def test_corruption_does_not_touch_entities(self):
        self.assertEqual(corrupt("<PHONE> 5000 <UPI_ID>", rate=1), "<PHONE> 5000 <UPI_ID>")


class DataTests(unittest.TestCase):
    @unittest.skipUnless(all((ROOT/"data/processed"/f"{task}.csv").exists() for task in TASKS),
                         "Private processed datasets are not available (expected in public CI)")
    def test_no_duplicate_leakage_and_label_coverage(self):
        for task in TASKS:
            df = load(task)
            self.assertEqual(df.groupby("group").split.nunique().max(), 1)
            self.assertFalse(df[df.split == "train"].group.duplicated().any())
            self.assertEqual(set(df[df.split == "train"].label), set(df[df.split == "validation"].label))
            self.assertFalse(df.text.isna().any())

    @unittest.skipUnless(all((ROOT/"models"/f"{task}_baseline.joblib").exists() for task in TASKS),
                         "Trained local models are not available (expected in public CI)")
    def test_inference(self):
        from cyberlens.predict import Predictor
        result = Predictor().predict("mujhy police ne bola paisa bhejo warna arrest karenge OTP 123456")
        self.assertNotIn("123456", result["masked_text"])
        self.assertEqual(set(result["predictions"]), set(TASKS))
        for output in result["predictions"].values():
            self.assertEqual(len(output["top"]), 3)
            self.assertTrue(0 <= output["top"][0]["score"] <= 1)
        with self.assertRaises(ValueError):
            Predictor().predict(" ")

    def test_subword_hash_and_gradient(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional PyTorch is not installed in core-only CI")
        from cyberlens.embeddings import ngrams, SkipGram
        self.assertEqual(ngrams("mujhe"), ngrams("mujhe"))
        self.assertTrue(set(ngrams("mujhe")) & set(ngrams("mujhey")))
        model = SkipGram(5, 8)
        loss = model(torch.tensor([1,2]), torch.tensor([2,3]), torch.tensor([[3,4],[1,4]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.output.weight.grad.abs().sum(), 0)


if __name__ == "__main__":
    unittest.main()
