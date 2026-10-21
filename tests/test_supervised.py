import io
import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None,"Optional PyTorch is unavailable")
class SupervisedTests(unittest.TestCase):
    def test_hash_features_are_stable_and_share_spelling_components(self):
        from cyberlens.supervised import encode_text,CONFIG
        a,b = encode_text("mujhe paisa bhejo"),encode_text("mujhey paisa bhejo")
        for first,second in zip(a,encode_text("mujhe paisa bhejo")):
            np.testing.assert_array_equal(first,second)
        self.assertTrue(set(a[1])&set(b[1]))
        for channel,key in ((0,"word_buckets"),(1,"char_buckets")):
            self.assertTrue(np.all((a[channel]>=0)&(a[channel]<CONFIG[key])))
            self.assertTrue(len(encode_text("")[channel]))

    def test_supervised_embeddings_receive_gradient_and_weights_reload(self):
        from cyberlens.supervised import HashBagNet,encode_text,collate
        torch.manual_seed(42)
        model = HashBagNet(3)
        values,y = collate([(encode_text("police paisa bhejo"),0),(encode_text("bank account hacked"),1)])
        logits = model(*values)
        torch.nn.functional.cross_entropy(logits,y).backward()
        self.assertTrue(torch.isfinite(logits).all())
        self.assertGreater(model.word.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.char.weight.grad.abs().sum().item(),0)
        model.eval()
        expected = model(*values).detach()
        stream = io.BytesIO()
        torch.save(model.state_dict(),stream)
        stream.seek(0)
        restored = HashBagNet(3).eval()
        restored.load_state_dict(torch.load(stream,weights_only=True))
        torch.testing.assert_close(restored(*values),expected)


if __name__ == "__main__":
    unittest.main()
