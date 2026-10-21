import io
import unittest
import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC
from cyberlens.nb_features import NBScaledLinear
from cyberlens.uncertainty import paired_cluster_interval, macro_from_confusion
from cyberlens.improve import validation_rule


class ImprovementTests(unittest.TestCase):
    def test_nb_collapsed_weights_match_explicit_scaled_svm(self):
        x = csr_matrix([[1.,1.,0.],[1.,0.,0.],[0.,1.,1.],[0.,0.,1.],[1.,0.,0.],[0.,0.,1.]])
        y = np.array(["a","a","b","b","a","b"])
        model = NBScaledLinear(workers=1).fit(x,y)
        for i,label in enumerate(model.classes_):
            p = np.asarray(x[y==label].sum(axis=0)).ravel()+1
            q = np.asarray(x[y!=label].sum(axis=0)).ravel()+1
            ratio = np.log((p/p.sum())/(q/q.sum()))
            explicit = LinearSVC(C=1,class_weight="balanced",dual="auto",max_iter=2000,
                                 tol=.001,random_state=42).fit(x.multiply(ratio).tocsr(),y==label)
            np.testing.assert_allclose(model.decision_function(x)[:,i],explicit.decision_function(x.multiply(ratio)),rtol=1e-7,atol=1e-7)
        np.testing.assert_allclose(model.predict_proba(x).sum(axis=1),np.ones(len(y)))
        np.testing.assert_array_equal(model.classes_[model.predict_proba(x).argmax(axis=1)],model.predict(x))
        stream = io.BytesIO()
        joblib.dump(model,stream)
        stream.seek(0)
        np.testing.assert_array_equal(joblib.load(stream).predict(x),model.predict(x))

    def test_paired_cluster_bootstrap_zero_and_perfect_differences(self):
        y = np.array(["a","b","a","b"])
        groups = ["first","first","second","second"]
        identical = paired_cluster_interval(y,y,y,groups,["a","b"],repetitions=50)
        self.assertEqual((identical["lower_95"],identical["upper_95"]),(0.,0.))
        improved = paired_cluster_interval(y,y,np.where(y=="a","b","a"),groups,["a","b"],repetitions=50)
        self.assertEqual((improved["lower_95"],improved["upper_95"]),(1.,1.))
        self.assertAlmostEqual(macro_from_confusion(np.array([[2,1],[1,4]])),(.6666666666666666+.8)/2)

    def test_confirmation_requires_gain_and_accuracy_retention(self):
        def assessment(mean,clean_f1,accuracy,noisy_f1):
            return {"mean":mean,"clean":{"macro_f1":clean_f1,"accuracy":accuracy},"typos":{"macro_f1":noisy_f1}}
        reference = assessment(.3,.31,.5,.29)
        self.assertTrue(validation_rule(assessment(.32,.33,.51,.31),reference))
        self.assertFalse(validation_rule(assessment(.305,.32,.51,.29),reference))
        self.assertFalse(validation_rule(assessment(.32,.33,.48,.31),reference))
        self.assertFalse(validation_rule(assessment(.32,.30,.51,.34),reference))
        self.assertFalse(validation_rule(assessment(.32,.36,.51,.28),reference))


if __name__ == "__main__":
    unittest.main()
