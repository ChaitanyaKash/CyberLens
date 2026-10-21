"""Class-specific NB feature scaling with one-versus-rest linear margins.

Inspired by Wang & Manning (2012), https://aclanthology.org/P12-2018/.
This multiclass variant uses balanced binary SVMs without NB/SVM interpolation.
"""
import numpy as np
from joblib import Parallel, delayed
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC


def fit_binary(x, positive, total, c):
    p = np.asarray(x[positive].sum(axis=0)).ravel()+1.
    q = total-p+2.  # total - unsmoothed positive counts + smoothing
    ratio = np.log((p/p.sum())/(q/q.sum()))
    model = LinearSVC(C=c,class_weight="balanced",dual="auto",max_iter=2000,
                      tol=.001,random_state=42).fit(x.multiply(ratio).tocsr(),positive)
    # Per-estimator iteration metadata avoids process-global warning capture in threads.
    diagnostics = ["IterationLimitReached"] if int(np.max(model.n_iter_)) >= model.max_iter else []
    return model.coef_[0]*ratio,float(model.intercept_[0]),diagnostics


class NBScaledLinear(ClassifierMixin, BaseEstimator):
    def __init__(self, c=1., workers=4):
        self.c,self.workers = c,workers

    def fit(self,x,y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2 or x.shape[0] != len(y):
            raise ValueError("At least two classes and matching row counts are required")
        total = np.asarray(x.sum(axis=0)).ravel()
        results = Parallel(n_jobs=self.workers,prefer="threads")(
            delayed(fit_binary)(x,y==label,total,self.c) for label in self.classes_)
        self.coef_ = np.vstack([row[0] for row in results])
        self.intercept_ = np.array([row[1] for row in results])
        self.fit_warnings_ = {str(label):row[2] for label,row in zip(self.classes_,results) if row[2]}
        self.n_features_in_ = x.shape[1]
        return self

    def decision_function(self,x):
        return np.asarray(x @ self.coef_.T)+self.intercept_

    def predict(self,x):
        return self.classes_[self.decision_function(x).argmax(axis=1)]

    def predict_proba(self,x):
        # Ranking interface only: this does not calibrate the SVM margins.
        return softmax(self.decision_function(x),axis=1)
