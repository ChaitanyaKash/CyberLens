"""Paired cluster bootstrap of a fixed pair of multiclass predictions."""
import numpy as np
from scipy.sparse import csr_matrix


def macro_from_confusion(matrix):
    denominator = matrix.sum(axis=0)+matrix.sum(axis=1)
    return float(np.divide(2*np.diag(matrix),denominator,out=np.zeros(len(matrix)),where=denominator>0).mean())


def paired_cluster_interval(y,new,reference,groups,labels,repetitions=500,seed=20261021):
    y,new,reference,groups = map(np.asarray,(y,new,reference,groups))
    if not (len(y)==len(new)==len(reference)==len(groups)) or not len(y):
        raise ValueError("Nonempty, equally sized paired inputs are required")
    index = {label:i for i,label in enumerate(labels)}
    actual = np.array([index[label] for label in y])
    _,inverse = np.unique(groups,return_inverse=True)
    n_groups,k = int(inverse.max()+1),len(labels)
    matrices = []
    for prediction in (new,reference):
        predicted = np.array([index[label] for label in prediction])
        code = actual*k+predicted
        matrices.append(csr_matrix((np.ones(len(y)),(inverse,code)),shape=(n_groups,k*k)))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repetitions):
        weights = np.bincount(rng.integers(0,n_groups,n_groups),minlength=n_groups)
        values = [np.asarray(weights @ matrix).reshape(k,k) for matrix in matrices]
        deltas.append(macro_from_confusion(values[0])-macro_from_confusion(values[1]))
    return {"method":"paired normalized-template cluster bootstrap; existing benchmark, not external confirmation",
            "groups":n_groups,"repetitions":repetitions,"seed":seed,
            "lower_95":float(np.quantile(deltas,.025)),"upper_95":float(np.quantile(deltas,.975))}
