import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import _num_samples

class LeavePairOut(BaseCrossValidator):
    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        if n_samples <= 2:
            raise ValueError(
                "p={} must be strictly less than the number of samples={}".format(
                    2, n_samples
                )
            )

        labels = np.unique(y)
        if len(labels) != 2:
            print("Number of unique y labels is not 2")

        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if y[i] != y[j]:
                    yield np.array([i, j])

    def get_n_splits(self, X, y=None, groups=None):
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        arr_y = np.array(y)
        labels = np.unique(arr_y)
        if len(labels) != 2:
            raise ValueError("Number of unique y labels is not 2")
        return sum(arr_y == labels[0]) * sum(arr_y == labels[1])
