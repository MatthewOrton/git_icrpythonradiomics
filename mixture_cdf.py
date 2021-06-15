from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
import warnings
import numpy as np
from scipy.stats import norm

def gmm_cdf(x, means_, covariances_, weights_):
    _cdf = np.zeros_like(x)
    for n, mu in enumerate(means_):
        gg = norm(loc=mu[0], scale=np.sqrt(covariances_[n][0][0]))
        _cdf = _cdf + weights_[n] * gg.cdf(x)
    return _cdf


class GaussianMixtureCdf(GaussianMixture):

    def cdf(self, x):

        if not self.means_.shape[1] == 1:
            warnings.warn("cdf (cumulative distribution function) only available for 1D densities")
            return None

        return gmm_cdf(x, self.means_, self.covariances_, self.weights_)


class BayesianGaussianMixtureCdf(BayesianGaussianMixture):

    def cdf(self, x):
        if not self.means_.shape[1] == 1:
            warnings.warn("cdf (cumulative distribution function) only available for 1D densities")
            return None

        return gmm_cdf(x, self.means_, self.covariances_, self.weights_)