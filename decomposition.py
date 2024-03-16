import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CustomPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.E = None
        self.n_components = n_components
        self.eigen_values = None

    def eigen_helper(self, covarianceMat):
        E, self.eigen_values, _ = np.linalg.svd(covarianceMat)
        # sorting eigen values
        sorted_E = np.append(E, [self.eigen_values], axis=0)
        sorted_E = sorted_E[:, np.argsort(sorted_E[-1, :])]
        return sorted_E[:-1, :]

    def fit(self, X, y=None):
        cov_mat = np.dot(X.T, X) * 1/len(X)
        self.E = self.eigen_helper(cov_mat)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        cov_mat = np.dot(X.T, X) * 1/(len(X))
        self.E = self.eigen_helper(cov_mat)
        return np.dot(X, self.E[:, :self.n_components])

    def transform(self, X):
        return np.dot(X, self.E[:, :self.n_components])


class CustomKernelPCA(BaseEstimator, TransformerMixin):
    def fit_transform(self, X, y=None, **fit_params):
        # compute x^2
        X2 = np.sum(np.square(X), axis=1)
        # Pairwise squared euclidean distance
        euclid_dist = X2[:, np.newaxis] + X2 - 2 * np.dot(X, X.T)
        # kernel matrix of size NxN
        sigma = 5
        K = np.exp(-euclid_dist/sigma)
        # normalise kernel to correspond to zero-mean
        U = np.ones_like(K)/len(K)
        Kn = K - U * K - K * U + U * K * U
        # obtain the eigen values and eigen vectors for the kernel matrix
        eigen_vectors, eigen_values, _ = np.linalg.svd(Kn)
        return Kn * eigen_vectors.T
