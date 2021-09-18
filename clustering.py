import numpy as np 
from scipy import spatial

def K_Means(X,K,mu=[]):
    if K < 0:
        raise ValueError('K must be greater than 0')
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(mu, list):
        mu = np.array(mu)
    if mu.size == 0:
        mu = X[:]
        np.random.shuffle(mu)
        mu = mu[:K]
    mu_prev = np.zeros_like(mu, dtype=np.float64); mu_prev[:] = np.nan
    while (mu_prev != mu).any():
        clusters = np.empty([K]+ list(X.shape)); clusters[:] = np.nan
        distance_matrix = spatial.distance_matrix(X, mu)
        for n, sample in enumerate(X):
            clusters[distance_matrix[n].argmin()][n] = sample
        mu_prev = mu
        mu = np.nanmean(clusters, axis=1)
    return mu