import numpy as np 
from scipy import spatial

def K_Means(X, K, mu=[]):
    if K < 0:
        raise ValueError('K must be greater than 0')
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(mu, list):
        mu = np.array(mu)
    if mu.size == 0:
        mu = np.unique(X, axis=0)
        np.random.shuffle(mu)
        mu = mu[:K]
    mu_prev = np.zeros_like(mu, dtype=np.float64); mu_prev[:] = np.nan
    while (mu_prev != mu).any():
        clusters = np.zeros([K]+ list(X.shape), dtype=np.float64); clusters[:] = np.nan
        distance_matrix = spatial.distance_matrix(X, mu)
        for n, sample in enumerate(X):
            clusters[distance_matrix[n].argmin()][n] = sample
        mu_prev = mu
        mu = np.nanmean(clusters, axis=1)
        
    return mu

def K_Means_better(X, K):
    mus_count = {str(set(map(tuple, list(K_Means(X, K))))): 1}
    for _ in range(1000):
        mu = set(map(tuple, list(K_Means(X, K))))
        if mu in map(eval, mus_count.keys()):
            mus_count[str(mu)] += 1
        else:
            mus_count[str(mu)] = 1
    for mu, count in mus_count.items():
        if all([count >= other_count for other_count in mus_count.values()]):
            mu = eval(mu)
            mu = np.array(list(mu), dtype=np.float64)
            return mu
