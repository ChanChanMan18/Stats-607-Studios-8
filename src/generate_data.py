# this file generates synthetic data and returns the parameters for the simulation as well as the data
import numpy as np
from scipy import stats
import math

def sample_X(n, p, rng):
    """
    Standard gaussian design 
    """
    normal_mat = rng.standard_normal(size = (n, p))
    return normal_mat

def make_beta(p, r):
    """
    Create coefficient vector
    """
    b = math.sqrt(r**2/p) * np.ones(p)
    return b


def sample_errors(n, sigma, rng):
    """
    Standard gaussian errors scaled to sigma = 1.
    """
    err = rng.standard_normal(n) * sigma
    return err


def simulate_dataset(gamma, n = 200, seed = None):
    if seed is None:
        rng = np.random.default_rng(82803)
    else:
        rng = np.random.default_rng(seed)
    
    p = int(np.floor(gamma * n)) #should be the floor
    sigma = 1.0
    r = math.sqrt(5.0)
    X = sample_X(n, p, rng)
    beta = make_beta(p, r)
    eps = sample_errors(n, sigma, rng=rng)
    y = X @ beta + eps

    return {
        "X": X,
        "y": y,
        "beta": beta,
        "params": {"n": n, "p": p, "gamma": gamma, "sigma": sigma, "r": r, "seed": seed}
    }