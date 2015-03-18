import numpy as np

def norm(v):
    norm = sum([sum([(x**2).sum() for x in l]) for l in v])
    return np.sqrt(norm)
