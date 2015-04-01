import numpy as np
from pprint import pprint

# Checks for negative curvature
# MAX = max iterations
# K = see Martens
# EPS = see Martens
# x0 = initial point, just use 0, but could use xprev?
def cg(x0, bbA, bbMinv, b, MAX, K, EPS, NU): 
    x = x0
    Ax = bbA(x)
    r = Ax - b
    y = bbMinv(r)
    p = -r
    q = x.dot(r)
    VALS = [q];
    ry_prev = r.dot(y)
    k=0

    dir_hist = []

    bnorm = np.linalg.norm(b)

    while not cg_term(bnorm, np.linalg.norm(r), VALS, MAX, K, EPS, NU):
        k += 1
        Ap = bbA(p)

        pAp = p.dot(Ap)
        if pAp < 0:
            # DNC
            if k==1:
                return (None, p, dir_hist)
            else:
                return (x, p, dir_hist)
        #pprint(locals())
        alpha = r.dot(y) / pAp
        x = x + alpha * p
        r = r + alpha * Ap
        q = x.dot(r)
        VALS.append(q)
        y = bbMinv(r)
        ry = r.dot(y)
        beta = ry / ry_prev
        ry_prev = ry
        dir_hist.append((p,q))
        p = -y + beta * p
    return (x, None, dir_hist)
        

# All termination conditions except negative curvature
def cg_term(r0, r, vals, m, k, eps, resid_mult):

    if r < (r0 * resid_mult):
        return True

    if len(vals) >= m: 
        return True

    if vals[-1] >= 0:
        return False

    if len(vals)<=k:
        return False

    last = vals[-1]
    first = vals[-(k+1)]

    if (last-first)/first < k*eps:
        return True
            
