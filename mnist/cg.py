import numpy as np
from pprint import pprint

# Checks for negative curvature
# MAX = max iterations
# K = see Martens
# EPS = see Martens
# A = sufficient information for implicit computation of bbA
# C = sufficient information for implicit computation of bbC
# x0 = initial point, just use 0, but could use xprev?
def cg(x0, A, M, b, bbA, bbMinv, MAX, K, EPS, NU): 
    x = x0
    Ax = bbA(A,x)
    r = Ax - b
    y = bbMinv(M, r)
    p = -r
    q = x.dot(r)
    VALS = [q];
    ry_prev = r.dot(y)
    k=0

    while not cg_term(np.linalg.norm(b), np.linalg.norm(r), VALS, MAX, K, EPS, NU):
        k += 1
        Ap = bbA(A,p)

        pAp = p.dot(Ap)
        if pAp < 0:
            # DNC
            if k==1:
                return (None,p)
            else:
                return (x,p)
        #pprint(locals())
        alpha = r.dot(y) / pAp
        x = x + alpha * p
        r = r + alpha * Ap
        q = x.dot(r)
        VALS.append(q)
        y = bbMinv(M,r)
        ry = r.dot(y)
        beta = ry / ry_prev
        ry_prev = ry
        p = -y + beta * p
    return (x,None)
        

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
            

def bbMinv_diag(M, x):
    return np.linalg.solve(M,x)
    #return (1/M).dot(x)

def bb_simpleMult(A, x):
    return A.dot(x)

