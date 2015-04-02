import numpy as np
from pprint import pprint

# Checks for negative curvature
# MAX = max iterations
# K = see Martens
# EPS = see Martens
# x0 = initial point, just use 0, but could use xprev?
def cg(f0, x0, bbA, bbMinv, b, MAX, K, EPS, NU): 
    x = x0
    Ax = bbA(x)
    r = Ax - b
    y = bbMinv(r)
    p = -r
    q = f0 + (.5 * x.dot(Ax)) - b.dot(x)
    VALS = [q];
    #print "Q=",q, " r=",np.linalg.norm(r)
    exit
    ry_prev = r.dot(y)
    k=0

    dir_hist = []

    bnorm = np.linalg.norm(b)
    finished=False
    reason = None

    while not finished:
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
        rdotytemp = r.dot(y)
        alpha = r.dot(y) / pAp
        x = x + alpha * p
        r = r + alpha * Ap
        #q = x.dot(r)
        q = f0 + (.5 * x.dot(bbA(x))) - b.dot(x)
        #print "  rdoty={}  pAp={}  alpha={}".format(
        #    rdotytemp, pAp, alpha)
        VALS.append(q)
        y = bbMinv(r)
        ry = r.dot(y)
        beta = ry / ry_prev
        ry_prev = ry
        dir_hist.append((p,q))
        p = -y + beta * p
        (finished, reason) = cg_term(bnorm, np.linalg.norm(r), VALS, MAX, K, EPS, NU)

    return (x, None, dir_hist, reason)
        

# All termination conditions except negative curvature
def cg_term(r0, r, vals, m, k, eps, resid_mult):

    if r < (r0 * resid_mult):
        return (True,"ResidMult={:.3},{:.3}".format(r,r0))

    if len(vals) >= m: 
        return (True,"Max")

    if vals[-1] >= 0:
        return (False,"")

    if len(vals)<=k:
        return (False,"")

    last = vals[-1]
    first = vals[-(k+1)]

    if (last-first)/first < k*eps:
        return (True,"Eps({:.3},{:.3},{:.5})".format(first, last, (last-first)/first))
            
