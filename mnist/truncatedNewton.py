import numpy as np
import cg
import time

# Columns of X are data points
# Columns of y are corresponding label vectors 

def data_batch(X,y,start,n,N):
    assert(start + n <= N)
    end = start + n
    X_curr = X[:,start:end]
    y_curr = y[:,start:end]
    start = end % N
    return (start, X_curr, y_curr)

def bb_shifted(bb,lam):
    return (lambda(v): bb(v)+lam*v)

class UpdateParams(object):
    def __init__(self, line_search, ls_curv, ):
        self.line_search = line_search
        if line_search:
            self.curvilinear = ls_curv
            self.wolfe_c1 = ls_wolfe_c1
            self.wolfe_c2 = ls_wolfe_c2
            

def truncatedNewton(w0, model, lambda_0, 
                    X, y, 
                    bbHv_orig, bbMinv, n, 
                    MAXTRUNC,
                    MAXCG=50,
                    backtrack=False, 
                    momentum=False,
                    damp_dnc=True,
                    trust_region=True, 
                    verbose=False):
    start = 0
    converged = False
    N = X.shape[1]
    if n==0: n = N

    w = w0.copy()

    K=10
    EPS=.001
    NU=.00001
    
    d0 = np.zeros_like(w0)

    lam = lambda_0

    bbHv = bb_shifted(bbHv_orig, lam)

    i = 0

    starttime = time.clock()
    while not converged:
        
        i += 1

        # Get subset for gradient/Hv computation
        (start, X_curr, y_curr) = data_batch(X,y,start,n,N)
        model.set_X_y(X_curr, y_curr)
        
        # Compute value and gradient

        (f,g) = model.f_g(w)

        (dd, dnc, dd_hist, reason) = cg.cg(f, d0, bbHv, bbMinv, -g, MAXCG, K, EPS, NU)

        dirderiv=999 if dd is None else g.dot(dd)
        assert dd is None or dirderiv < 0
        if dnc is not None and g.dot(dnc) > 0:
            dnc = -dnc

        # Determine search direction
        if backtrack:
            bestd = None
            bestf = None
            bestapproxf = None # Value of quadratic approximation
            for d,approxf in reversed(dd_hist):
                (fnew,_) = model.f_g(w+d, compute_g=False)
                if bestf is None or bestf > fnew:
                    (bestf,bestd,bestapproxf) = (fnew,d,approxf)
                    if verbose:
                        print "BACKTRACK: F={:.3}  |D|={:.3}  Q={:.3}".format(bestf,np.linalg.norm(bestd),bestapproxf)
                elif bestf is not None:
                    # Stop backtracking as soon as score stops improving
                    if verbose:
                        print "STOP-BACKTRACK: ",fnew
                    break
        else:
            bestd = dd
            (bestf, _) = model.f_g(w+dd, compute_g=False) 

        # Set lambda
        actual = f - bestf
        predicted = f - bestapproxf
        ratio = actual / predicted


        if trust_region:
            if damp_dnc and dnc is not None: 
                lam *= 1.5
            else:
                if ratio < .25: 
                    lam *= 1.5
                elif ratio > .75: 
                    lam /= 1.5
            bbHv = bb_shifted(bbHv_orig, lam)

        # TODO: use directions of negative curvature in line search
        elapsed=time.clock()-starttime
        print "***OBJ**={:.3}  elapsed={:.3}s  nextobj={:.3}  starti={}  NormW={:.3}  NormD={:.3}  NormG={:.3}  DirDeriv={:.3}  DNC={}  CG_HIST={}  decf={:.2}  decq={:.2}  redratio={:.2}  lambda={:.2}  reason={}".format(f, elapsed, bestf, start, np.linalg.norm(w), np.linalg.norm(bestd), np.linalg.norm(g), dirderiv/(np.linalg.norm(g)*np.linalg.norm(dd)), dnc is not None, len(dd_hist),  actual, predicted, ratio, lam, reason)

        # Update w
        if actual > 0:
            w += bestd

        # Set starting dir for next time
        if momentum:
            d0 = dd
        
        if i==MAXTRUNC:
            converged = True


