import numpy as np
import scipy
import scipy.optimize
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


class TNOpts(object):
    def __init__(self,
                 backtrack=False, 
                 momentum=False,
                 damp_dnc=True,
                 trust_region=True, 
                 curvilinear_line_search=False,
                 descent_line_search=False,
                 verbose=False):
        self.backtrack=backtrack
        self.momentum=momentum
        self.damp_dnc=damp_dnc
        self.trust_region=trust_region
        self.curvilinear_line_search=curvilinear_line_search
        self.descent_line_search=descent_line_search
        self.verbose=verbose

def truncatedNewton(w0, model, lambda_0, 
                    X, y, 
                    bbHv_orig, bbMinv, n, MAXCG, MAXTRUNC, tnopts, statc=None):
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
    samples_trained = 0

    assert sum([tnopts.trust_region, tnopts.curvilinear_line_search, tnopts.descent_line_search]) == 1

    while not converged:
        
        i += 1

        # Get subset for gradient/Hv computation
        (start, X_curr, y_curr) = data_batch(X,y,start,n,N)
        model.set_X_y(X_curr, y_curr)
        
        # Compute value and gradient

        (f,g) = model.f_g(w)

        (dd, dnc, dnc_pAp, dd_hist, reason) = cg.cg(f, d0, bbHv, bbMinv, -g, MAXCG, K, EPS, NU)


        if dnc is not None and g.dot(dnc) > 0:
            dnc = -dnc

        if dd is None:
            dd = -g
            dd_hist = [(dd,None)]

        wstep = None
        fnew = None
        
        if tnopts.trust_region:

            # Determine search direction
            fapproxnew = None # Value of quadratic approximation

            if tnopts.backtrack:
                for d,approxf in reversed(dd_hist):
                    (fcandidate,_) = model.f_g(w+d, compute_g=False)
                    if fnew is None or fnew > fcandidate:
                        (fnew,wstep,fapproxnew) = (fcandidate,d,approxf)
                        #if tnopts.verbose:
                        print "BACKTRACK: F={:.3}  |D|={:.3}  Q={}".format(fnew,np.linalg.norm(wstep),fapproxnew)
                    elif fnew is not None:
                        # Stop backtracking as soon as score stops improving
                        if tnopts.verbose:
                            print "STOP-BACKTRACK: ",fcandidate
                        break
            else:
                wstep = dd
                fapproxnew = dd_hist[-1][1]
                (fnew, _) = model.f_g(w+wstep, compute_g=False) 

            # Set lambda
            fimprovement = f - fnew
            fimprovement_predicted = f - fapproxnew if fapproxnew is not None else None
            ratio = fimprovement / fimprovement_predicted if fapproxnew is not None else None

            dirderiv = g.dot(wstep)
            assert dirderiv<0, "Dirderiv is nonnegative: {}".format(dirderiv)

            if tnopts.damp_dnc and dnc is not None: 
                lam *= 1.5
            else:
                if ratio < .25: 
                    lam *= 1.5
                elif ratio > .75: 
                    lam /= 1.5
            bbHv = bb_shifted(bbHv_orig, lam)

            report_str = "DirDeriv={:.3}  decf={:.2}  decq={}  redratio={}  lambda={:.2}".format(
                dirderiv/(np.linalg.norm(g)*np.linalg.norm(dd)), fimprovement, fimprovement_predicted, ratio, lam)

        elif tnopts.curvilinear_line_search:
            # Perform linesearch
            if dnc is not None:
                (wstep,fnew) = curvilinear_ls(model, w, dd, dnc, dnc_pAp)
            else:
                (wstep,fnew) = descent_ls(model, w, dd)

            report_str = ""

        elif tnopts.descent_line_search:
            (wstep,fnew) = descent_ls(model, w, dd)
            report_str = ""



        # Update w
        if fnew < f:
            w += wstep
        else:
            print "*** WARNING: no improvement ***"

        samples_trained += n

        # Don't count stats collection in time reporting since it can take a while
        statc.add(w, samples_trained, fnew)
        elapsed = statc.elapsed_time()

        print "obj={:.3}  elapsed={:.3}s  starti={}  NormW={:.3}  NormD={:.3}  NormG={:.3}  DNC={}  CG_HIST={}  {}  reason={}".format(fnew, elapsed, start, np.linalg.norm(w), np.linalg.norm(wstep), np.linalg.norm(g), dnc is not None, len(dd_hist),  report_str, reason)
                      

        # Set starting dir for next time
        if tnopts.momentum:
            d0 = wstep
        
        if i==MAXTRUNC:
            converged = True


def curvilinear_ls(model, w, dd, dnc, dnc_pAp, amax=50, beta=.5, sigma=1e-4):

    (f0,g0) = model.f_g(w, compute_g=True)

    for i in xrange(amax):
        alpha = beta ** i
        step = alpha*dd + alpha*alpha*dnc
        est_change = alpha*g0.dot(dd) + .5*(alpha**4)*dnc_pAp
        (f,_) = model.f_g(w+step, compute_g=False)
        actual_change = f - f0
        if actual_change <= sigma * est_change:
            return (step,f)
    assert False

def descent_ls(model, w, dd, amax=50, beta=.1, sigma=1e-1):
    return curvilinear_ls(model, w, dd, 0, 0, amax=amax)
