import numpy as np
import cg

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
    return lambda(v): bb(v)+lam*v

def truncatedNewton(w0, model, lambda_0, 
                    X, y, 
                    bbHv_orig, bbMinv, n, 
                    backtrack=False, 
                    momentum=False,
                    damp_dnc=True):
    start = 0
    converged = False
    N = X.shape[1]
    if n==0: n = N

    w = w0

    K=10
    MAX=25
    EPS=.01
    NU=.01
    
    d0 = np.zeros_like(w0)

    lam = lambda_0

    bbHv = bb_shifted(bbHv_orig, lam)

    while not converged:
        # Get subset for gradient/Hv computation
        (start, X_curr, y_curr) = data_batch(X,y,start,n,N)
        model.set_X_y(X_curr, y_curr)
        
        # Compute value and gradient

        (f,g) = model.f_g(w)

        (dd, dnc, dd_hist) = cg.cg(d0, bbHv, bbMinv, -g, MAX, K, EPS, NU)

        print len(dd_hist)
        # Determine search direction
        if backtrack:
            bestd = None
            bestf = None
            bestapproxf = None # Value of quadratic approximation
            print type(dd_hist)
            for d,approxf in reversed(dd_hist):
                (fnew,_) = model.f_g(w+d, compute_g=False)
                if bestf is None or bestf > fnew:
                    (bestf,bestd,bestapproxf) = (fnew,d,approxf)
                elif bestf is not None:
                    # Stop backtracking as soon as score stops improving
                    break
        else:
            bestd = dd
            (bestf, _) = model.f_g(w+dd, compute_g=False) 

        # Set lambda
        actual = f - bestf
        predicted = f - bestapproxf
        ratio = actual / predicted


        if damp_dnc and dnc is not None: 
            lam *= 1.5
        else:
            if ratio < .25: 
                lam *= 1.5
            elif ratio > .75: 
                lam /= 1.5

        bbHv = bb_shifted(bbHv_orig, lam)

        # TODO: use directions of negative curvature in line search
        print "Best dir: norm={:.3}  DNC={}  CG_HIST={}  obj={:.3}  decf={:.2}  decq={:.2}  redratio={:.2}  lambda={:.2}".format(np.linalg.norm(bestd), dnc is not None, len(dd_hist), bestf, actual, predicted, ratio, lam)

        # Update w
        w += bestd

        # Set starting dir for next time
        if momentum:
            d0 = dd
        
    


