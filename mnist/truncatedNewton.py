import numpy as np
import cg

# Columns of X are data points
# Columns of y are corresponding label vectors 

def truncatedNewton(w0, f_df, Hv, lambda_0, 
                    X, y, 
                    n_CG = 1000, 
                    n_g = 0,
                    use_GN = False):
    start = 0
    converged = False
    if n_g==0: n_g = X.shape(axis=1)

    w = w0

    while not converged:
        # Get subset for gradient computation
        assert(start + n_g <= X.shape(axis=1))
        end = start+n_g
        X_curr = X[:,start:end]
        y_curr = y[:,start:end]
        
        # Compute value and gradient
        (_,f,g) = f_df(w,X_curr,y_curr,calc_ypred=False)

        
        
        
        converged = False ### todo
    


