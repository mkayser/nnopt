import numpy as np


def gradcheck(f, g, w, delta=1e-4):
    grad = g()
    for indices, val in np.ndenumerate(w):
        # Note we are modifying in place and so we don't pass the modified weights
        # into the f() or g() functions
        w[indices] = val-delta
        y1 = f()
        w[indices] = val+delta
        y2 = f()
        w[indices] = val
        fdiff = (y2 - y1)/(2*delta)
        deriv = grad[indices]
        
        rel_error = abs(fdiff-deriv)/(abs(deriv)+abs(fdiff))
        print "fdiff = {}   deriv = {}   relerror = {}".format(fdiff, deriv, rel_error)
