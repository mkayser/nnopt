import numpy as np


def gradcheck(f, g, w, delta=1e-4):
    grad = g()
    for indices, val in np.ndenumerate(w):
        # Note we are modifying in place and so we don't pass the modified weights
        # into the f() or g() functions
        w[indices] = val-delta
        y1 = f()

        # print "y1----------------------------------------"
        # print "IO"
        # print mlpa.iostack.m
        # print "Params"
        # print mlpa.paramstack.m


        w[indices] = val+delta
        y2 = f()

        # print "y2----------------------------------------"
        # print "IO"
        # print mlpa.iostack.m
        # print "Params"
        # print mlpa.paramstack.m

        w[indices] = val
        fdiff = (y2 - y1)/(2*delta)
        deriv = grad[indices]
        
        rel_error = abs(fdiff-deriv)/(abs(deriv)+abs(fdiff))
        print "indices={}  y1={:.6}  y2={:.6}  fdiff = {}   deriv = {}   relerror = {}".format(indices, y1, y2, fdiff, deriv, rel_error)
