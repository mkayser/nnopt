import numpy as np
import random


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
        print "indices={}  y1={:.6}  y2={:.6}  fdiff = {}   deriv = {}   relerror = {}".format(indices, y1, y2, fdiff, deriv, rel_error)


def Hv_check(Hv, g, v, w, delta=1e-4, state_printer=None, random_subset_size=None):

    pairs = list(np.ndenumerate(w))
    if random_subset_size is not None:
        pairs = random.sample(pairs, random_subset_size)

    for indices, val in pairs:
        # Compute Hessian vector product
        v[...] = 0
        v[indices] = 1
        Hv_val = Hv()
        if state_printer: state_printer("Hv")

        # Compute centered finite difference
        # g0 is just debugging info
        g0 = g().copy()
        if state_printer: state_printer("G0")

        w[indices] = val-delta
        g1 = g().copy()
        if state_printer: state_printer("G1")

        w[indices] = val+delta
        g2 = g().copy()
        if state_printer: state_printer("G2")

        w[indices] = val
        gdiff = (g2 - g1)/(2*delta)
        
        rel_error = np.linalg.norm(gdiff-Hv_val)/(np.linalg.norm(gdiff)+np.linalg.norm(Hv_val))
        #print "indices={}  g1={}  g2={}  gdiff = {}   Hv = {}   relerror = {}".format(indices, g1, g2, gdiff, Hv_val, rel_error)
        print "indices={}  relerror = {}".format(indices, rel_error)
