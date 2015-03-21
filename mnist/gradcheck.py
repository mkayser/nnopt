import numpy as np


def gradcheck(f, f_df, wll, delta=1e-4):
    (objval, gradll) = f_df()
    for wl,gl in zip(wll,gradll):
        for w,g in zip(wl,gl):
            for indices, val in np.ndenumerate(w):
                w[indices] = val-delta
                y1 = f()
                w[indices] = val+delta
                y2 = f()
                w[indices] = val
                fdiff = (y2 - y1)/(2*delta)
                deriv = g[indices]

                rel_error = abs(fdiff-deriv)/(abs(deriv)+abs(fdiff))
                print "fdiff = {}   deriv = {}   relerror = {}".format(fdiff, deriv, rel_error)
