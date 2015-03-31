import numpy as np
import sys
import scipy.linalg


def ritz_arnoldi(bbHv, v0, iters_to_compute=(20,40,60,100)):
    maxiter = max(iters_to_compute)
    ritzvals = []
    j=0
    n = v0.shape[0]
    V = np.zeros((n,maxiter+1))
    H = np.zeros((maxiter+1,maxiter+1))

    V[:,0] = v0 / np.linalg.norm(v0)

    for maxiter in iters_to_compute:
        while j < maxiter:
            v = bbHv(V[:,j])
            for i in xrange(0,j+1):
                H[i,j] = V[:,i].dot(v)
                v -= H[i,j] * V[:,i]
            H[j+1,j] = np.linalg.norm(v) 
            V[:,j+1] = v / H[j+1,j]
            j += 1
        Hpart = H[0:j-1,0:j-1]
        #, fmt="%7.3f"
        #np.savetxt(sys.stdout, Hpart, fmt="%7.3f")
        print "norm asym: {}, norm: {}".format(np.linalg.norm(Hpart-Hpart.T), np.linalg.norm(Hpart))
    
        np.savetxt(sys.stdout, np.sort(scipy.linalg.eigvalsh(Hpart))[None,:], fmt="%7.3f")
        print ""


def ritz_lanczos(bbHv, v0, iters_to_compute=(20,40,60,100)):
    maxiter = max(iters_to_compute)
    ritzvals = []
    j=0
    n = v0.shape[0]
    H = np.zeros((maxiter+1,maxiter+1))

    v1 = v0 / np.linalg.norm(v0)
    v0 = None

    for maxiter in iters_to_compute:
        while j < maxiter:
            v = bbHv(v1)
            if j>0:
                H[j-1,j] = v0.dot(v)
                v -= H[j-1,j] * v0

            H[j,j] = v1.dot(v)
            v -= H[j,j] * v1

            H[j+1,j] = np.linalg.norm(v) 
            v0 = v1
            v1 = v / H[j+1,j]
            j += 1
        Hpart = H[0:j-1,0:j-1]
        #, fmt="%7.3f"
        #np.savetxt(sys.stdout, Hpart, fmt="%7.3f")
        print "norm asym: {}, norm: {}".format(np.linalg.norm(Hpart-Hpart.T), np.linalg.norm(Hpart))
    
        np.savetxt(sys.stdout, np.sort(scipy.linalg.eigvalsh(Hpart))[None,:], fmt="%7.7f")
        print ""
