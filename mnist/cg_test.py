import cg
import numpy as np
import scipy.linalg


def testConvex1():
    dim = 3
    x0 = np.zeros((dim,))
    A = np.random.randn(dim,dim)
    A = A.dot(A.T) - (np.eye(dim)*.3)
    b = np.random.randn(dim)
    print scipy.linalg.eigvalsh(A)
    bbA = lambda(v): A.dot(v)
    bbMinv = lambda(v): v
    MAX=10
    K=10
    EPS=0.1
    NU=0.0001

    (dd,dnc,hist) = cg.cg(x0,bbA,bbMinv,b,MAX,K,EPS,NU)
    #print "dd = ", dd, "  \ndnc = ", dnc
    if dnc is not None:
        resid = A.dot(dd) - b
        print np.linalg.norm(b), np.linalg.norm(resid)
        print "g^T d = ", (dd.dot(-b))
        print "DNC: {:.3}".format(dnc.dot(A.dot(dnc)))
    else:
        resid = A.dot(dd) - b
        print np.linalg.norm(resid)


def testDNC1():
    x0 = np.zeros((2,))
    A = 2 * np.eye(2)
    A[1,1] = -1
    M = np.eye(2)
    b = np.array([1,2])
    bbA = cg.bb_simpleMult
    bbMinv = cg.bbMinv_diag
    MAX=3
    K=3
    EPS=0.1
    NU=0.0001

    (dd,dnc) = cg.cg(x0,A,M,b,bbA,bbMinv,MAX,K,EPS,NU)
    print "dd = ", dd, "  \ndnc = ", dnc



if __name__ == "__main__":
    np.random.seed(1003)
    for i in xrange(100):
        testConvex1()
    #testDNC1()
    
