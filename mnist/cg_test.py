import cg
import numpy as np


def testConvex1():
    x0 = np.zeros((2,))
    A = 2 * np.eye(2)
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
    #testConvex1()
    #testDNC1()
    
