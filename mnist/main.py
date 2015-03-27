import numpy as np
import scipy as sp
import sgd
import mlp
import mnist
import gradcheck
from pylab import *

def showimage(m, maxval):
    assert len(m.shape)==2
    
    m = m - m.min()
    m = m / m.max()
    m = (m * 255.0).astype(int)
    #m = np.maximum(255.0, ((m / maxval)* 255.0)).astype(int)
    imshow(m, cmap=cm.gray)
    show()

def downsample(images):
    # assumption: dimensions are (n,h,w)
    (n,h,w) = images.shape
    assert h % 2 == 0
    assert w % 2 == 0
    result = np.zeros((n,h/2,w/2))
    for hh in xrange(0,h/2,1):
        for ww in xrange(0,w/2,1):
            result[:,hh,ww] = images[:,hh*2:hh*2+1,ww*2:ww*2+1].mean(axis=(1,2))
    return result

def do_mnist():
    np.random.seed(1003)
    images, labels = mnist.load_mnist('training')

    images = images.astype(float)
    images = (images - images.mean())

    #images = downsample(images)
    #images = downsample(images)

    maxd = images.shape[1] * images.shape[2]

    (n,H,W) = images.shape

    n = 10000
    mbsize = 100
    niter = 5000

    lr = .1
    (lrstep, lrmult) = (1000, .9)
    l2reg = 0.0
    max_grad_norm = .1 / lr
    std = .01

    X = images[:n,:,:]
    X = X.reshape((X.shape[0],-1)).T

    X = X - X.min()
    X = X / X.max()

    X = X[:maxd,:]

    d = X.shape[0]
    n = X.shape[1]

    print "X shape = ", X.shape

    #archstr = "a.{}.100_t_a.100.{}".format(d,d)
    #archstr = "a.{}.100_s.100_a.100.100_s.100_a.100.{}".format(d,d)
    #archstr = "a.{}.100_a.100.{}".format(d,d)
    archstr = "a.{0}.{1}_t.{1}_a.{1}.{1}_t.{1}_a.{1}.{1}_t.{1}_a.{1}.{0}".format(d,50)
    affinit = mlp.GaussianAffineInitializer(std,std)
    mlpa = mlp.MLPAutoencoder(archstr, mbsize, l2reg, affinit)

    w = mlpa.w_debug()[0][0]
    #print np.linalg.norm(w), np.max(w), np.min(w), np.shape(w)
    #print "Done."
    #showimage(w, 1)

    #mlpa.set_w(np.concatenate((np.eye(784).flatten(),np.zeros(784))))

    train = True
    check = False

    if train:
        sgdobj = sgd.SGD(X, X, mbsize, lrstep, lrmult, lr, max_grad_norm=max_grad_norm)
        sgdobj.train(mlpa, niter)

    
    if check:
        mlpa.set_X_y(X[:,:mbsize], X[:,:mbsize])
        
        # Fwd and Bwd passes
        mlpa.fwd(do_Hv=False, do_Gv=False)
        mlpa.bwd(do_Hv=False, do_Gv=False)
        f = lambda: sum(mlpa.fwd(do_Hv=False, do_Gv=False))
        g = lambda: mlpa.get_g()
        gradcheck.gradcheck(f, g, mlpa.get_w())


    if train:
        mlpa.set_X_y(X[:,:mbsize], X[:,:mbsize])
        
        (_, _) = mlpa.fwd(do_Hv=False, do_Gv=False)
        
        ypred = mlpa.get_ypred()
        
        s = 5
        indices = np.random.choice(mbsize, s)
        
        ypredsample = ypred[:,indices]
        Xsample = X[:,indices]
        
        ypredsample = ypredsample.T.reshape((s, H, W))
        Xsample = Xsample.T.reshape((s,H,W))
        
        a = np.lib.pad(ypredsample, ((0,0),(4,4),(4,4)), 'constant')
        b = np.lib.pad(Xsample, ((0,0),(4,4),(4,4)), 'constant')
        
        a = a.reshape(-1,W+8)
        b = b.reshape(-1,W+8)
        
        c = np.hstack((a,b))
        
        showimage(c, c.max())
        
        w_debug = mlpa.w_debug()
        
        matrix0 = w_debug[0][0]
        showimage(matrix0, matrix0.max())
        

def do_gradcheck():
    np.random.seed(1003)

    d = 1
    n = 1
    #X = np.random.randn(d,n)
    X = np.ones((d,n))
    std=.01
    archstr = "a.{0}.{0}_a.{0}.{0}".format(d)
    #archstr = "a.{0}.{0}".format(d)
    affinit = mlp.GaussianAffineInitializer(std,std)
    mlpa = mlp.MLPAutoencoder(archstr, n, 0.0, affinit)

    #sgdobj = sgd.SGD(X, X, n, 1000, 1, .01)
    #sgdobj.train(mlpa, 1000)

    mlpa.set_X_y(X[...], X[...])
    w = mlpa.get_w()
    w[...] = 1.0

    mlpa.fwd(do_Hv=False, do_Gv=False)
    mlpa.bwd(do_Hv=False, do_Gv=False)

    np.set_printoptions(threshold=np.inf)

    f = lambda: sum(mlpa.fwd(do_Hv=False, do_Gv=False))
    g = lambda: mlpa.get_g()
    gradcheck.gradcheck(f, g, mlpa.get_w(), mlpa)
    


def do_Hv_check():
    np.random.seed(1003)

    d = 40
    n = 10
    #X = np.random.randn(d,n)
    X = np.random.randn(d,n)
    std=.01
    archstr = "a.{0}.{0}_s.{0}_a.{0}.{0}_t.{0}_a.{0}.{0}".format(d)
    #archstr = "a.{0}.{0}_t.{0}_a.{0}.{0}".format(d)
    #archstr = "a.{0}.{0}_a.{0}.{0}".format(d)
    #archstr = "a.{0}.{0}".format(d)
    affinit = mlp.GaussianAffineInitializer(std,std)
    mlpa = mlp.MLPAutoencoder(archstr, n, 0.0, affinit)

    #sgdobj = sgd.SGD(X, X, n, 1000, 1, .01)
    #sgdobj.train(mlpa, 1000)

    mlpa.set_X_y(X[...], X[...])
    #w = mlpa.get_w()
    #w[...] = 1.0

    #mlpa.fwd(do_Hv=True, do_Gv=False)
    #mlpa.bwd(do_Hv=True, do_Gv=False)

    np.set_printoptions(threshold=np.inf)

    Hv = lambda: compute_Hv(mlpa)
    g  = lambda: compute_g(mlpa)
    #p  = lambda(msg): print_state(msg, mlpa)
    p = None
    gradcheck.Hv_check(Hv, g, mlpa.get_v(), mlpa.get_w(), state_printer=p)

def print_state(msg, mlpa):
    print "--------",msg,"------------"
    print "IO"
    print mlpa.iostack.m
    print "PARAM"
    print mlpa.paramstack.m

def compute_g(mlpa):
    mlpa.fwd()
    mlpa.bwd()
    return mlpa.get_g()

def compute_Hv(mlpa):
    mlpa.fwd(do_Hv=True)
    mlpa.bwd(do_Hv=True)
    return mlpa.get_Hv()


def do_test():
    np.random.seed(1003)
    np.set_printoptions(precision=1)

    N = 3
    niter = 10000
    lr = 1
    (lrstep, lrmult) = (1000, .1)
    l2reg = .0001
    max_grad_norm = 1000000000000 / lr
    std = 0.0001
    mbsize = N

    #X = (np.arange(N)+1).reshape(1,N)
    #X = np.concatenate((X,X**2,X**3))
    #X = np.concatenate((X**2,X**3))
    X = np.random.randn(2,N)
    X = (X - X.mean())/X.std()
    y = X*2

    d = X.shape[0]
    n = X.shape[1]

    print "X shape = ", X.shape

    #archstr = "a.{}.784_a.784.{}".format(d, d)
    archstr = "a.{}.{}".format(d,d)
    affinit = mlp.GaussianAffineInitializer(std,std)
    mlpa = mlp.MLPAutoencoder(archstr, mbsize, l2reg, affinit)

    #mlpa.fwd(X,y)
    #mlpa.bwd()
    #f = lambda: sum(mlpa.fwd(X,y)[1:])
    #g = lambda: mlpa.grad()
    #gradcheck.gradcheck(f, g, mlpa.w())
    sgdobj = sgd.SGD(X, y, mbsize, lrstep, lrmult, lr, max_grad_norm=max_grad_norm)
    sgdobj.train(mlpa, niter)
    


    
if __name__ == "__main__":
    do_Hv_check()
    #do_mnist()
    #do_test()
    #do_gradcheck()
