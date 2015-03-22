import numpy as np
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

def do_mnist():
    np.random.seed(1003)
    images, labels = mnist.load_mnist('training')

    images = images.astype(float)
    images = (images - images.mean())

    maxd = 784

    n = 500
    niter = 500
    lr = .000001
    (lrstep, lrmult) = (100, .9)
    l2reg = 0.01
    max_grad_norm = .1 / lr
    std = 0.000001

    X = images[:n,:,:]
    X = X.reshape((X.shape[0],-1)).T

    X = X[:maxd,:]

    d = X.shape[0]
    n = X.shape[1]

    print "X shape = ", X.shape

    #archstr = "a.{}.784_a.784.{}".format(d, d)
    archstr = "a.{}.{}_s".format(d,d)
    affinit = mlp.GaussianAffineInitializer(std,std)
    mlpa = mlp.MLPAutoencoder(archstr, l2reg, affinit)

    w = mlpa.w_debug()[0][0]
    #print np.linalg.norm(w), np.max(w), np.min(w), np.shape(w)
    #print "Done."
    #showimage(w, 1)

    #sgdobj = sgd.SGD(X, X, n, lrstep, lrmult, lr, max_grad_norm=max_grad_norm)
    #sgdobj.train(mlpa, niter)

    mlpa.fwd(X,X)
    mlpa.bwd()
    f = lambda: sum(mlpa.fwd(X,X)[1:])
    g = lambda: mlpa.grad()
    gradcheck.gradcheck(f, g, mlpa.w())


    w = mlpa.w_debug()[0][0]
    #print np.linalg.norm(w), np.max(w), np.min(w), np.shape(w)
    #print "Done."
    showimage(w, w.max())

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
    mlpa = mlp.MLPAutoencoder(archstr, l2reg, affinit)

    #mlpa.fwd(X,y)
    #mlpa.bwd()
    #f = lambda: sum(mlpa.fwd(X,y)[1:])
    #g = lambda: mlpa.grad()
    #gradcheck.gradcheck(f, g, mlpa.w())
    sgdobj = sgd.SGD(X, y, n, lrstep, lrmult, lr, max_grad_norm=max_grad_norm)
    sgdobj.train(mlpa, niter)
    


    
if __name__ == "__main__":
    do_mnist()
    #do_test()
