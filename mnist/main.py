import numpy as np
import sgd
import mlp
import mnist
from pylab import *

def showimage(m, maxval):
    assert len(m.shape)==2
    
    m = np.maximum(255.0, ((m / maxval)* 255.0)).astype(int)
    imshow(m, cmap=cm.gray)
    show()

def do_mnist():
    np.random.seed(1003)
    images, labels = mnist.load_mnist('training')

    images = images.astype(float)
    images = (images - images.mean())

    maxd = 784

    n = 100
    niter = 15
    lr = .5
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

    sgdobj = sgd.SGD(X, X, n, lrstep, lrmult, lr, max_grad_norm=max_grad_norm)

    sgdobj.train(mlpa, niter)

    w = mlpa.w()[0][0]
    print np.linalg.norm(w), np.max(w), np.min(w), np.shape(w)
    #showimage(w, 1)

def do_test():
    np.random.seed(1003)

    N = 10
    niter = 1000
    lr = .0001
    (lrstep, lrmult) = (10, .9)
    l2reg = .01
    max_grad_norm = 1000000000000 / lr
    std = 0.0001

    X = (np.arange(N)+1).reshape(1,N)
    X = np.concatenate((X,X**2,X**3))
    y = X*2

    d = X.shape[0]
    n = X.shape[1]

    print "X shape = ", X.shape

    #archstr = "a.{}.784_a.784.{}".format(d, d)
    archstr = "a.{}.{}".format(d,d)
    affinit = mlp.GaussianAffineInitializer(std,std)
    mlpa = mlp.MLPAutoencoder(archstr, l2reg, affinit)

    sgdobj = sgd.SGD(X, X, n, lrstep, lrmult, lr, max_grad_norm=max_grad_norm)

    sgdobj.train(mlpa, niter)

    w = mlpa.w()
    print type(w), type(w[0]), type(w[0][0]), w[0][0].shape
    #w = mlpa.w()[0][0]
    #print np.linalg.norm(w), np.max(w), np.min(w), np.shape(w)
    print w[0][0][:10,:10]
    #showimage(w, 1)
    
if __name__ == "__main__":
    #do_mnist()
    do_test()
