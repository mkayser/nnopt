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

if __name__ == "__main__":
    np.random.seed(1003)
    images, labels = mnist.load_mnist('training')

    images = images.astype(float) * (1.0 / 255.0)

    maxd = 784

    n = 100
    niter = 15
    lr = .5
    (lrstep, lrmult) = (100, .9)
    l2reg = 0
    max_grad_norm = 10000 / lr

    X = images[:n,:,:]
    X = X.reshape((X.shape[0],-1)).T

    X = X[:maxd,:]

    d = X.shape[0]
    n = X.shape[1]

    print "X shape = ", X.shape

    #archstr = "a.{}.784_a.784.{}".format(d, d)
    archstr = "a.{}.{}_s".format(d,d)
    affinit = mlp.GaussianAffineInitializer(.01, .01)
    mlp = mlp.MLPAutoencoder(archstr, l2reg, affinit)

    sgd = sgd.SGD(X, X, n, lrstep, lrmult, lr, max_grad_norm=max_grad_norm)

    sgd.train(mlp, niter)

    w = mlp.w()[0][0]
    print np.linalg.norm(w), np.max(w), np.min(w), np.shape(w)
    #showimage(w, 1)
    

