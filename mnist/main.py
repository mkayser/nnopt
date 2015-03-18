import numpy as np
import sgd
import mlp
import mnist


if __name__ == "__main__":
    images, labels = mnist.load_mnist('training', digits=[2])
    
    images = images[:100,:,:]
    images = images.reshape[:,-1].T

    imagesize = images.shape[1]

    archstr = "a.{}.128_a.128_{}".format(imagesize, imagesize)
    affinit = GaussianAffineInitializer(.01, .01)
    mlp = mlp.MLPAutoencoder(archstr, 0, affinit)

    
    

