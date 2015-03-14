import os, struct
from pylab import *
import numpy as np
import array
import layers

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'data/train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'data/train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 'data/t10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'data/t10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array.array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array.array("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(len(ind)):
        images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


def show_mnist():
    images, labels = load_mnist('training', digits=[2])
    imshow(images.mean(axis=0), cmap=cm.gray)
    show()




if __name__ == "__main__":
    #show_mnist()
    layers.hello()


