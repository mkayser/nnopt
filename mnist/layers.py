import numpy as np


def aff_fwd(xin,w,b):
    # xin is dimin x n
    # w is dimout x dimin
    # b is dimout x 1
    # xout is dimout x n
    (dimin, n) = xin.shape
    (dimout, dimin2) = w.shape
    (dimout2) = b.shape

    assert dimin == dimin2
    assert dimout == dimout2

    cache = {}
    xout = w.dot(xin) + b
    cache['xin'] = xin
    cache['w'] = w
    
    return (xout, cache)

def aff_bwd(dout,cache):
    # dout is dimout x n
    # din is dimin x n
    # xin is dimin x n
    # w is dimout x dimin
    # dw is dimout x dimin
    # db is dimout x 1
    # returns: din, dw, db

    xin = cache['xin']
    w = cache['w']

    db = np.sum(dout, axis=1)
    dw = dout.dot(xin)
    din = W.T.dot(dout)

    return (din,dw,db)

def relu_fwd(xin):
    cache('xin') = xin
    xout = xin * (xin > 0)
    return (xout, cache)


def relu_bwd(dout):
    xin = cache('xin')
    din = (xin > 0) * dout
    return (din)


def softmax_fwd(xin):
    pass

def softmax_bwd(dout):
    pass

def crossent_fwd(xin,y):
    pass

def crossent_bwd(loss, 
