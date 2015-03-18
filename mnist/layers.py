import numpy as np

class AbstractLayer(object):
    def __init__(self):
        pass

    # return xout
    def fwd(self, xin):
        pass

    # return din
    def bwd(self, dout):
        pass

    # return gradients
    def grad(self):
        pass
       
    # return weights
    def w(self):
        pass

    # return weights
    def wset(self, w):
        pass

    # return weights
    def wadd(self, wincr, mult):
        pass

    def l2reg_loss(self):
        pass

    def l2reg_loss_grad(self):
        pass
    

class AffineLayer(AbstractLayer):
    def __init__(self,w,b):
        self.w_ = w
        self.b = b
        self.dw = None
        self.db = None
        self.cached_xin = None

    def fwd(self, xin):
        # xin is dimin x n
        # w is dimout x dimin
        # b is dimout x 1
        # xout is dimout x n
        (dimin, n) = xin.shape
        (dimout, dimin2) = self.w_.shape
        (dimout2, _) = self.b.shape

        #print "Affine layer"
        #print "xin shape = ", xin.shape
        #print "w shape = ", self.w_.shape
        #print "b shape = ", self.b.shape
        
        assert dimin == dimin2, "input dims {} does not agree with {}".format(dimin, dimin2)
        assert dimout == dimout2
        
        xout = self.w_.dot(xin) + self.b
        self.cached_xin = xin
    
        return xout

    def bwd(self, dout):
        # dout is dimout x n
        # din is dimin x n
        # xin is dimin x n
        # w is dimout x dimin
        # dw is dimout x dimin
        # db is dimout x 1
        # returns: din, dw, db

        xin = self.cached_xin

        #print "Affine BWD: dout is ", dout.shape, " ", 
        self.db = np.sum(dout, axis=1, keepdims=True)
        self.dw = dout.dot(xin.T) 
        din = self.w_.T.dot(dout)

        return din

    # return gradients
    def grad(self):
        return [self.dw, self.db]
       
    # return weights
    def w(self):
        return [self.w_, self.b]

    # return weights
    def wset(self, wval):
        assert len(wval)==2
        assert np.shape(wval[0]) == np.shape(self.w_)
        assert np.shape(wval[1]) == np.shape(self.b)
        self.w_ = wval[0]
        self.b = wval[1]

    # increment weights
    def wadd(self, wval, mult):
        assert len(wval)==2
        assert np.shape(wval[0]) == np.shape(self.w_)
        assert np.shape(wval[1]) == np.shape(self.b)
        self.w_ += wval[0] * mult
        self.b += wval[1] * mult

    def l2reg_loss(self):
        return np.sum(self.w_**2)

    def l2reg_loss_grad(self):
        return [2 * self.w_, np.zeros_like(self.b)]


class ReluLayer(AbstractLayer):
    def __init__(self):
        self.cached_xin = None

    # return xout
    def fwd(self, xin):
        self.cached_xin = xin
        xout = xin * (xin > 0)
        return xout

    # return din
    def bwd(self, dout):
        xin = self.cached_xin
        din = (xin > 0) * dout
        return din

    # return gradients
    def grad(self):
        return []
       
    # return weights
    def w(self):
        return []

    # return weights
    def wset(self, w):
        assert len(w)==0

    # return weights
    def wadd(self, wincr, mult):
        assert len(wincr)==0

    def l2reg_loss(self):
        return 0

    def l2reg_loss_grad(self):
        []


class TanhLayer(AbstractLayer):
    def __init__(self):
        self.cached_tanh_xin = None

    # return xout
    def fwd(self, xin):
        xout = np.tanh(xin)
        self.cached_tanh_xin = xout
        return xout

    # return din
    def bwd(self, dout):
        tanh_xin = self.cached_tanh_xin
        din = 1 - (tanh_xin**2)
        return din

    # return gradients
    def grad(self):
        return []
       
    # return weights
    def w(self):
        return []

    # return weights
    def wset(self, w):
        assert len(w)==0

    # return weights
    def wadd(self, wincr, mult):
        assert len(wincr)==0

    def l2reg_loss(self):
        return 0

    def l2reg_loss_grad(self):
        return []


class SigmoidLayer(AbstractLayer):
    def __init__(self):
        self.cached_sigmoid_xin = None

    # return xout
    def fwd(self, xin):
        xout = 1.0/(1.0 + np.exp(-xin))
        self.cached_sigmoid_xin = xout
        assert np.all(xout <= 1)
        assert np.all(xout >= 0)
        return xout

    # return din
    def bwd(self, dout):
        sigmoid_xin = self.cached_sigmoid_xin
        din = sigmoid_xin * (1.0 - sigmoid_xin)
        return din

    # return gradients
    def grad(self):
        return []
       
    # return weights
    def w(self):
        return []

    # return weights
    def wset(self, w):
        assert len(w)==0

    # return weights
    def wadd(self, wincr, mult):
        assert len(wincr)==0

    def l2reg_loss(self):
        return 0

    def l2reg_loss_grad(self):
        return []




###### Unused stuff
# def softmax_loss(x,y):
#     p = np.exp(x - x.max(axis=0,keepdims=True))
#     p /= np.sum(p, axis=0, keepdims=True)
#     n = p.shape[1]
#     loss = -np.sum(np.log(p[y,np.arange(n)])) / n
#     dx = p.copy()
#     dx[y,np.arange(n)] -= 1
#     dx /= n
#     # gradient with respect to probability judgements
    

# def softmax_fwd(xin,w,b):
#     xout = w.dot(xin) + b
#     cache = {}
#     cache['xin'] = xin
#     cache['w'] = w
#     return (xout, cache)

# def softmax_bwd(dout):
#     pass

# def crossent_fwd(xin,y):
#     pass

#def crossent_bwd(loss, 
