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
        self.w = w
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
        (dimout, dimin2) = self.w.shape
        (dimout2) = self.b.shape
        
        assert dimin == dimin2
        assert dimout == dimout2
        
        xout = self.w.dot(xin) + self.b
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

        self.db = np.sum(dout, axis=1, keepdims=True)
        self.dw = dout.dot(xin) 
        din = self.w.T.dot(dout)

        return din

    # return gradients
    def grad(self):
        return [self.dw, self.db]
       
    # return weights
    def w(self):
        return [self.w, self.b]

    # return weights
    def wset(self, wval):
        assert len(wval)==2
        assert np.shape(wval[0]) == np.shape(self.w)
        assert np.shape(wval[1]) == np.shape(self.b)
        self.w = wval[0]
        self.b = wval[1]

    # increment weights
    def wadd(self, wval, mult):
        assert len(wval)==2
        assert np.shape(wval[0]) == np.shape(self.w)
        assert np.shape(wval[1]) == np.shape(self.b)
        self.w += wval[0] * mult
        self.b += wval[1] * mult

    def l2reg_loss(self):
        return np.sum(self.w**2)

    def l2reg_loss_grad(self):
        return [2 * self.w, np.zeros_like(self.b)]


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
    def wadd(self, wval, mult):
        assert len(w)==0

    def l2reg_loss(self):
        pass

    def l2reg_loss_grad(self):
        pass


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
        assert len(w)==0

    def l2reg_loss(self):
        pass

    def l2reg_loss_grad(self):
        pass



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
