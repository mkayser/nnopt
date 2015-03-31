import numpy as np
from blobstack import BlobStack

class AbstractLayer(object):
    def __init__(self):
        pass

    def fwd(self, do_val=False, do_gtval=False):
        pass

    def bwd(self, do_g=False, do_Hv=False, do_Gv=False):
        pass

    def get_istack(self):
        return self.istack

    def get_ostack(self):
        return self.ostack

    def w_debug(self):
        return []

    def l2reg_loss(self):
        pass

    

class AffineLayer(AbstractLayer):
    # Parameters are allocated from an underlying memory block
    # There are no accessors for parameters or gradients.
    # Instead, an external class owns parameters and gradients as a single
    # large flattened array.
    def __init__(self, wstack, bstack, istack, ostack, l2reg):
        self.wstack = wstack
        self.bstack = bstack
        self.istack = istack
        self.ostack = ostack
        self.l2reg = l2reg

    def fwd(self, do_val=False, do_gtval=False):

        # Compute in place
        if do_val:
            o = self.ostack.val
            i = self.istack.val
            w = self.wstack.val
            b = self.bstack.val

            o[...] = w.dot(i) + b
            #print "FWD: ({})dot({}) + {} = {}".format(w.flatten()[0], i.flatten()[0], b.flatten()[0], o.flatten()[0])

        if do_gtval:
            i = self.istack.val
            w = self.wstack.val

            ro = self.ostack.gtval
            ri = self.istack.gtval
            rw = self.wstack.gtval
            rb = self.bstack.gtval

            # Product rule plus the bias:
            # w*ri + rw*i + rb
            ro[...] = w.dot(ri) + rw.dot(i) + rb

    def bwd(self, do_g=False, do_Hv=False, do_Gv=False):

        # Compute in place
        if do_Hv or do_g:

            i = self.istack.val
            w = self.wstack.val

            go = self.ostack.g
            gi = self.istack.g
            gw = self.wstack.g
            gb = self.bstack.g

            gb[...] = np.sum(go, axis=1, keepdims=True)
            gw[...] = go.dot(i.T) + self.l2reg * 2 * w
            gi[...] = w.T.dot(go)

        if do_Hv:
            w = self.wstack.val
            i = self.istack.val

            go = self.ostack.g

            rw = self.wstack.gtval
            ri = self.istack.gtval

            Hvb = self.bstack.Hv
            Hvw = self.wstack.Hv
            Hvo = self.ostack.Hv
            Hvi = self.istack.Hv

            # Each of these equations is simply R{} applied to the above
            Hvb[...] = np.sum(Hvo, axis=1, keepdims=True)
            Hvw[...] = Hvo.dot(i.T) + go.dot(ri.T) + self.l2reg * 2 * rw
            Hvi[...] = w.T.dot(Hvo) + rw.T.dot(go)

        if do_Gv:
            i = self.istack.val
            w = self.wstack.val

            Gvo = self.ostack.Gv
            Gvi = self.istack.Gv
            Gvw = self.wstack.Gv
            Gvb = self.bstack.Gv

            Gvb[...] = np.sum(Gvo, axis=1, keepdims=True)
            Gvw[...] = Gvo.dot(i.T) + self.l2reg * 2 * w
            Gvi[...] = w.T.dot(Gvo)


    def w_debug(self):
        w = self.wstack.val
        b = self.bstack.val
        return [w, b]

    def l2reg_loss(self):
        w = self.wstack.val
        return self.l2reg * np.sum(w**2)



class ReluLayer(AbstractLayer):
    def __init__(self, istack, ostack):
        self.istack = istack
        self.ostack = ostack

    def fwd(self, do_val=False, do_gtval=False):
        # Compute in place
        if do_val:
            o = self.ostack.val
            i = self.istack.val
            
            o[...] = i * (i>0)

        if do_gtval:
            i = self.istack.val

            ro = self.ostack.gtval
            ri = self.istack.gtval

            # Result of R{} applied to above
            ro[...] = ri * (i>0)

    def bwd(self, do_g=False, do_Hv=False, do_Gv=False):

        # Compute in place
        if do_Hv or do_g:
            i = self.istack.val

            go = self.ostack.g
            gi = self.istack.g

            gi[...] = go * (i>0)

        if do_Hv:
            i = self.istack.val

            Hvo = self.ostack.Hv
            Hvi = self.istack.Hv

            Hvi[...] = Hvo * (i>0)

        if do_Gv:
            i = self.istack.val

            Gvo = self.ostack.Gv
            Gvi = self.istack.Gv

            Gvi[...] = Gvo * (i>0)

    def l2reg_loss(self):
        return 0


class TanhLayer(AbstractLayer):
    def __init__(self, istack, ostack):
        self.istack = istack
        self.ostack = ostack

    def fwd(self, do_val=False, do_gtval=False):
        # Compute in place
        if do_val:
            o = self.ostack.val
            i = self.istack.val
            
            o[...] = np.tanh(i)

        if do_gtval:
            o = self.ostack.val
            i = self.istack.val

            ro = self.ostack.gtval
            ri = self.istack.gtval

            ro[...] = (1 - (o**2))*ri

    def bwd(self, do_g=False, do_Hv=False, do_Gv=False):

        # Compute in place
        if do_Hv or do_g:
            o = self.ostack.val
            i = self.istack.val

            go = self.ostack.g
            gi = self.istack.g

            gi[...] = (1 - (o ** 2)) * go

        if do_Hv:
            o = self.ostack.val
            i = self.istack.val

            go = self.ostack.g

            ro = self.ostack.gtval

            Hvo = self.ostack.Hv
            Hvi = self.istack.Hv

            Hvi[...] = Hvo - ((Hvo * o**2) + (go * 2 * o * ro))

        if do_Gv:
            o = self.ostack.val
            i = self.istack.val

            Gvo = self.ostack.Gv
            Gvi = self.istack.Gv

            Gvi[...] = (1 - (o ** 2)) * Gvo

    def l2reg_loss(self):
        return 0


class SigmoidLayer(AbstractLayer):
    def __init__(self, istack, ostack):
        self.istack = istack
        self.ostack = ostack

    def fwd(self, do_val=False, do_gtval=False):

        # Compute in place
        if do_val:
            o = self.ostack.val
            i = self.istack.val
            zeromask = i<-45
            onemask  = i>45
            calcmask = np.logical_not(np.logical_or(zeromask,onemask))
            
            # Readability over efficiency for now
            o[...] = np.zeros_like(i)
            o[onemask]  = 1
            o[calcmask] = 1.0/(1.0 + np.exp(-i[calcmask]))
            
        if do_gtval:
            o = self.ostack.val
            ri = self.istack.gtval
            ro = self.ostack.gtval

            # R{sigmoid(i)} = R{i} * sigmoid'(i)
            ro[...] = ri * o * (1-o)

    def bwd(self, do_g=False, do_Hv=False, do_Gv=False):

        #din[...] = sigmoid_xin * (1.0 - sigmoid_xin) * dout
        # Compute in place
        if do_Hv or do_g:
            o = self.ostack.val
            i = self.istack.val

            go = self.ostack.g
            gi = self.istack.g

            gi[...] = o * (1-o) * go

        if do_Hv:
            o = self.ostack.val
            i = self.istack.val

            go = self.ostack.g

            ro = self.ostack.gtval

            Hvo = self.ostack.Hv
            Hvi = self.istack.Hv

            # R{gi} = R{o go - o**2 go} 
            #       = R{o} go + o R{go} - (R{o**2} go + o**2 R{go}) 
            #       = R{o} go + o R{go} - (2o * R{o} go + o**2 R{go}) 
            Hvi[...] = ro*go + o*Hvo - (2*o*ro*go + (o**2 * Hvo))

        if do_Gv:
            o = self.ostack.val
            i = self.istack.val

            Gvo = self.ostack.Gv
            Gvi = self.istack.Gv

            Gvi[...] = o * (1-o) * Gvo


    def l2reg_loss(self):
        return 0


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
