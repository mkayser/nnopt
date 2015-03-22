import numpy as np
import layers
import sys
import re

class AbstractAffineInitializer(object):
    def __init__(self):
        pass
    def initialize(w,b):
        pass

class GaussianAffineInitializer(AbstractAffineInitializer):
    def __init__(self, wstd, bstd):
        self.wstd = wstd;
        self.bstd = bstd;

    def initialize(self, w, b):
        w[...] = self.wstd * np.random.randn(w.size)
        b[...] = self.bstd * np.random.randn(b.size)

class MLPAutoencoder(object):
    def __init__(self, specifier, l2reg, affinit):
        tokens = specifier.split('_')
        self.layers = []
        self.layerSizes = []
        self.affinit = affinit

        # Step through layer specifications and get parameter
        # sizes
        totalSize = 0
        for tok in tokens:
            size = self.makeLayer(tok, l2reg, None, None, getParamSize=True)
            self.layerSizes.append(size)
            totalSize += size

        # Allocate underlying arrays
        # Each NN layer will have views into these arrays
        # For higher level classes like SGD, parameters can be
        # treated as a single flat vector, which is convenient
        self.w_ = np.zeros(totalSize)
        self.grad_ = np.zeros(totalSize)

        # Make the actual layers
        start=0
        for tok,size in zip(tokens,self.layerSizes):
            wslice = self.w_[start:start+size]
            gslice = self.grad_[start:start+size]
            layer = self.makeLayer(tok, l2reg, wslice, gslice)
            self.layers.append(layer)
        self.l2reg = l2reg
        self.dout = None
        
    def data_loss(self, ypred, y):

        diff = ypred-y
        loss = np.sum(.5 * (diff**2)) / diff.shape[1]
        din = diff / diff.shape[1]
        return (loss, din)

    def fwd(self, X, y):
        curr = X
        reg_loss = 0
        data_loss = None

        for l in self.layers:
            curr = l.fwd(curr)
            reg_loss += l.l2reg_loss()

        ypred = curr

        if y is not None:
            (data_loss,self.dout) = self.data_loss(ypred, y)
        else:
            self.dout = None
            reg_loss = None

        return (ypred, data_loss, reg_loss)

    def bwd(self):
        curr = self.dout
        for l in reversed(self.layers):
            curr = l.bwd(curr)
        
    def grad(self):
        return self.grad_

    def w(self):
        return self.w_

    def w_debug(self):
        wlist = []
        for l in self.layers:
            wlist.append(l.w_debug())
        return wlist

    # def wset(self, w):
    #     for (l,lw) in zip(self.layers, w):
    #         l.wset(lw)

    # def wadd(self, wincr, mult):
    #     for (l,lwincr) in zip(self.layers, wincr):
    #         l.wadd(lwincr, mult)

    def makeLayer(self, specifier, l2reg, wslice, gslice, getParamSize=False):
        print "Layer spec: ",specifier
        maff = re.match(r"a\.(\d+)\.(\d+)$", specifier)
        mrelu = re.match(r"r$", specifier)
        mtanh = re.match(r"t$", specifier)
        msigmoid = re.match(r"s$", specifier)
       
        if maff:
            din = int(maff.group(1))
            dout = int(maff.group(2))
            if getParamSize: return din*dout + dout

            assert wslice.size == (din*dout)+dout
            assert gslice.size == (din*dout)+dout

            # This is the conversion from a flat array to the
            # matrices that AffineLayer uses for its parameters.
            # Note that we are only slicing and reshaping, and because
            # of the nature of the reshaping, we never copy the array.

            # This is important because it means that there are now
            # two views of the same parameter vector: the external view
            # which MLP uses, which is a flat vector, and the
            # layer-specific view, which sees slices of this vector
            # in the shape of matrices.
            _wslice = wslice[:(din*dout)]
            _bslice = wslice[(din*dout):]
            _gwslice = gslice[:(din*dout)]
            _gbslice = gslice[(din*dout):]
            
            self.affinit.initialize(_wslice,_bslice)

            _wslice  = _wslice.reshape(dout, din)
            _bslice  = _bslice.reshape(dout, 1)
            _gwslice = _gwslice.reshape(dout, din)
            _gbslice = _gbslice.reshape(dout, 1)
            return layers.AffineLayer(_wslice,_bslice, _gwslice, _gbslice, l2reg)
            
        elif mrelu:
            if getParamSize: return 0 
            assert wslice.size==0
            assert gslice.size==0
            return layers.ReluLayer()

        elif mtanh:
            if getParamSize: return 0
            assert wslice.size==0
            assert gslice.size==0
            return layers.TanhLayer()

        elif msigmoid:
            if getParamSize: return 0
            assert wslice.size==0
            assert gslice.size==0
            return layers.SigmoidLayer()

        else:
            sys.exit("Unknown layer: "+ specifier)
