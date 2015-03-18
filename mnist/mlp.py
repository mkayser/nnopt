import numpy as np
import layers
import sys


class AbstractAffineInitializer(object):
    def __init__(self):
        pass
    def initialize(din, dout):
        pass

class GaussianAffineInitializer(AbstractAffineInitializer):
    def __init__(self, wstd, bstd):
        self.wstd = wstd;
        self.bstd = bstd;

    def initialize(din, dout):
        w = self.wstd * np.random.randn(dout,din)
        b = self.bstd * np.random.randn(dout,1)
        return (w,b)

class MLPAutoencoder(object):
    def __init__(self, specifier, l2reg, affinit):
        tokens = specifier.split('_')
        layers = []
        self.affinit = affinit
        for tok in tokens:
            layers.append(makeLayer(tok))
        self.l2reg = l2reg
        self.dout = None
        
    def data_loss(self, ypred, y):
        diff = ypred-y
        loss = .5 * (diff**2)
        din = diff
        return (loss, din)

    def fwd(self, X, y):
        curr = X
        reg_loss = 0
        data_loss = None

        for l in layers:
            curr = l.fwd(curr)
            reg_loss += l.l2reg_loss() * self.l2reg

        ypred = curr

        if y is not None:
            (data_loss,self.dout) = data_loss(ypred, y)
        else:
            self.dout = None
            reg_loss = None

        return (ypred, data_loss, reg_loss)

    def bwd(self):
        curr = self.dout
        for l in reversed(self.layers):
            curr = l.bwd(curr)
        
    def grad(self):
        val = []
        for l in self.layers:
            grad = l.grad() + (self.l2reg * l.l2reg_loss_grad())
            val.append(grad)
        return val

    def w(self):
        val = []
        for l in self.layers:
            val.append(l.w())
        return val

    def wset(self, w):
        for (l,lw) in zip(self.layers, w):
            l.wset(lw)

    def wadd(self, wincr, mult):
        for (l,lwincr) in zip(self.layers, wincr):
            l.wadd(lwincr, mult)

    def makeLayer(specifier):
        maff = re.match(r"a\.(\d+)\.(\d+)$", specifier)
        mrelu = re.match(r"r$", specifier)
        mtanh = re.match(r"t$", specifier)
       
        if maff:
            din = int(maff.group(1))
            dout = int(maff.group(2))
            (w,b) = self.affinit.initialize(din,dout)
            return layers.AffineLayer(w,b)
            
        elif mrelu:
            return layers.ReluLayer()

        elif mtanh:
            return layers.TanhLayer()

        else:
            sys.exit("Unknown layer: "+ specifier)
