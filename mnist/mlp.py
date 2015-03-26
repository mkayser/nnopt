import numpy as np
import layers
import sys
import re
from enum import Enum
from blobstack import BlobStack

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

        

# This is a little quirky because the autoencoder needs to know about minibatch size,
# which is a training-specific piece of information. Could fix this later
class MLPAutoencoder(object):
    def __init__(self, specifier, mbsize, l2reg, affinit):
        tokens = specifier.split('_')
        self.layers = []
        self.layerSizes = []
        self.affinit = affinit

        # Step through layer specifications and get parameter
        # sizes
        totalPSize = 0
        totalIOSize = 0
        for i,tok in enumerate(tokens):
            (isize, osize, psize) = self.makeLayer(tok, l2reg, mbsize, None, None, None, getParamSize=True)
            self.layerSizes.append((isize, osize, psize))
            totalPSize += psize
            if i==0: totalIOSize += isize
            totalIOSize += osize
            
        
        
        # Allocate underlying array
        # Each NN layer will have views into this array
        # For higher level classes like SGD, parameters can be
        # treated as a single flat vector, which is convenient
        # w, g*, gtval, Hv*, Gv*
        self.paramstack = BlobStack(np.zeros((5,totalPSize), order='c'))
        self.iostack = BlobStack(np.zeros((5,totalIOSize), order='c'))

        # Make the actual layers
        pstart=0
        iostart=0
        for i, (tok, (isize, osize, psize)) in enumerate(zip(tokens,self.layerSizes)):
            pslice = self.paramstack.subblob(pstart,pstart+psize)

            islice = self.iostack.subblob(iostart, iostart+isize)
            iostart += isize
            oslice = self.iostack.subblob(iostart, iostart+osize)

            layer = self.makeLayer(tok, l2reg, mbsize, pslice, islice, oslice)
            self.layers.append(layer)
        self.l2reg = l2reg
        
    def data_loss(self, ypred, y):

        diff = ypred-y
        loss = np.sum(.5 * (diff**2)) / diff.shape[1]
        din = diff / diff.shape[1]
        return (loss, din)

    #TODO: modify SGD to use make_blobs and use input/output memory cleanly
    def fwd(self, propmode=PropMode.Normal):
        reg_loss = 0
        data_loss = None

        for i,l in enumerate(self.layers):
            inblob = self.iblobs[i] 
            outblob = self.oblobs[i]
            rinblob = self.r_iblobs[i]
            routblob = self.r_oblobs[i]
            l.fwd(inblob, outblob, rinblob, routblob)
            reg_loss += l.l2reg_loss()

        if self.y is not None:
            (data_loss,self.dout) = self.data_loss(self.ypred, self.y)
            # TODO: backprop R
            # TODO: dout is just another dout_blob

        else:
            self.dout = None
            reg_loss = None

        return (data_loss, reg_loss)

    def bwd(self, r=False):
        for i,l in reversed(list(enumerate(self.layers))):
            doutblob = self.g_oblobs[i]
            dinblob  = self.g_iblobs[i]
            l.bwd(doutblob, dinblob)
        
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

    def makeLayer(self, specifier, l2reg, mbsize, pstack, istack, ostack):
        print "Layer spec: ",specifier
        maff = re.match(r"a\.(\d+)\.(\d+)$", specifier)
        mrelu = re.match(r"r\.(\d+)$", specifier)
        mtanh = re.match(r"t\.(\d+)$", specifier)
        msigmoid = re.match(r"s\.(\d+)$", specifier)

        allNone = (pstack is None and istack is None and ostack is None)
        someNone = (pstack is None or istack is None or ostack is None)
        
        assert (allNone or not someNone)

        getSizeOnly = allNone

        if maff:
            din = int(maff.group(1))
            dout = int(maff.group(2))
            psize = din*dout + dout
            isize = din * mbsize
            osize = dout * mbsize
            if getSizeOnly: return (isize, osize, psize)

            #assert x.size == (din*dout)+dout for x in [wslice, gslice, rslice, rgslice]

            # This is the conversion from a flat array to the
            # matrices that AffineLayer uses for its parameters.
            # Note that we are only slicing and reshaping, and because
            # of the nature of the reshaping, we never copy the array.

            # This is important because it means that there are now
            # two views of the same parameter vector: the external view
            # which MLP uses, which is a flat vector, and the
            # layer-specific view, which sees slices of this vector
            # in the shape of matrices.

            assert pstack.size == psize
            assert istack.size == isize
            assert ostack.size == osize
            
            wstack = pstack.subblob(0,din*dout, shape=(dout,din))
            bstack = pstack.subblob(din*dout, psize, shape=(dout,1))

            istack.reshape_all((din,mbsize))
            ostack.reshape_all((dout,mbsize))

            self.affinit.initialize(wstack.m, bstack.m)

            return layers.AffineLayer(wstack, bstack, istack, ostack, l2reg)
            
        elif mrelu:
            dim = int(mrelu.group(1))
            isize = dim * mbsize
            osize = dim * mbsize
            psize = 0
            if getSizeOnly: return (isize, osize, psize)

            assert pstack.size == psize
            assert istack.size == isize
            assert ostack.size == osize

            istack.reshape_all((dim,mbsize))
            ostack.reshape_all((dim,mbsize))
            
            return layers.ReluLayer(istack, ostack)

        elif mtanh:
            dim = int(mtanh.group(1))

            isize = dim * mbsize
            osize = dim * mbsize
            psize = 0
            if getSizeOnly: return (isize, osize, psize)

            assert pstack.size == psize
            assert istack.size == isize
            assert ostack.size == osize

            istack.reshape_all((dim,mbsize))
            ostack.reshape_all((dim,mbsize))
            
            return layers.TanhLayer(istack, ostack)

        elif msigmoid:
            dim = int(msigmoid.group(1))

            isize = dim * mbsize
            osize = dim * mbsize
            psize = 0
            if getSizeOnly: return (isize, osize, psize)

            assert pstack.size == psize
            assert istack.size == isize
            assert ostack.size == osize

            istack.reshape_all((dim,mbsize))
            ostack.reshape_all((dim,mbsize))
            
            return layers.SigmoidLayer(istack, ostack)

        else:
            sys.exit("Unknown layer: "+ specifier)
