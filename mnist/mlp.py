import numpy as np
import layers
import sys
import re
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
        w[...] = self.wstd * np.random.standard_normal(w.shape)
        b[...] = self.bstd * np.random.standard_normal(b.shape)

        

# This is a little quirky because the autoencoder needs to know about minibatch size,
# which is a training-specific piece of information. Could change this later so that
# the trainer initializes and owns the input/output blobs.
class MLPAutoencoder(object):
    def __init__(self, specifier, mbsize, l2reg, affinit):
        tokens = specifier.split('_')
        self.layers = []
        self.layerSizes = []
        self.affinit = affinit
        self.ready_to_backprop = False

        # Step through layer specifications and get parameter
        # sizes
        totalPSize = 0
        totalIOSize = 0
        for i,tok in enumerate(tokens):
            (isize, osize, psize) = self.makeLayer(tok, l2reg, mbsize, None, None, None)
            self.layerSizes.append((isize, osize, psize))
            totalPSize += psize
            if i==0: totalIOSize += isize
            totalIOSize += osize
        
        # Allocate underlying arrays
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
            pstart += psize

            islice = self.iostack.subblob(iostart, iostart+isize)
            iostart += isize
            oslice = self.iostack.subblob(iostart, iostart+osize)

            layer = self.makeLayer(tok, l2reg, mbsize, pslice, islice, oslice)
            self.layers.append(layer)

            if i==0:
                istack = layer.get_istack()
                self.X = istack.val
            if i==len(self.layerSizes)-1:
                ostack = layer.get_ostack()
                self.ypred = ostack.val

        # Note that this field represents all parameters in a flat vector
        self.w = self.paramstack.val

        # Likewise, this field represents the gradient as a flat vector
        self.g = self.paramstack.g

        # This field represents the vector "v" in the product Hv, again as a flat vector
        self.v = self.paramstack.gtval

        # This field represents the result of a Hessian-vector product
        self.Hv = self.paramstack.Hv

        # For each minibatch we will assign to self.y, so for now it's None
        self.y = None

        self.l2reg = l2reg

    def set_v(self, v):
        assert v.shape == self.v.shape
        self.v[...] = v

    def get_g(self):
        return self.g

    def get_Hv(self):
        return self.Hv

    def get_w(self):
        return self.w

    def get_v(self):
        return self.v

    def set_w(self, w):
        assert w.shape == self.w.shape
        self.w[...] = w

    def set_X_y(self, X, y):
        assert X.shape == self.X.shape
        assert y.shape == self.ypred.shape

        # Note that self.y is one of the only cases where we assign a reference rather
        # than populating in-place.
        self.X[...] = X
        self.y = y
        self.ypred[...] = 0

    def get_ypred(self):
        return self.ypred
        
    # TODO eventaully generalize to softmax/crossentropy
    def prop_data_loss(self, do_Hv=False, do_Gv=False):

        go  = self.layers[-1].get_ostack().g
        Hvo = self.layers[-1].get_ostack().Hv
        Gvo = self.layers[-1].get_ostack().Gv

        ro   = self.layers[-1].get_ostack().gtval

        y = self.y
        ypred = self.ypred

        diff = ypred-y
        n = diff.shape[1]

        loss = np.sum(.5 * (diff**2)) / n

        go[...]  = diff / n

        if do_Hv:
            Hvo[...] = ro / n
        if do_Gv:
            assert False, "Gaussian-vector products not implemented yet."
            pass

        return loss

    def fwd(self, do_Hv=False, do_Gv=False):
        reg_loss = 0
        data_loss = None

        do_gtval = do_Hv or do_Gv

        for i,l in enumerate(self.layers):
            l.fwd(do_val=True, do_gtval=do_gtval)
            reg_loss += l.l2reg_loss()

        if self.y is not None:
            self.ready_to_backprop = True
            data_loss = self.prop_data_loss(do_Hv=do_Hv, do_Gv=do_Gv)

        else:
            self.ready_to_backprop = False
            reg_loss = None

        return (data_loss, reg_loss)

    def bwd(self, do_Hv=False, do_Gv=False):
        assert self.ready_to_backprop
        for i,l in reversed(list(enumerate(self.layers))):
            l.bwd(do_g=True, do_Hv=do_Hv, do_Gv=do_Gv)
        
    def w_debug(self):
        wlist = []
        for l in self.layers:
            wlist.append(l.w_debug())
        return wlist

    def makeLayer(self, specifier, l2reg, mbsize, pstack, istack, ostack):
        #print "Layer spec: ",specifier
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

            self.affinit.initialize(wstack.val, bstack.val)

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
