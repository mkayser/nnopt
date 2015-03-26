import numpy as np


class BlobStack(object):
    def __init__(self, m):
        self.m = m
        # w, g*, gtval, Hv*, Gv*
        assert m.shape[0] == 5
        self.val = m[0]
        self.g = m[1]
        self.gtval = m[2]
        self.Hv = m[3]
        self.Gv = m[4]
        
    def subblob(self, start, end, shape=None):
        result = BlobStack(self.m[:,start:end])
        if shape is not None:
            result.reshape_all(shape)
        return result
        
    def reshape_all(self, shape):
        self.val   = self.val.reshape(shape)
        self.g     = self.g.reshape(shape)
        self.gtval = self.gtval.reshape(shape)
        self.Hv    = self.Hv.reshape(shape)
        self.Gv    = self.Gv.reshape(shape)
