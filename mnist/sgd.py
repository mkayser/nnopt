import numpy as np
import mlp


class SGD(object):
    def __init__(self, X, y, mb, lrstep, lrmult, lrinit):
        self.lr = lrinit
        self.lrmult = lrmult
        self.mb = mb
        self.lrstep = lrstep
        self.X = X
        self.y = y
        self.n = X.shape[1]
        assert X.shape[1] % mb == 0
        assert X.shape[1] == y.shape[1]
        self.start = 0
        
    def train(self, mlp, mbn):
        
        for i in xrange(mbn):
            end = self.start + mb  
            ## because always divisible, this won't break
            assert end <= n
        
            X_curr = X[:,self.start:end]
            y_curr = y[:,self.start:end]
            
            (_, data_loss, reg_loss) = mlp.fwd(X_curr, y_curr)
            mlp.bwd()
            grad = mlp.grad()

            wincr = [[e * lr for e in l] for l in grad]
            mlp.wadd(wincr, 1.0)

            self.start = end % n

            print "data_loss=",data_loss, " reg_loss=",reg_loss

        

        

