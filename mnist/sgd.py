import numpy as np
import mlp
import paramutils


class SGD(object):
    def __init__(self, X, y, mb, lrstep, lrmult, lrinit, max_grad_norm=10e5):
        self.lr = lrinit
        self.lrmult = lrmult
        self.mb = mb
        self.lrstep = lrstep
        self.X = X
        self.y = y
        self.n = X.shape[1]
        self.d = X.shape[0]
        yd = y.shape[0]
        assert self.n % mb == 0, "{} does not divide {}".format(mb, self.n)
        assert self.d == yd
        self.start = 0
        self.curriter = 0
        self.max_grad_norm = max_grad_norm
        
    def train(self, mlp, mbn):
        
        for i in xrange(mbn):
            end = self.start + self.mb  
            ## because always divisible, this won't break
            assert end <= self.n, "end, {} is not leq than set size, {}".format(end, self.n)
            X_curr = self.X[:,self.start:end]
            y_curr = self.y[:,self.start:end]
            
            (_, data_loss, reg_loss) = mlp.fwd(X_curr, y_curr)
            mlp.bwd()
            orig_grad = mlp.grad()

            (grad, clipped) = self.clip_grad(orig_grad)

            wincr = [[-e * self.lr for e in l] for l in grad]
            norm_wincr = paramutils.norm(wincr)
            norm_w     = paramutils.norm(mlp.w())
            mlp.wadd(wincr, 1.0)

            self.start = end % self.n
            self.curriter += 1
            if self.curriter % self.lrstep == 0:
                newlr = self.lr * self.lrmult
                #print "Reducing lr: {} => {}".format(self.lr, newlr)
                self.lr = newlr

            print "data_loss={:.2E}  reg_loss={:.2E}  lr={:.2E}  orig_grad={:.2E}  clip_grad={:.2E}  {}  stepsize={:.2E}  wnorm={:.2E}".format(data_loss, reg_loss, self.lr, paramutils.norm(orig_grad), paramutils.norm(grad), ("C" if clipped else "."), norm_wincr, norm_w)

    def clip_grad(self, grad):
        norm = paramutils.norm(grad)
        #print "gradnorm = ",norm
        clipped = False
        if norm > self.max_grad_norm:
            mult = self.max_grad_norm / norm
            grad = [[e * mult for e in l] for l in grad]
            clipped = True
            
        return (grad, clipped)


        

