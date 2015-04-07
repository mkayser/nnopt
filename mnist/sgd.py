import numpy as np
import mlp
import paramutils
import time


class SGD(object):
    def __init__(self, X, y, mb, lrstep, lrmult, lrinit, max_grad_norm=10e5, statc=None):
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
        self.statc=statc
        
    def train(self, mlp, mbn):

        wdelta = np.zeros_like(mlp.get_w())

        samples_seen=0

        for i in xrange(mbn):
            end = self.start + self.mb  
            ## because always divisible, this won't break
            assert end <= self.n, "end, {} is not leq than set size, {}".format(end, self.n)
            X_curr = self.X[:,self.start:end]
            y_curr = self.y[:,self.start:end]

            samples_seen += self.mb

            mlp.set_X_y(X_curr, y_curr)
            
            # Fwd and Bwd passes
            (data_loss, reg_loss) = mlp.fwd(do_Hv=False, do_Gv=False)
            mlp.bwd(do_Hv=False, do_Gv=False)
                        
            grad = mlp.get_g()
            w = mlp.get_w()

            wdelta[...] = -grad * self.lr

            # Compute some stats for reporting
            norm_wdelta = np.linalg.norm(wdelta)
            norm_w     = np.linalg.norm(w)
            wmin       = np.min(w)
            wmax       = np.max(w)
            
            #if self.curriter % 10 == 1: 
            #    print "data_loss={:.2E}  reg_loss={:.2E}  lr={:.2E}  w={}  -grad={}".format(data_loss, reg_loss, self.lr, w, -grad)
            if self.curriter % 10 == 0: 
                self.statc.add(w, samples_seen, data_loss+reg_loss)
                elapsed_time = self.statc.elapsed_time()
                print "it={}  data_loss={:.2E}  reg_loss={:.2E}  elapsed={:.3}s  lr={:.2E}  orig_grad={:.2E}  clip_grad={:.2E}  {}  stepsize={:.2E}  wnorm={:.2E}  wmin,max={:.2E},{:.2E}".format(self.curriter, data_loss, reg_loss, elapsed_time, self.lr, np.linalg.norm(grad), np.linalg.norm(grad), ("C" if False else "."), norm_wdelta, norm_w, wmin, wmax)

            # Update w in place
            w[...] += wdelta

            # Update iter, data start index, and learning rate
            self.start = end % self.n
            self.curriter += 1
            if self.curriter % self.lrstep == 0:
                newlr = self.lr * self.lrmult
                self.lr = newlr


    def clip_grad(self, grad):
        norm = paramutils.norm(grad)
        #print "gradnorm = ",norm
        clipped = False
        if norm > self.max_grad_norm:
            mult = self.max_grad_norm / norm
            grad = [[e * mult for e in l] for l in grad]
            clipped = True
            
        return (grad, clipped)


        

