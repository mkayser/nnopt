import numpy as np
import time


class StatCollector(object):
    def __init__(self, vmodel):
        self.vmodel = vmodel
        self.stats = []
        self.start_time = None
        self.elapsed_time_ = 0.0

    def start_timer(self):
        assert self.start_time is None
        self.start_time = time.clock()

    def add(self, w, i, trainf):
        assert self.start_time is not None
        now = time.clock()
        self.elapsed_time_ += now - self.start_time

        (valf, _) = self.vmodel.f_g(w)
        self.stats += [i,self.elapsed_time_,trainf,valf]

        self.start_time = time.clock()

    def elapsed_time(self):
        assert self.start_time is not None
        return self.elapsed_time_

    def retrieve(self):
        return np.reshape(self.stats, newshape=(len(self.stats)/4, 4))

        

class Point(object):
    def __init__(self, elapsedSec, samplesSeen, score):
        self.elapsedSec = elapsedSec
        self.samplesSeen = samplesSeen
        self.score = score


