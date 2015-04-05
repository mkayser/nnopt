import numpy as np


class Point(object):
    def __init__(self, elapsedSec, samplesSeen, score):
        self.elapsedSec = elapsedSec
        self.samplesSeen = samplesSeen
        self.score = score


