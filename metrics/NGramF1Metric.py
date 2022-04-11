from utilities.utilities import NGramF
import numpy as np

class NGramF1Metric:

    def __init__(self, n):
        self.n = n
        self.util = NGramF(n)


        self.data = []



    def add(self, src, hyp, target):
        self.data.append({"src": src, "hyp": hyp, "target": target})

    def compute(self):

        scores = [self.util(d["src"], d["hyp"], d["target"]) for d in self.data]
        m = np.mean(scores)
        return m
