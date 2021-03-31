import numpy as np
from numpy import random as rd


class LFM:
    def __init__(class_count):
        self.user_num = 6040
        self.movie_num = 3952
        self.class_count = class_count
        self.lamda1 = 0.01
        self.lamda2 = 0.01
        self.p = rd.random((self.user_num, self.class_count))
        self.q = rd.random((self.movie_num, self.class_count))
    
    def forward(i, j):
        pred_rank = np.dot(p[i], q[j])
        return pred_rank
    
    def omega():
        omega = 0.
        for i in range(self.p.shape[0]):
            omega += np.linalg.norm(self.p[i])
        return omega