import sys
import os

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from ksd.kernels import InverseMultiquadricKernel, InverseMultiquadricKernel_pre

import numpy as np
from tqdm import tqdm
# from itertools import partial


class KSD:
    def __init__(self, name, kernel_params, p, q):
        self.kernel_params = kernel_params

        if name == 'imq':
            self.kernel = InverseMultiquadricKernel(name, kernel_params)
        # elif name == 'imq_pre':
        #     self.kernel = InverseMultiquadricKernel_pre(name, kernel_params)
        else:
            raise NotImplementedError

        self.p = p # density of p 
        self.q = q # set of discrete support of unknown q

        # uniform weights for the discrete measure q
        self.weights = np.ones(len(self.q))/len(self.q)

    # inputs are all 2d arraies
    def discrepancy(self):
        # we can do this since the form of log density is known
        log_px = self.p.grad_log_density(self.q)
        log_py = log_px

        kxy = self.kernel.stein_k(
                                self.q, self.q, 
                                log_px, log_py, 
                                self.weights, self.weights
                                )
        return kxy

    # def h(self, x, y, wx, wy):
    #     # we can do this since the form of log density is known
    #     log_px = self.p.grad_log_density(x)
    #     log_py = self.p.grad_log_density(y)

    #     p1, p2, p3, p4 = self.kernel.stein_k_pre(x, y, log_px, log_py)

    #     kxy = p1 + p2 + p3 + p4

    #     # returns a ndarray
    #     return wx * wy * kxy

    # def discrepancy_pre(self):
    #     # x_samples = self.q.sampler(n)
    #     # y_samples = self.q.sampler(m)

    #     # before the sum, this is a 3d tensor
    #     stein_average = np.array([
    #                         np.array([
    #                             self.h(self.q[i], self.q[j], self.weights[i], self.weights[j]) 
    #                                             for i in range(len(self.q))]) 
    #                                             for j in tqdm(range(len(self.q)))]
    #                                                     )
                                                        

    #     return np.sum(stein_average)



# Wasserstein for 1d distrbutions
class Wasserstein:
    pass



    
