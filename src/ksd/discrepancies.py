import numpy as np
from kernels import Kernel, InverseMultiquadricKernel
from tqdm import tqdm


class KSD():
    def __init__(self, name, kernel_params, p, q):
        self.kernel_params = kernel_params

        if name == 'imq':
            self.kernel = InverseMultiquadricKernel(name, kernel_params)
        else:
            raise NotImplementedError

        self.p = p # density of p (1d????)
        self.q = q # set of discrete support of unknown q
        # uniform weights for the discrete measure q
        self.weights = np.ones(len(self.q))/len(self.q)

    def h(self, x, y, wx, wy):
        
        log_px = self.p.grad_log_density(x)
        log_py = self.p.grad_log_density(y)

        kxy = self.kernel.stein_k(x, y, log_px, log_py)

        # returns a ndarray
        return wx * wy * kxy

    def ksd_1d(self, n, m=5):
        x_samples = self.q.sampler(n)
        y_samples = self.q.sampler(m)

        # before the sum, this is a 3d tensor
        stein_average = np.sum(
                        np.array([
                            np.array([
                                self.h(y_i,x_j) for y_i in y_samples]) 
                                                for x_j in tqdm(x_samples)]
                                                        )
                                                        )

        return np.sqrt(stein_average)


# Wasserstein for 1d distrbutions
class Wasserstein():
    pass
