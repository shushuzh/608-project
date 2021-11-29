# import torch
# from torch.autograd import Variable

from autograd import grad, jacobian
import autograd.numpy as np

class Kernel():
    def __init__(self, name, params):
        self.name = name
        self.params = params

    def k(self, x, y):
        raise NotImplementedError

    # def rbf(self, x, y, beta=0.01):
    #     return np.exp(- ((x - y)**2).sum() * beta)

    # def imq(self, x, y, c=1, beta=-1):
    #     return (c**2 + ((x - y)**2).sum())**(-beta)

    # def polynomial(self, x, y, c=1, degree=2):
    #     return (c + np.dot(x,y))**degree

    # here we fix y and take a derivative wrt x
    def grad_kx(self, x, y):
        k_x = lambda x_: self.k(x_,y)
        return grad(k_x)(x)
        

    # here we fix x and take a derivative wrt y
    def grad_ky(self, x, y):
        k_y = lambda y_: self.k(x,y_)
        return grad(k_y)(y)

    # here we take the derivative of k wrt x and y
    def grad_kxy(self, x, y):
        kx_y = lambda y_: self.grad_kx(x, y_)
        return jacobian(kx_y)(y)

    # this is d dimensional if p has dimension d
    def stein_k(self, x, y, log_px, log_py):

        p1 = self.k(x,y) * np.dot(log_py, log_px)
        p2 = np.dot(self.grad_kx(x,y), log_py)
        p3 = np.dot(self.grad_ky(x,y), log_px)
        p4 = np.sum(self.grad_kxy(x,y))

        return p1 + p2 + p3+ p4

class InverseMultiquadricKernel(Kernel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def k(self, x, y):
        c = self.params['c']
        beta = self.params['beta']

        if c is None:
            c = 1
        if beta is None:
            beta = -0.5

        return (c**2 + ((x - y)**2).sum()) ** beta

