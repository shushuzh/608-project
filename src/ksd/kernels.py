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
    def stein_k(self, x, y, log_px, log_py, wx, wy):
        
        p1 = self.k(x,y) * np.einsum("ik, jk -> ij", log_py, log_px)
        p2 = np.einsum("ijk, jk -> ij", self.grad_x(x,y), log_py)
        p3 = np.einsum("ijk, ik -> ij", self.grad_y(x,y), log_px)
        p4 = self.grad_xy(x,y).sum(2)

        h = p1 + p2 + p3 + p4

        return wx.T @ h @ wy



class InverseMultiquadricKernel(Kernel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.c = self.params['c']
        self.beta = self.params['beta']

        if self.c is None:
            self.c = 1

        if self.beta is None:
            self.beta = -0.5
        elif self.beta > 0:
            self.beta = -self.beta
    
    # we take gradient manually to vectorise over the samples
    def grad_x(self, x, y):
        r = ((x - y[:,np.newaxis])**2).sum(2, keepdims=True)
        dx = 2 * self.beta * (x - y[:,np.newaxis]) * (self.c**2 + r)**(self.beta - 1.)
        return dx.transpose((1,0,2))

    def grad_y(self, x, y):
        r = ((x - y[:,np.newaxis])**2).sum(2, keepdims=True)
        dy = -2 * self.beta * (x - y[:,np.newaxis]) * (self.c**2 + r)**(self.beta - 1.)
        return dy.transpose((1,0,2))

    def grad_xy(self, x, y):
        r = ((x - y[:,np.newaxis])**2).sum(2, keepdims=True)

        _y = -2 * self.beta * (self.c**2 + r)**(self.beta - 1.)
        _xy = (-4 * self.beta * (self.beta - 1) * (x - y[:, np.newaxis])**2 
                * (self.c**2 + r)**(self.beta - 2.))
        dxy = (_y + _xy).transpose((1,0,2))

        # so along the first axis, (xi, yj)
        return dxy

    def k(self, x, y):

        k = (self.c**2 + ((x - y[:,np.newaxis])**2).sum(2))**self.beta

        return k.transpose((1,0))

    

# class InverseMultiquadricKernel_pre(Kernel):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
#         self.c = self.params['c']
#         self.beta = self.params['beta']

#         if self.c is None:
#             self.c = 1

#         if self.beta is None:
#             self.beta = 0.5
#         elif self.beta < 0:
#             self.beta = -self.beta


#     def k(self, x, y):
#         return 1./(self.c**2 + ((x - y)**2).sum()) ** self.beta

#     # this is d dimensional if p has dimension d
#     def stein_k_pre(self, x, y, log_px, log_py):
        
#         log_px = log_px[0]
#         log_py = log_py[0]
        
#         p1 = self.k(x,y) * np.dot(log_py, log_px)
#         p2 = np.dot(self.grad_kx(x,y), log_py)
#         p3 = np.dot(self.grad_ky(x,y), log_px)
#         p4 = np.sum(np.diag(self.grad_kxy(x,y)))

#         return p1, p2, p3, p4


if __name__ == '__main__':
    from example_Gaussian import Gaussian
    mu = np.array([1.2, .6])
    cov = (
        0.9*(np.ones([2, 2]) -
             np.eye(2)).T +
             np.eye(2)*1.3
    )
    
    n = 10
    c = 1
    beta = 0.5
    q = Gaussian({'mu': mu, 'sigma':cov})
    q_dat = q.sampler(n)

    def k1(x,y): 
        return (c**2 + ((x - y[:,np.newaxis])**2).sum(2)) ** (-beta)

    def k2(x,y):
        kxy = []
        for i in range(n):
            for j in range(n):
                kxy.append(1. / (c**2 + ((x[i] - y[j])**2).sum()) ** beta)

        return np.array(kxy).reshape((n,n))

    print(k1(q_dat, q_dat))
    print(k2(q_dat, q_dat))