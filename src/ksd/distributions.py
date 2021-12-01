import torch
from torch.autograd import Variable

import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt

torch.manual_seed(19)

print(torch.cuda.is_available())
if not torch.cuda.is_available():
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')
else:
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

"""
Gaussian Distribution
"""

class Gaussian:

    def __init__(self, params):
        self.mu = params['mu']
        self.sigma = params['sigma']

    def density(self, x, mu=None, sigma=None):
        if mu == None: mu = self.mu
        if sigma == None: sigma = self.sigma

        p_1 = (((2 * np.pi)**(len(mu)/2)) * (np.linalg.det(sigma))**0.5)
        p_2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(sigma))).dot((x-mu))

        return np.exp(p_2)/p_1

    def log_density(self, x):
        return np.log(self.density(x))

    # def grad_log_density(self, x, z_optim=False):
    #     dtype = torch.FloatTensor

    #     mu = Variable(torch.Tensor(self.mu).type(dtype), requires_grad=z_optim)
    #     sigma = Variable(torch.Tensor(self.sigma).type(dtype), requires_grad=False)
    #     x = Variable(torch.Tensor(x).type(dtype), requires_grad=True)

    #     y = (-1/2) * torch.dot(x - mu, torch.inverse(sigma).mv(x - mu))
    #     y.backward()

    #     if z_optim:
    #         return dict(x_grad=x.grad, mu_grad=mu.grad)

    #     return x.grad.data.numpy()
    
    def grad_log_density(self, x):
        mu = self.mu
        inv_sigma = np.linalg.inv(self.sigma)

        return - np.einsum("jk, ij -> ik", inv_sigma, (x - mu[np.newaxis,:]))

        ###### check this grad density

    def sampler(self, N, mu=None, sigma=None):
        if mu == None: mu = self.mu
        if sigma == None: sigma = self.sigma

        # m, n = sigma.size
        # l = torch.from_numpy(np.linalg.cholesky(sigma))
        # x = torch.randn(n,)

        # return multivariate_normal.rvs(mean=mu, cov=sigma, size=N)
        g = torch.distributions.multivariate_normal.MultivariateNormal(torch.Tensor(mu), torch.Tensor(sigma))
        return g.sample(torch.Size([N])).numpy()



if __name__ == '__main__':

    mu = torch.Tensor([1.2, .6], device=device)
    cov = (
        0.9*(torch.ones([2, 2], device=device) -
             torch.eye(2, device=device)).T +
        torch.eye(2, device=device)*1.3
    )
    
    g = Gaussian({'mu': mu, 'sigma':cov})

    x = np.arange(10).reshape((5,2))
    x = np.vstack([x, [0, 0]])
    # grad_autodiff = g.grad_log_density(x)
    grad_manual = g.grad_log_density(x)

    # print(grad_autodiff)
    print(grad_manual)




