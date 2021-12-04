import torch
from langevin_sampling.samplers import LangevinDynamics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import copy
from tqdm import tqdm
from  scipy.stats import multivariate_normal
import tensorflow as tf
import tensorflow_probability as tfp
np.random.seed(19)
torch.manual_seed(19)

if not torch.cuda.is_available():
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')
else:
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

class GaussianDistribution(object):
    def __init__(self, mu, cov, device='cuda'):
        super(GaussianDistribution, self).__init__()
        # where is the super class from?
        self.mu = mu
        self.cov = cov
        self.precision = torch.inverse(cov)

        self.R = torch.linalg.cholesky(self.cov)
        self.normal = torch.distributions.normal.Normal(torch.zeros_like(mu),
            torch.ones_like(mu))

    def nl_pdf(self, x):
        return 0.5*(
            ((x - self.mu).T).matmul(self.precision)).matmul(x - self.mu)

    def sample(self):
        return self.R.matmul(self.normal.sample()) + self.mu


if __name__ == '__main__':

    dim = 2

    mu = torch.Tensor([1.2, .6], device=device)
    cov = (
        0.9*(torch.ones([2, 2], device=device) -
             torch.eye(2, device=device)).T +
        torch.eye(2, device=device)*1.3
    )
    gaussian_dist = GaussianDistribution(mu, cov, device=device)
    
    # tensorflow distrbution
    tfd = tfp.distributions
    m = mu.numpy()
    sigma = cov.numpy()
    gaussian = tfd.MultivariateNormalFullCovariance(
               loc=m,
               covariance_matrix=sigma)   

    # contour plot
    N = 300
    X = np.linspace(-2, 7, N)
    Y = np.linspace(-3, 6, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mu, cov)
    Z = rv.pdf(pos)
    
    est_samples = dict()
    max_itr = int(1e4)
    burn_in = 500

    x = torch.zeros([2], requires_grad=True, device=device)
    langevin_dynamics = LangevinDynamics(
        x,
        gaussian_dist.nl_pdf,
        lr=0.2,
        lr_final=4e-2,
        max_itr=max_itr+burn_in,
        device=device
    )

    hist_samples_sgula = []
    loss_log_sgula = []
    for j in tqdm(range(max_itr)):
        est, loss = langevin_dynamics.sample()
        loss_log_sgula.append(loss)
        # if j%3 == 0:
        # without thinning -- same setup as MCMC
        hist_samples_sgula.append(est.cpu().numpy())
    est_samples_sgula = np.array(hist_samples_sgula[burn_in:])[200:]

    num_samples_sgula = est_samples_sgula.shape[0]
    true_samples_sgula = np.zeros([num_samples_sgula, 2])
    for j in range(num_samples_sgula):
        true_samples_sgula[j, :] = gaussian_dist.sample().cpu().numpy()
    est_samples['sgula'] = est_samples_sgula

    fig = plt.figure("training logs - net", dpi=150, figsize=(7, 2.5))
    plt.plot(loss_log_sgula)
    plt.title("Unnormalized PDF")
    plt.xlabel("Iterations")
    plt.ylabel(r"$- \log \ \mathrm{N}(\mathbf{x} | \mu, \Sigma) + const.$")
    plt.grid()
    plt.savefig('img/Gaussian_SG_ULA_pdf.png')

    fig = plt.figure(dpi=150, figsize=(9, 4))
    plt.subplot(121)
    plt.contour(X, Y, Z, alpha=0.5)
    plt.scatter(est_samples_sgula[:, 0], est_samples_sgula[:, 1], s=.5,
                color="#db76bf")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.xlim([-3, 6])
    plt.ylim([-4, 5])
    plt.title("Langevin dynamics")

    plt.subplot(122)
    plt.contour(X, Y, Z, alpha=0.5)
    p2 = plt.scatter(true_samples_sgula[:, 0], true_samples_sgula[:, 1], s=.5,
                     color="#5e838f")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.xlim([-3, 6])
    plt.ylim([-4, 5])
    plt.title(r"$\mathbf{x} \sim \mathrm{N}(\mu, \Sigma)$")
    plt.tight_layout()
    plt.savefig('img/Gaussian_SG_ULA_LD.png')