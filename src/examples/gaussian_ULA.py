import torch
from langevin_sampling.samplers import LangevinDynamics
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
from scipy.stats import multivariate_normal
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
    
    est_samples = dict()
    max_itr = int(1e3*2)
    burn_in = 500
    
    # ULA
    ## Define ULA sampler with `step_size` equal to 0.75
    samples_ula = tfp.mcmc.sample_chain(
        num_results=max_itr,
        current_state=tf.zeros(2),
        kernel=tfp.mcmc.UncalibratedLangevin(
            target_log_prob_fn=gaussian.log_prob,
            step_size=0.5)
        # ,num_steps_between_results=3
        ,num_burnin_steps=burn_in
        ,trace_fn=None
    )

    est_samples_ula = np.array(samples_ula)[200:]
    loss_log_ula = np.apply_along_axis(gaussian.log_prob, 1, samples_ula.numpy())

    num_samples_ula = est_samples_ula.shape[0]
    true_samples_ula = np.zeros([num_samples_ula, len(mu)])
    for j in range(num_samples_ula):
        true_samples_ula[j, :] = gaussian_dist.sample().cpu().numpy()
    est_samples['ula'] = est_samples_ula

    # contour plot
    N = 300
    X = np.linspace(-2, 7, N)
    Y = np.linspace(-3, 6, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mu, cov)
    Z = rv.pdf(pos)

    fig = plt.figure("training logs - net", dpi=150, figsize=(7, 2.5))
    plt.plot(-loss_log_ula)
    plt.title("Unnormalized PDF")
    plt.xlabel("Iterations")
    plt.ylabel(r"$- \log \ \mathrm{N}(\mathbf{x} | \mu, \Sigma) + const.$")
    plt.grid()
    plt.savefig('img/Gaussian_ULA_pdf.png')

    fig = plt.figure(dpi=150, figsize=(9, 4))
    plt.subplot(121)
    plt.contour(X, Y, Z, alpha=0.5)
    plt.scatter(est_samples_ula[:, 0], est_samples_ula[:, 1], s=.5,
                color="#db76bf")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.xlim([-3, 6])
    plt.ylim([-4, 5])
    plt.title("Langevin dynamics")
    plt.subplot(122)
    plt.contour(X, Y, Z, alpha=0.5)
    p2 = plt.scatter(true_samples_ula[:, 0], true_samples_ula[:, 1], s=.5,
                     color="#5e838f")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.xlim([-3, 6])
    plt.ylim([-4, 5])
    plt.title(r"$\mathbf{x} \sim \mathrm{N}(\mu, \Sigma)$")
    plt.tight_layout()
    plt.savefig('img/Gaussian_ULA_LD.png')