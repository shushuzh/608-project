import torch
from langevin_sampling.samplers import LangevinDynamics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import copy
from tqdm import tqdm
from  scipy.stats import multivariate_normal
import scipy.stats as ss
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
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
#         if (cov.shape[0]>0):
#             self.precision = torch.inverse(cov)
#         else:
        self.precision = 1/cov

        #self.R = torch.linalg.cholesky(self.cov)
        self.R = np.sqrt(self.cov)
        self.normal = torch.distributions.normal.Normal(torch.zeros_like(mu),
            torch.ones_like(mu))

    def nl_pdf(self, x):
        return 0.5*((x - self.mu)*self.precision*(x - self.mu))

    def sample(self):
        return self.R*self.normal.sample() + self.mu
    
class MixGaussianDistribution(object):
    def __init__(self, mu, cov, pi, device='cuda'):
        """
        Initializes the model and brings all tensors into their required shape.
        The class owns:
            mu:              mean of the Gaussians, k dimension
            var:             sigma^2 of the Gaussians, k dimension
            pi:              weight, k dimension
        """
        super(MixGaussianDistribution, self).__init__()
        self.mu = mu
        self.cov = cov
        self.pi = pi #weights
        self.K = mu.shape[0] #number of Gaussians

    def nl_pdf(self, x):
        pdf = 0
        for i in range(self.K):
            pdf += self.pi[i]*GaussianDistribution(self.mu[i],self.cov[i]).nl_pdf(x[i])
        return pdf

    def sample(self):
        sample = 0
        for i in range(self.K):
            sample += self.pi[i]*GaussianDistribution(self.mu[i],self.cov[i]).sample()
        return sample


if __name__ == '__main__':

    dim = 1

    mu = torch.Tensor([1, 5], device=device)
    cov = torch.Tensor([1, 1], device=device)
    pi = 1/2*torch.ones(2)
    mix_gaussian_dist = MixGaussianDistribution(mu, cov, pi, device=device)

    # contour plot
    N = 300
    X = np.linspace(-2, 7, N)
    Y = np.linspace(-3, 6, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mu, cov)
    Z = rv.pdf(pos)
    
    # tensorflow distrbution
    tfd = tfp.distributions
    m = mu.numpy()
    sigma = cov.numpy()
    p = pi.numpy()
    mix_gaussian = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=p),
    components_distribution=tfd.Normal(
      loc=m,
      scale=sigma))
    
    est_samples = dict()
    max_itr = int(1e3)
    burn_in = 500

    # ULA
    ## Define ULA sampler with `step_size` equal to 0.75
    samples_ula = tfp.mcmc.sample_chain(
        num_results=max_itr,
        current_state=tf.zeros(2),
        kernel=tfp.mcmc.UncalibratedLangevin(
            target_log_prob_fn=mix_gaussian.log_prob,
            step_size=0.5)
        # ,num_steps_between_results=3
        ,num_burnin_steps=burn_in
        ,trace_fn=None
    )
    
    est_samples_ula = np.array(samples_ula)[200:]
    loss_log_ula = np.apply_along_axis(mix_gaussian.log_prob, 1, samples_ula.numpy())

    num_samples_ula = est_samples_ula.shape[0]
    true_samples_ula = np.zeros([num_samples_ula, len(mu)])
    for j in range(num_samples_ula):
        true_samples_ula[j, :] = mix_gaussian_dist.sample().cpu().numpy()
    est_samples['ula'] = est_samples_ula

        
    #est_samples_ula_mix = pd.DataFrame(np.matmul(est_samples_ula,pi.T))
    #true_samples_ula_mix = pd.DataFrame(np.matmul(true_samples_ula,pi.T))
    #est_samples['ula'] = est_samples_ula
    
    # Theoretical PDF plotting -- generate the x and y plotting positions
    #xs = np.linspace(est_samples_ula_mix.min(), est_samples_ula_mix.max(), 200)
    xs = np.linspace(mu[0]-2,mu[1]+2, 200)
    ys = np.zeros_like(xs)
    # y1 = ss.norm.pdf(xs, loc=mu[0], scale=cov[0])
    # y2 = ss.norm.pdf(xs, loc=mu[1], scale=cov[1])
    # ys = pi[0].numpy()*y1 + pi[1].numpy()*y2
    for i in range(mu.size(dim=0)):
        ys += ss.norm.pdf(xs, loc=mu[i], scale=cov[i]) * pi[i].numpy()

    df = pd.DataFrame(np.array(est_samples_ula).flatten(), columns = ['sample'] ) #Converting array to pandas DataFrame
    ax = df.plot(kind = 'density')
    #plt.plot(xs,y1,color='green',label="N(1,1)")
    #plt.plot(xs,y2,color='yellow',label="N(5,1)")
    plt.plot(xs,ys,color = 'red',label="mixed")
    plt.title(f"Samples and True PDF for {pi[0]} N({mu[0]}, {cov[0]})+{pi[1]} N({mu[1]}, {cov[1]})")
    plt.xlabel(r"$x$")
    plt.ylabel("PDF")
    plt.grid()
    plt.savefig('img/mix_Gaussian_ULA_pdf_TrueSample.png')

    fig = plt.figure(dpi=150, figsize=(9, 4))
    plt.plot(loss_log_ula)
    plt.title("Unnormalized PDF")
    plt.xlabel("Iterations")
    plt.ylabel(r"$- \log \ \mathrm{N}(\mathbf{x} | \mu, \Sigma) + const.$")
    plt.grid()
    plt.savefig('img/mix_Gaussian_ULA_pdf_it.png')

    fig = plt.figure(dpi=150, figsize=(9, 4))
    #plt.subplot(121)
    #plt.contour(X, Y, Z, alpha=0.5)
    #est_samples_ula_mix.plot.kde(color="#db76bf")
    plt.scatter(est_samples_ula[:,0],est_samples_ula[:,1], s=.5,
                     color="#db76bf")
    #plt.contour(X, Y, Z, alpha=0.5)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.xlim([0, range(est_samples_ula_mix.size)])
    #plt.ylim([-4, 5])
    plt.title("Langevin dynamics")
    plt.savefig('img/mix_Gaussian_ULA_LD.png')

    # plt.subplot(122)

    # p2 = plt.scatter(range(true_samples_ula.size),true_samples_ula, s=.5,
    #                  color="#5e838f")
    # plt.xlabel("Iterations")
    # plt.ylabel(r"$1/2*x_1+1/2*x_2$")
    # #plt.xlim([0, range(true_samples_ula_mix.size)])
    # #plt.ylim([-4, 5])
    # plt.title(r"$\mathbf{x} \sim 1/2\mathrm{N}(1, 1)+1/2\mathrm{N}(2, 1)$")
    #plt.tight_layout()