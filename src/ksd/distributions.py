import torch
from torch.autograd import Variable
import torch.distributions as D

import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

# for checking
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import matplotlib.pyplot as plt

# torch.manual_seed(19)


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
        self.g = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.Tensor(self.mu), 
            torch.Tensor(self.sigma))

    def density(self, x):
        # sigma = np.diag(self.sigma[0]) 
        # mu = self.mu[np.newaxis,:]
        # print(x.shape)
        # print(mu.shape)
        # print(sigma.shape)
        sigma = self.sigma
        mu = self.mu

        p_1 = (((2 * np.pi)**(mu.shape[0]/2)) * (np.linalg.det(sigma))**0.5)
        # p_2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(sigma))).dot((x-mu))
        p_2 = np.einsum("li, ij -> lj", (x - mu)[np.newaxis,:], np.linalg.inv(sigma)) * (x - mu)[np.newaxis,:]

        return np.exp(-1/2 * p_2.sum(1, keepdims=True))/p_1

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

    def sampler(self, N):
        # m, n = sigma.size
        # l = torch.from_numpy(np.linalg.cholesky(sigma))
        # x = torch.randn(n,)

        # return multivariate_normal.rvs(mean=mu, cov=sigma, size=N)
        # g = torch.distributions.multivariate_normal.MultivariateNormal(torch.Tensor(mu), torch.Tensor(sigma))
        return self.g.sample(torch.Size([N])).numpy()

   
class GaussianMixture:
    """This class subsumes the Gaussian distrbution class
    """
    def __init__(self, params):

        # assert sigma.numpy().shape[:-1] == mu.numpy().shape[:-1]
        self.mu = params['mu']
        self.sigma = params['sigma']
        self.pi = params['pi'] #weights
 
        self.k = int(self.mu.shape[0]) #number of Gaussians
        self.d = int(self.mu.shape[1])

        # create tf object gaussian mixture
        _tf_mix =  tfd.Categorical(self.pi)
        _tf_comp = tfd.MultivariateNormalTriL(
                loc=self.mu,
                scale_tril=tf.linalg.cholesky(self.sigma)
            )

        self._tf_gmm = tfd.MixtureSameFamily(
            mixture_distribution=_tf_mix,
            components_distribution=_tf_comp
        )

        # create torch object gaussian mixture
        _tc_mix =  D.Categorical(torch.Tensor(self.pi))
        _tc_comp = D.MultivariateNormal(
            loc=torch.Tensor(self.mu), 
            scale_tril=torch.linalg.cholesky(torch.Tensor(self.sigma)))
        self._tc_gmm = D.MixtureSameFamily(_tc_mix, _tc_comp)


    def sample(self, N=1, _tf=False, _tc=False):
        # if mu == None: mu = self.mu
        # if sigma == None: sigma = self.sigma
        if _tf:
            return self._tf_gmm.sample(N).numpy()
        elif _tc:
            return self._tc_gmm.sample(torch.Size([N])).numpy()
        else:
            raise NameError('Specify either _tf=True, or _tc=True to use the sampler.')
        
    def _tf_density(self, x):
        return tf.math.exp(self._tf_gmm.log_prob(x))
    
    def _tc_density(self, x):
        _x = torch.Tensor(x)
        return self._tc_gmm.log_prob(_x).exp()
        
    def _density(self, x):
        """This returns a list of density for components of the mixture
        """
        pdf = []
        for i in range(self.k):
            # so we cannot implement mixture MVN in torch yet
            sigma_i = self.sigma[i]
            mu_i = self.mu[i][np.newaxis,:]
            # print(x.shape)
            # print(mu_i.shape)
            p_1 = (((2 * np.pi)**(self.d/2)) * (np.linalg.det(sigma_i))**0.5)
            # p_2 = -1/2 * ((x-mu_i).T @ np.linalg.inv(sigma_i)) @ (x-mu_i)

            p_2 = np.einsum("li, ij -> lj", (x - mu_i), np.linalg.inv(sigma_i)) * (x - mu_i)
            pdf.append(self.pi[i] * np.exp(-1/2 * p_2.sum(1, keepdims=True))/p_1)
        # print(pdf[1])
        return pdf

    def density(self, x, _tf=False, _tc=False):
        """Returns corresponding tensor or numpy array
        """
        if _tf:
            return self._tf_density(x)
        elif _tc:
            return self._tc_density(x)
        else:
            return sum(self._density(x))

    def log_density(self, x, _tf=False, _tc=False):
        """Returns corresponding tensor or numpy array
        """
        if _tf:
            return self._tf_gmm.log_prob(x)
        elif _tc:
            # _x = torch.Tensor(x, requires_grad=True, device=device)
            return self._tc_gmm.log_prob(x)
        else:
            return np.log(self.density(x))
    
    def neg_log_density(self, x, _tf=False, _tc=False):
        """Returns corresponding tensor or numpy array
        """
        if _tf:
            return -self.log_density(x, _tf=True)
        elif _tc:
            return -self.log_density(x, _tc=True)
        else:
            return -np.log(self.density(x))


    def _tf_grad_log_density(self, x):
        _dtype = np.float64

        _x = tf.Variable(x, dtype=_dtype, trainable=True)

        with tf.GradientTape() as tape:
            y = self._tf_gmm.log_prob(_x)

        grad = tape.gradient(y, _x)
        
        return grad.numpy()
    
    def _grad_log_density(self, x):
        mu = self.mu
        inv_sigma = np.linalg.inv(self.sigma)[0]

        # print(mu.shape)
        # print(inv_sigma.shape)
        return - np.einsum("jk, ij -> ik", inv_sigma, (x - mu))


    def grad_log_density(self, x, _tf=False):
        if _tf:
            return self._tf_grad_log_density(x)
        elif self.k == 1:
            return self._grad_log_density(x)
        else:
            gld = []
            comp_density = self._density(x)
            for i in range(self.k):
                mu_i = self.mu[i][np.newaxis,:]
                inv_sigma_i = np.linalg.inv(self.sigma[i])
                gld_i = - np.einsum("ij, kj -> ki", inv_sigma_i, (x - mu_i))
                
                gld.append(gld_i * comp_density[i])

            # print(self.density(x))
            return sum(gld)/self.density(x)

    def plot_contour(
        self, 
        X=None, Y=None, 
        N=300, 
        num_line=20,
        _tf=False,
        _tc=False, 
        _return_axes=True
        ):
        """Both X and Y are numpy arrays
        """
        if X is None:
            X = torch.linspace(-2, 7.5, N)
            
        if Y is None:
            Y = torch.linspace(-3, 7.5, N)

        _X, _Y = np.meshgrid(X, Y)
        pos = np.dstack((_X, _Y))
        print(pos.shape)

        if _tf:
            Z = self.density(pos, _tf=True)
            print(Z.shape)
        elif _tc:
            Z = self.density(pos, _tc=True)
        else:
            _X_dim = len(X)
            _Y_dim = len(Y)
            Z = self.density(
                pos.reshape(-1, pos.shape[-1])
                ).reshape(_X_dim, _Y_dim)

        if _return_axes:
            return X, Y, Z
        else:
            plt.figure()
            plt.contour(X, Y, Z, num_line, alpha=0.5)
            plt.xlabel(r"$x_1$")
            plt.ylabel(r"$x_2$")
            plt.title("Gaussian Mixture")
            plt.show()
            plt.close()


if __name__ == '__main__':

    # mu = torch.Tensor([1.2, .6], device=device)
    # sigma = (
    #     0.9*(torch.ones([2, 2], device=device) -
    #          torch.eye(2, device=device)).T +
    #     torch.eye(2, device=device)*1.3
    # )
    
    mu = np.array([[0.6, 1.2], [3., 2.6]])
    sigma = np.stack([np.diag([0.5, 1.2]) + np.ones([2,2])*0.3, np.diag([1.2, 0.6])])
    pi = 1/2 * np.ones(2)
    
    # mu = np.array([[0.6, 1.2]])
    # sigma = np.array([[[0.5, 0], [0, 1.2]]])
    # pi=np.ones(1)
    gmm = GaussianMixture({'mu':mu, 'sigma':sigma, 'pi':pi})

    # g = Gaussian({'mu':mu[0], 'sigma':np.diag(sigma[0])})
    # 2d position
    # N = 2
    # X = np.linspace(-2, 7, N)
    # Y = np.linspace(-3, 6, N)
    # X, Y = np.meshgrid(X, Y)
    # pos = np.dstack((X, Y))

    # pos = np.zeros([1,2])
    x = np.arange(10).reshape((5,2))
    pos = np.vstack([x, [0, 0]])

    # check density matches
    pdf_tf = gmm.density(pos, _tf=True)
    pdf_tc = gmm.density(pos, _tc=True)
    pdf_manual = gmm.density(pos).squeeze()

    print(pdf_tf)
    print(pdf_tc)
    print(pdf_manual)
    
    # Check contour
    gmm.plot_contour()

    # check grad log density matches
    gld_tf = gmm.grad_log_density(pos, _tf=True)
    gld_manual = gmm.grad_log_density(pos)

    print(gld_tf)
    print(gld_manual)
    
    # print(grad_autodiff)
    # print(grad_manual)

    # check the Gaussian Mixture with tf implementation

    # tfd = tfp.distributions
    # mix_gaussian = tfd.MixtureSameFamily(
    #     mixture_distribution=tfd.Categorical(
    #         probs=pi),
    #     components_distribution=
    #     tfd.MultivariateNormalTriL(
    #         loc=mu,
    #         scale_tril=tf.linalg.cholesky(np.array([[[0.5, 0],[0,1.2]]]))
    #     )
    # )
    # pdf_g_tf = np.exp(mix_gaussian.log_prob(pos).numpy())
    # print(pdf_g_tf)

