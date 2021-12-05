import sys
import os

module_path = os.path.abspath(__file__ + "/../../")
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import copy
from tqdm import tqdm
from functools import partial
# from  scipy.stats import multivariate_normal
# import scipy.stats as ss
import torch
import torch.distributions as D

import tensorflow as tf
import tensorflow_probability as tfp

from langevin_sampling.samplers import LangevinDynamics, MetropolisAdjustedLangevin
from ksd.distributions import GaussianMixture
from ksd.discrepancies import KSD
from ksd.kernels import InverseMultiquadricKernel

tf.random.set_seed(19)
np.random.seed(19)
torch.manual_seed(19)

if not torch.cuda.is_available():
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')
else:
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def _sample(
    name,
    init_state, 
    dist, 
    max_itr, 
    lr, 
    lr_final=None,
    burn_in=500, 
    thinning=None, 
    device=device,
    plot=False
    ):
    """ This function is used to sample from either 4 of the MCMC or SG-MCMC methods
    
    Input
    -----------
    name (string)           : 'mala', 'ula', 'sgmala', 'sgula'
    init_state (nd array)   : initial state of the chain
    dist                    : distrbution object from ksd.distrbutions
    lr (float)              : initial learning rate
    lr_final (float)        : final learning rate/ if None gives constant lr
    burn_in (int)           : number of burn in of the chain
    thinning (int)          : number of thinning of the chain
    device (string)         : specify device for SGLD samplers
    plot (bool)             : plot samples with contour 
    
    Return
    -----------
    est_samples (nd arrray) : samples drawn
    loss_log (nd array)     : negative loglikelihood evaluated at samples drawn
    """

    if thinning is None:
        n = max_itr + burn_in
    else:
        n = (thinning + 1) * max_itr + burn_in

    # Initialise sampler
    # note that only information on LL is used (to evaluate gradient)
    if name == 'mala':
        sampler = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=dist._tf_gmm.log_prob, # use the tf version of ll
            step_size=lr)

    elif name == 'ula':
        sampler = tfp.mcmc.UncalibratedLangevin(
            target_log_prob_fn=dist._tf_gmm.log_prob, # use the tf version of ll
            step_size=lr)

    elif name == 'sgmala':
        sampler = MetropolisAdjustedLangevin(
            torch.Tensor(init_state, requires_grad=True, device=device),
            partial(dist.neg_log_density(), _tc=True), # use the torch version of nll
            lr,
            lr_final=lr_final,
            max_itr=n,
            device=device
        )
    elif name == 'sgula':
        sampler = LangevinDynamics(
            torch.Tensor(init_state, requires_grad=True, device=device),
            partial(dist.neg_log_density(), _tc=True), # use the torch version of nl
            lr,
            lr_final=lr_final,
            max_itr=n,
            device=device
        )
    else:
        raise NameError('Specify sampler with name = mala, ula, sgmala, sgula')

    # sampling with the sampler
    if name == 'mala' or name == 'ula':
        _est_samples = tfp.mcmc.sample_chain(
            num_results=n,
            current_state=init_state,
            kernel=sampler,
            # ,num_steps_between_results=3
            num_burnin_steps=0,
            trace_fn=None
        )
        est_samples = np.array(_est_samples)[burn_in::thinning]
        loss_log = -np.apply_along_axis(dist._tf_gmm.log_prob, 1, est_samples)
    
    elif name == 'sgmala' or name == 'sgula':
        _est_samples = []
        _loss_log = []

        for j in range(n):
            est, loss = sampler.sample()
            _loss_log.append(loss)
            # if j%3 == 0: 
            # if without thinning -- same setup as MCMC 
            _est_samples.append(est.cpu().numpy())
        est_samples = np.array(_est_samples[burn_in::thinning])
        loss_log = np.array(_loss_log[burn_in::thinning])
    else:
        raise NameError('Specify sampler with name = mala, ula, sgmala, sgula')
    
    if plot:
        # use the plot_contour function to specify the axes
        X, Y, Z = dist.plot_contour(_return_axes=True)

        plt.figure("training logs - net", dpi=150, figsize=(7, 2.5))
        plt.plot(loss_log)
        plt.title("Negative Log-Likelihood")
        plt.xlabel("Iterations")
        plt.ylabel(r"$- \log \ \mathrm{N}(\mathbf{x} | \mu, \Sigma) + const.$")
        plt.grid()
        plt.show()
        plt.close()

        plt.figure(dpi=150, figsize=(9, 4))
        plt.contour(X, Y, Z, 20, alpha=0.5)
        plt.scatter(est_samples[:, 0], 
                    est_samples[:, 1], 
                    s=.5,
                    color="#db76bf")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        # plt.xlim([-3, 6])
        # plt.ylim([-4, 5])
        plt.title("Samples drawn with " + name)
        plt.show()
        plt.close()


    return est_samples, loss_log


def _eval_ksd(
    name,
    init_state, 
    dist, 
    max_itr, 
    lr_lis, 
    lr_final=None,
    burn_in=500, 
    thinning=None, 
    device=device,
    plot=False
    ):

    ksd_lis = []

    for i in tqdm(range(len(lr_lis))):
        est_samp = _sample(
        name,
        init_state, 
        dist, 
        max_itr, 
        lr_lis[i], 
        lr_final, 
        burn_in, 
        thinning, 
        device,
        plot=plot
        )
        
        _ksd = KSD('imq', dict(c=1, beta=-0.5), p=dist, q=est_samp)
        ksd_lis.append(_ksd.discrepancy())

    return ksd_lis

def compare_ksd(
    init_state, 
    dist, 
    max_itr, 
    lr_lis, 
    lr_final, # need to specify for the decay lr
    burn_in=500, 
    thinning=None, 
    device=device,
    plot=False
    ):

    name_lis = ['mala', 'ula', 'sgmala', 'sgula']
    ksd_dict = dict()
    # min_idx_lis = []
    min_lr_dict = dict()
    nan_lr_dict = dict() # find the lr for first occurance of nan

    # find ksd for sampler with constant lr
    plt.figure()
    for name in name_lis:
        print('sampling from '+name)

        # constant learning rate --------------------
        ksd_val = _eval_ksd(
        name,
        init_state, 
        dist, 
        max_itr, 
        lr_lis, 
        None, 
        burn_in, 
        thinning, 
        device,
        plot
        )
        ksd_dict[name+'_const'] = ksd_val

        # get the best lr in the list
        _min_idx = np.nanargmin(ksd_val)
        # _min_lr = lr_lis[_min_idx]
        # get the index for the first occurance of nan
        _nan_idx = np.where(np.isnan(ksd_val))[0]

        # min_idx_lis.append(_min_idx)
        min_lr_dict[name+'_const'] = (lr_lis[_min_idx])
        nan_lr_dict[name+'_const'] = (lr_lis[_nan_idx])

        plt.plot(lr_lis, ksd_val, label=name+'_const')

        # decay learning rate -----------------------
        ksd_val_decay = _eval_ksd(
        name,
        init_state, 
        dist, 
        max_itr, 
        lr_lis, 
        lr_final, # use decay learning rate
        burn_in, 
        thinning, 
        device,
        plot
        )
        ksd_dict[name+'_decay'] = ksd_val_decay

        # get the best lr in the list
        _min_idx = np.nanargmin(ksd_val_decay)
        # _min_lr = lr_lis[_min_idx]
        # get the index for the first occurance of nan
        _nan_idx = np.where(np.isnan(ksd_val_decay))[0]

        # min_idx_lis.append(_min_idx)
        min_lr_dict[name+'_decay'] = (lr_lis[_min_idx])
        nan_lr_dict[name+'_decay'] = (lr_lis[_nan_idx])

        plt.plot(lr_lis, np.log(ksd_val), label=name+'_decay')
    
    plt.title('Kernel Stein Discrepancy')
    plt.xlabel('Constant/Initial Step Size')
    plt.ylabel('log(KSD)')
    plt.yscale('log')
    plt.show()
    plt.close()

    return ksd_dict, min_lr_dict, nan_lr_dict


# TODO:
# - debug the code
# - need to write code to control the relative position of the mean of the gmm 
# - then use compare_ksd to select the best lr
    # - in the gmm case, check how the best lr changes w.r.t. 
        # the Euclidean dist between modes of components
    # - in the single MVN case (eucl_dist=0), 
        # check that it cororborates the Roberts & Tweedie result

def compare_dist():
    pass


