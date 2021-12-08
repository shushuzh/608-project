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
import seaborn as sns
import copy
import time

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
        # uses zeros for initialisation for now
        # init = torch.from_numpy(init_state)
        den_fct = partial(dist.neg_log_density, _tc=True, _tf=False)
        
        sampler = MetropolisAdjustedLangevin(
            torch.zeros([dist.d], requires_grad=True, device=device)
            #  + init
             ,
            den_fct, # use the torch version of nll
            lr,
            lr_final=lr_final,
            max_itr=n,
            device=device
        )
    elif name == 'sgula':
        # init = torch.from_numpy(init_state)
        den_fct = partial(dist.neg_log_density, _tc=True, _tf=False)

        sampler = LangevinDynamics(
            torch.zeros([dist.d], requires_grad=True, device=device)
            #  + init
             ,
            den_fct, # use the torch version of nll
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
        ess = tfp.mcmc.effective_sample_size(_est_samples[burn_in::thinning])
        # print(ess)
        # print("number of samples accepted", len(est_samples))
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
        ess = tfp.mcmc.effective_sample_size(tf.convert_to_tensor(_est_samples)[burn_in::thinning])
    else:
        raise NameError('Specify sampler with name = mala, ula, sgmala, sgula')
    
    if plot:
        flag = (name == 'sgmala' or name == 'sgula') and (lr_final is not None)

        # if flag:
        #     lr_type = '(decay lr)'
        # else:
        #     lr_type = '(const lr)'
        # use the plot_contour function to specify the axes
        X, Y, Z = dist.plot_contour(_return_axes=True)

        # plt.figure("training logs - net", dpi=150, figsize=(7, 2.5))
        # plt.plot(-loss_log)
        # plt.title("Negative Log-Likelihood")
        # plt.xlabel("Iterations")
        # plt.ylabel(r"$- \log \ \mathrm{N}(\mathbf{x} | \mu, \Sigma) + const.$")
        # plt.grid()
        # plt.show()
        # plt.close()

        # plt.figure(dpi=150, figsize=(6, 6))
        plt.contour(X, Y, Z, 20, alpha=0.5)
        plt.scatter(est_samples[:, 0], 
                    est_samples[:, 1], 
                    s=.5,
                    color="#db76bf")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        # plt.xlim([-3, 6])
        # plt.ylim([-4, 5])
        


    return est_samples, loss_log, ess


def _plot_transit(
    step,
    name,
    mu0,
    sigma,
    pi,
    init_state, 
    max_itr, 
    lr, 
    lr_final=None,
    dis_lim=5,
    burn_in=500, 
    thinning=None, 
    device=device
    # ,
    # plot=True
    ):

    ksd_lis_const = []
    dis_lis = []

    dis = 0
    _lr = int(lr*10)

    # check if we have decay stepsize
    flag = (name == 'sgmala' or name == 'sgula') and (lr_final is not None)
    if flag:
        ksd_lis_decay = []

    # print("sample from bimodal --------")
    mu1 = copy.deepcopy(mu0)
    counter = 0
    while dis < dis_lim:
        # update distance
        dis = np.linalg.norm(mu1-mu0)
        dis_lis.append(dis)

        # update iteration count
        counter += 1

        # update second mode
        mu1 += step
        mu = np.concatenate((mu0, mu1), axis=0)
        gmm = GaussianMixture({'mu':mu, 'sigma':sigma, 'pi':pi})

        plt.figure(dpi=150, figsize=(6, 6))
        est_samp, _, _ = _sample(
            name,
            init_state, 
            gmm, 
            max_itr, 
            lr, 
            None, 
            burn_in, 
            thinning, 
            device,
            plot=True
        )
        plt.title("Samples drawn with " + name + " " + "(lr={})".format(lr))
        plt.savefig('img/contour/contour_{0}_const_{1}_{2}.pdf'.format(name, _lr, counter), 
                    bbox_inches="tight")
        # plt.show()
        plt.close()

        _ksd = KSD('imq', dict(c=1, beta=-0.5), p=gmm, q=est_samp)
        ksd_lis_const.append(_ksd.discrepancy())

        # print(sum(np.isnan(est_samp)))
        # print(est_samp[:,0])

        if flag:
            plt.figure(dpi=150, figsize=(6, 6))
            est_samp, _, _ = _sample(
                name,
                init_state, 
                gmm, 
                max_itr, 
                lr, 
                lr_final, 
                burn_in, 
                thinning, 
                device,
                plot=True
            )
            plt.title("Samples drawn with " + name + " " + "(init lr={})".format(lr))
            plt.savefig('img/contour/contour_{0}_decay_{1}_{2}.pdf'.format(name, _lr, counter), 
                        bbox_inches="tight")
            # plt.show()
            plt.close()
            
            _ksd = KSD('imq', dict(c=1, beta=-0.5), p=gmm, q=est_samp)
            ksd_lis_decay.append(_ksd.discrepancy())

    plt.figure(dpi=150, figsize=(7, 2.5))
    plt.plot(dis_lis, ksd_lis_const)
    plt.savefig('img/contour/ksd_{0}_const_{1}.pdf'.format(name, _lr), 
                        bbox_inches="tight")
    # plt.show()
    plt.close()

    if flag:
        plt.figure(dpi=150, figsize=(7, 2.5))
        plt.plot(dis_lis, ksd_lis_decay)
        plt.savefig('img/contour/ksd_{0}_decay_{1}.pdf'.format(name, _lr), 
                        bbox_inches="tight")
        # plt.show()
        plt.close()

    # pass



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
    plot=False,
    _plot = False
    ):

    ksd_lis = []
    elp_lis = []
    ess_lis = []

    for i in tqdm(range(len(lr_lis))):
        start = time.time()

        est_samp, _, ess = _sample(
        name,
        init_state, 
        dist, 
        max_itr, 
        lr_lis[i], 
        lr_final, 
        burn_in, 
        thinning, 
        device,
        plot=_plot
        )
        end = time.time()

        _ksd = KSD('imq', dict(c=1, beta=-0.5), p=dist, q=est_samp)

        ksd_lis.append(_ksd.discrepancy())
        elp_lis.append(end - start)
        ess_lis.append(ess)

    if plot:
        plt.figure()
        plt.plot(lr_lis, ksd_lis)
        plt.show()
        plt.close()
 
        plt.figure()
        plt.plot(lr_lis, elp_lis)
        plt.show()
        plt.close()

        plt.figure()
        plt.plot(lr_lis, ess_lis)
        plt.show()
        plt.close()

    return ksd_lis, elp_lis, np.vstack(ess_lis)

def compare_ksd(
    name_lis,
    init_state, 
    dist, 
    max_itr, 
    lr_lis, 
    lr_final, # need to specify for the decay lr
    burn_in=500, 
    thinning=None, 
    device=device
    # ,
    # plot=False
    # _plot=False
    ):

    # name_lis = ['mala', 'ula', 'sgmala', 'sgula']
    # name_lis = ['mala']
    ksd_dict = dict()
    elp_dict = dict()
    ess_dict = dict()
    # min_idx_lis = []
    min_lr_dict = dict()
    # detect divergences
    # find the lr for first occurance of nan
    nan_lr_dict = dict() 

    # find ksd for sampler with constant lr
    
    for name in name_lis:
        print('sampling from '+name)

        # constant learning rate --------------------
        ksd_val, elp_time, ess = _eval_ksd(
        name,
        init_state, 
        dist, 
        max_itr, 
        lr_lis, 
        None, 
        burn_in, 
        thinning, 
        device
        # ,
        # plot,
        # _plot
        )
        ksd_dict[name+'_const'] = ksd_val
        # print(ksd_val)
        elp_dict[name+'_const'] = elp_time
        ess_dict[name+'_const'] = ess[:,0]
        # get the best lr in the lis
        _min_idx = np.nanargmin(ksd_val)
        # _min_lr = lr_lis[_min_idx]
        # get the index for the first occurance of nan
        _nan_idx = np.where(np.isnan(ksd_val))[0]

        # min_idx_lis.append(_min_idx)
        min_lr_dict[name+'_const'] = (lr_lis[_min_idx])
        nan_lr_dict[name+'_const'] = (lr_lis[_nan_idx])

        # decay learning rate -----------------------
        flag = (name == 'sgmala' or name == 'sgula') and (lr_final is not None)
        
        if flag:
            ksd_val_decay, elp_time, ess = _eval_ksd(
                name,
                init_state, 
                dist, 
                max_itr, 
                lr_lis, 
                lr_final, # use decay learning rate
                burn_in, 
                thinning, 
                device
            )
            ksd_dict[name+'_decay'] = ksd_val_decay
            elp_dict[name+'_decay'] = elp_time
            ess_dict[name+'_decay'] = ess[:,0]

            # get the best lr in the lis
            _min_idx = np.nanargmin(ksd_val_decay)
            # _min_lr = lr_lis[_min_idx]
            # get the index for the first occurance of nan
            _nan_idx = np.where(np.isnan(ksd_val_decay))[0]

            # min_idx_lis.append(_min_idx)
            min_lr_dict[name+'_decay'] = (lr_lis[_min_idx])
            nan_lr_dict[name+'_decay'] = (lr_lis[_nan_idx])
    

    plt.figure(dpi=150, figsize=(9, 6))
    for _name, _val in ksd_dict.items():
        plt.plot(lr_lis, _val, label=_name)

    plt.title('Kernel Stein Discrepancy')
    plt.xlabel('Constant/Initial Step Size')
    plt.ylabel('KSD')
    # plt.yscale('log')
    # plt.legend()
    plt.legend(loc='upper center', 
            bbox_to_anchor=(0.5, -0.1),
            borderaxespad=1,
            fancybox=True, ncol=4)
    plt.savefig('img/ksd_compare_{}.pdf'.format(dist.k), bbox_inches="tight")
    plt.show()
    plt.close()

    plt.figure(dpi=150, figsize=(9, 6))
    for _name, _val in elp_dict.items():
        plt.plot(lr_lis, _val, label=_name)
    plt.title('Time Elapsed')
    plt.xlabel('Constant/Initial Step Size')
    plt.ylabel('Time')
    # plt.yscale('log')
    # plt.legend()
    plt.legend(loc='upper center', 
            bbox_to_anchor=(0.5, -0.1),
            borderaxespad=1,
            fancybox=True, ncol=4)
    plt.savefig('img/time_compare_{}.pdf'.format(dist.k), bbox_inches="tight")
    plt.show()
    plt.close()

    plt.figure(dpi=150, figsize=(9, 6))
    for _name, _val in ess_dict.items():
        plt.plot(lr_lis, _val, label=_name)
    plt.title('Essential Sample Size')
    plt.xlabel('Constant/Initial Step Size')
    plt.ylabel('ESS')
    plt.yscale('log')
    # plt.legend()
    plt.legend(loc='upper center', 
            bbox_to_anchor=(0.5, -0.1),
            borderaxespad=1,
            fancybox=True, ncol=4)
    plt.savefig('img/ess_compare_{}.pdf'.format(dist.k), bbox_inches="tight")
    plt.show()
    plt.close()

    return ksd_dict, min_lr_dict, nan_lr_dict

"""
TODO
- [x] debug the code
- [] write code for plotting contour of evolutions of samples, compare to ground truth
- [x] need to write code to control the relative position of the mean of the gmm 
- [] then use compare_ksd to select the best lr
  - [] in the gmm case, check how the best lr changes w.r.t. 
      the Euclidean dist between modes of components
  - [x] in the single MVN case (eucl_dist=0), 
      check that it cororborates the Roberts & Tweedie result (MALA)
"""

def _compare_mean_dist(
        step,
        name,
        mu0,
        sigma,
        pi,
        init_state, 
        n, 
        lr_lis, 
        lr_final=None, # need to specify for the decay lr
        burn_in=500, 
        thinning=None, 
        dis_lim=5,
        device=device
        ):

    dis = 0

    # check if we have decay stepsize
    flag = (name == 'sgmala' or name == 'sgula') and (lr_final is not None)

    ksd_lis_const = []
    elp_lis_const = []
    ess_lis_const = []
    dis_lis = []
    

    if flag:
        ksd_lis_decay = []
        elp_lis_decay = []
        ess_lis_decay = []

    # bimodal case -----------------------
    mu1 = copy.deepcopy(mu0)
    print("sample from bimodal --------")
    while dis < dis_lim:
        mu1 += step
        mu = np.concatenate((mu0, mu1), axis=0)

        gmm = GaussianMixture({'mu':mu, 'sigma':sigma, 'pi':pi})

        ksd_val, elp_time, ess = _eval_ksd(
            name,
            init_state, 
            gmm, 
            n, 
            lr_lis, 
            None, # need to specify for the decay lr
            burn_in, 
            thinning, 
            device
        )
        ksd_lis_const.append(ksd_val)
        elp_lis_const.append(elp_time)
        ess_lis_const.append(ess[:,0])
        
        if flag:
            ksd_val, elp_time, ess = _eval_ksd(
                name,
                init_state, 
                gmm, 
                n, 
                lr_lis, 
                lr_final, # need to specify for the decay lr
                burn_in, 
                thinning, 
                device
            )
            ksd_lis_decay.append(ksd_val)
            elp_lis_decay.append(elp_time)
            ess_lis_decay.append(ess[:,0])

        dis = np.linalg.norm(mu1-mu0)
        dis_lis.append(dis)

        # print(dis)
    
    ksd_mat_const = np.vstack(ksd_lis_const)
    elp_mat_const = np.vstack(elp_lis_const)
    ess_mat_const = np.vstack(ess_lis_const)

    # plot the heatmap
    plt.figure(dpi=150, figsize=(10, 6))
    sns.heatmap(ksd_mat_const, cmap="YlGnBu", 
                xticklabels=np.round(lr_lis,2),
                yticklabels=np.round(dis_lis ,2))
    plt.title('KSD (Constant Step Size)')
    plt.xlabel('Step Size')
    plt.ylabel('Mode Distance')
    # plt.savefig('img/ksd_mean_{}_const.pdf'.format(name), bbox_inches="tight")
    plt.savefig('img/ksd_mean_{}_test.pdf'.format(name), bbox_inches="tight")
    plt.show()
    plt.close()

    plt.figure(dpi=150, figsize=(10, 6))
    sns.heatmap(elp_mat_const, cmap="YlGnBu", 
                xticklabels=np.round(lr_lis,2),
                yticklabels=np.round(dis_lis,2))
    plt.title('Time Elapsed')
    plt.xlabel('Step Size')
    plt.ylabel('Mode Distance')
    # plt.savefig('img/time_mean_{}_const.pdf'.format(name), bbox_inches="tight")
    plt.savefig('img/time_mean_{}_test.pdf'.format(name), bbox_inches="tight")
    plt.show()
    plt.close()

    plt.figure(dpi=150, figsize=(10, 6))
    sns.heatmap(ess_mat_const, cmap="YlGnBu", 
                xticklabels=np.round(lr_lis,2),
                yticklabels=np.round(dis_lis,2))
    plt.title('ESS')
    plt.xlabel('Step Size')
    plt.ylabel('Essential Sample Size')
    # plt.savefig('img/ess_mean_{}_const.pdf'.format(name), bbox_inches="tight")
    plt.savefig('img/ess_mean_{}_test.pdf'.format(name), bbox_inches="tight")
    plt.show()
    plt.close()

    if flag:
        ksd_mat_decay = np.vstack(ksd_lis_decay)
        elp_mat_decay = np.vstack(elp_lis_decay)
        ess_mat_decay = np.vstack(ess_lis_decay)

        plt.figure(dpi=150, figsize=(10, 6))
        sns.heatmap(ksd_mat_decay, cmap="YlGnBu", 
                    xticklabels=np.round(lr_lis,2),
                    yticklabels=np.round(dis_lis,2))
        plt.title('KSD (Decay Step Size)')
        plt.xlabel('Initial Step Size')
        plt.ylabel('Mode Distance')
        plt.savefig('img/ksd_mean_{}_decay.pdf'.format(name), bbox_inches="tight")
        plt.show()
        plt.close()

        plt.figure(dpi=150, figsize=(10, 6))
        sns.heatmap(elp_mat_decay, cmap="YlGnBu", 
                    xticklabels=np.round(lr_lis,2),
                    yticklabels=np.round(dis_lis,2))
        plt.title('Time Elapsed')
        plt.xlabel('Step Size')
        plt.ylabel('Mode Distance')
        plt.savefig('img/time_mean_{}_decay.pdf'.format(name), bbox_inches="tight")
        plt.show()
        plt.close()

        plt.figure(dpi=150, figsize=(10, 6))
        sns.heatmap(ess_mat_decay, cmap="YlGnBu", 
                    xticklabels=np.round(lr_lis,2),
                    yticklabels=np.round(dis_lis,2))
        plt.title('ESS')
        plt.xlabel('Step Size')
        plt.ylabel('Essential Sample Size')
        plt.savefig('img/ess_mean_{}_decay.pdf'.format(name), bbox_inches="tight")
        plt.show()
        plt.close()


def compare_mean_dist(
        step,
        mu0,
        sigma,
        pi,
        init_state, 
        n, 
        lr_lis, 
        lr_final=None, # need to specify for the decay lr
        burn_in=500, 
        thinning=None, 
        device=device,
        k_is_1=False
        ):
    
    pi = np.arange(2,4)/sum(np.arange(2,4)) # !! remember to delete this

    name_lis = ['mala', 'ula', 'sgmala', 'sgula']
    for name in name_lis:
        _compare_mean_dist(
            step,
            name,
            mu0,
            sigma,
            pi,
            init_state, 
            n, 
            lr_lis, 
            lr_final, # need to specify for the decay lr
            burn_in, 
            thinning, 
            device,
            k_is_1
            )
    
    pass


if __name__ == '__main__':
    # # sgmala and sgula diverges around 1.7
    # mu = np.array([[0.6, 1.2], [0.6+5, 1.2+5]])
    # sigma = np.stack([np.diag([0.5, 1.2]) + np.ones([2,2])*0.3, np.diag([1.2, 0.6])])
    # # pi = 1/2 * np.ones(2)
    # pi = np.arange(2,4)/sum(np.arange(2,4))

    # # sgmala and sgula diverges around 1.3
    # # mu = np.array([[0.6, 1.2]])
    # # pi = np.ones(1)
    # # sigma = np.stack([np.diag([0.5, 1.2]) + np.ones([2,2])*0.3])

    # gmm = GaussianMixture({'mu':mu, 'sigma':sigma, 'pi':pi})
    # init_state = -np.ones(2)
    # n = 2000
    # burn_in = 0
    # lr = 0.2
    

    ### sampler check ------------------------------------
    # samp_, time_, ess_ = _sample(
    #     "sgmala",
    #     init_state, 
    #     gmm, 
    #     n, 
    #     lr, 
    #     # lr_final=4e-2,
    #     lr_final=None,
    #     burn_in=burn_in, 
    #     thinning=None, 
    #     device=device,
    #     plot=False
    # )
    # print(ess_)

    ### KSD check -----------------------------------------
    # lr_lis = np.linspace(0,1,5)
    # burn_in = 1000
    # n = 2000

    # ksd_lis, _, ess = _eval_ksd(
    #     "sgmala",
    #     init_state, 
    #     gmm, 
    #     n, 
    #     lr_lis, 
    #     lr_final=None,
    #     burn_in=burn_in, 
    #     thinning=None, 
    #     device=device,
    #     plot=True,
    #     _plot=False
    # )
    # print(ess)
    ### Compare KSD ------------------------------------
    # mu = np.array([[0.6, 1.2], [3.6, 4.2]])
    # sigma = sigma = np.stack([np.diag([0.5, 1.2]) + np.ones([2,2])*0.3, np.diag([1.2, 0.6])])
    # pi = np.arange(2,4)/sum(np.arange(2,4))

    # ## sgmala and sgula diverges around 1.3
    # # mu = np.array([[0.6, 1.2]])
    # # pi = np.ones(1)
    # # sigma = np.stack([np.diag([0.5, 1.2]) + np.ones([2,2])*0.3])

    # name_lis = ['mala', 'ula', 'sgmala', 'sgula']
    # # name_lis = ['sgmala']
    # gmm = GaussianMixture({'mu':mu, 'sigma':sigma, 'pi':pi})
    # init_state = np.zeros(2)
    # n = 2000
    # burn_in = 1000
    # lr = 0.72
    # lr_lis = np.linspace(0.045,1.3,10)

    # # sgmala and sgula diverges around 1.78 for the above MVN
    # ksd_dict, min_lr_dict, nan_lr_dict = compare_ksd(
    #     name_lis,
    #     init_state, 
    #     gmm, 
    #     n, 
    #     lr_lis, 
    #     lr_final=4e-2, # need to specify for the decay lr
    #     burn_in=burn_in, 
    #     thinning=None, 
    #     device=device
    #     )
    # print("ksd_dict")
    # print(ksd_dict)
    # print("min_lr_dict")
    # print(min_lr_dict)
    # print("nan_lr_dict")
    # print(nan_lr_dict)
    
    ### heatmap KSD ------------------------------------
    # init dist
    mu0 = np.array([[0.6, 1.2]])
    pi = np.arange(2,4)/sum(np.arange(2,4))
    sigma = np.stack([np.diag([0.5, 1.2]) + np.ones([2,2])*0.3, np.diag([1.2, 0.6])])
    
    # MCMC
    init_state = np.zeros(2)
    n = 2000
    burn_in = 1000
    lr_final = None
    dis_lim = 8
    step = 0.5

    name = 'mala'

    if name == 'sgula':
        if lr_final is None:
            lr_lis = np.linspace(0.001,1.3,10)
        else:
            lr_lis = np.linspace(0.045,1.3,10)
    elif name == 'sgmala':
        if lr_final is None:
            lr_lis = np.linspace(0.001,1.3,10)
        else:
            lr_lis = np.linspace(0.045,1.3,10)
    elif name == 'ula':
        lr_lis = np.linspace(0.001,3.1,10)
    elif name == 'mala':
        lr_lis = np.linspace(0.001,10,15)

    
    _compare_mean_dist(
        step,
        name,
        mu0,
        sigma,
        pi,
        init_state, 
        n, 
        lr_lis, 
        lr_final, # need to specify for the decay lr
        burn_in, 
        thinning=None, 
        dis_lim=dis_lim,
        device=device
        )

    ### plot contour ----------------------------------------
    # # init dist
    # mu0 = np.array([[0.6, 1.2]])
    # pi = np.arange(2,4)/sum(np.arange(2,4))
    # sigma = np.stack([np.diag([0.5, 1.2]) + np.ones([2,2])*0.3, np.diag([1.2, 0.6])])
    
    # # MCMC
    # init_state = np.zeros(2)
    # n = 2000
    # burn_in = 500
    # lr_final = 4e-2
    # dis_lim = 5
    # step = 0.25

    # name = 'ula'

    # if name == 'mala':
    #     # lr = 3.1
    #     lr = 1.33
    # elif name == 'ula':
    #     lr = 1.33
    # elif name == 'sgmala':
    #     lr = 1.33
    # elif name == 'sgula':
    #     lr = 1.33

    # _plot_transit(
    #     step,
    #     name,
    #     mu0,
    #     sigma,
    #     pi,
    #     init_state, 
    #     n, 
    #     lr, 
    #     lr_final=lr_final,
    #     dis_lim=dis_lim,
    #     burn_in=burn_in, 
    #     thinning=None, 
    #     device=device
    #     )

    # the unimodal case ------------------
    # mu0 = np.array([[0.6, 1.2]])
    # sigma0 = np.stack([np.diag([0.5, 1.2]) + np.ones([2,2])*0.3])
    
    # if k_is_1:
    #     pi0 = np.ones(1)
    #     sigma0 = np.stack([sigma[0]])
    #     gmm0 = GaussianMixture({'mu':mu0, 'sigma':sigma0, 'pi':pi0})

    #     # ksd_mat = np.array([])
    #     print("sample from unimodal --------")
    #     ksd_dict0, _, _ = compare_ksd(
    #             [name],
    #             init_state, 
    #             gmm0, 
    #             n, 
    #             lr_lis, 
    #             lr_final, # need to specify for the decay lr
    #             burn_in, 
    #             thinning, 
    #             device
    #         )
    #     # initialise
    #     ksd_mat_const = ksd_dict0[name+'_const']

    #     if flag:
    #         ksd_mat_decay = ksd_dict0[name+'_decay']
    # else:
    #     ksd_mat_decay = np.array([])