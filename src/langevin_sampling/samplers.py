#from .precondSGLD import pSGLD
import sys
import os

# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

from .SGLD import SGLD

import torch
import copy

class LangevinDynamics(object):
    def __init__(self, x, func, lr=1e2, lr_final=None, max_itr=1e4, 
                 device='cpu'):
        super(LangevinDynamics, self).__init__()

        self.x = x
        self.optim = SGLD([self.x], lr, momentum=0.0, weight_decay=0.0)
        self.lr = lr
        self.lr_final = lr_final
        self.max_itr = max_itr
        self.func = func
        # if without specifying the final lr, we use constant lr
        if self.lr_final is None:
            self.lr_fn = lambda t: self.lr
        else:
            self.lr_fn = self.decay_fn(lr=lr, lr_final=lr_final, max_itr=max_itr)
        self.counter = 0.0

    def sample(self):
        self.lr_decay()
        self.optim.zero_grad()

        if any(torch.isnan(self.x)):
            print('Chain diverges')
            # raise ValueError('Chain diverges')

        loss = self.func(self.x)
        loss.backward()
        self.optim.step()
        self.counter += 1
        return copy.deepcopy(self.x.data), loss.item()

    def decay_fn(self, lr=1e-2, lr_final=1e-4, max_itr=1e4):
        gamma = -0.55
        b = max_itr/((lr_final/lr)**(1/gamma) - 1.0)
        a = lr/(b**gamma)
        def lr_fn(t, a=a, b=b, gamma=gamma):
            return a*((b + t)**gamma)
        return lr_fn


    def lr_decay(self):
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr_fn(self.counter)


class MetropolisAdjustedLangevin(object):
    def __init__(self, x, func, lr=1e-2, lr_final=None, max_itr=1e4,
                 device='cpu'):
        super(MetropolisAdjustedLangevin, self).__init__()

        self.x = [
            torch.zeros(x.shape, device=x.device, requires_grad=True),
            torch.zeros(x.shape, device=x.device, requires_grad=True)
            ]
        self.x[0].data = x.data
        self.x[1].data = x.data

        self.loss = [torch.zeros([1], device=x.device),
                     torch.zeros([1], device=x.device)]
        self.loss[0] = func(self.x[0])
        self.loss[1].data = self.loss[0].data

        self.grad = [torch.zeros(x.shape, device=x.device),
                     torch.zeros(x.shape, device=x.device)]
        self.grad[0].data = torch.autograd.grad(self.loss[0], [self.x[0]],
            create_graph=False)[0].data
        self.grad[1].data = self.grad[0].data

        self.optim = SGLD([self.x[1]], lr, momentum=0.0, weight_decay=0.0)
        self.lr = lr
        self.lr_final = lr_final
        self.max_itr = max_itr
        self.func = func
        # if without specifying the final lr, we use constant lr
        if self.lr_final is None:
            self.lr_fn = lambda t: self.lr
        else:
            self.lr_fn = self.decay_fn(lr=lr, lr_final=lr_final, max_itr=max_itr)
        self.counter = 0.0

    def sample(self):
        accepted = False
        self.lr_decay()
        while not accepted:
            self.x[1].grad = self.grad[1].data
            self.P = self.optim.step()

            if any(torch.isnan(self.x[1])):
                print('Chain diverges')
                # raise ValueError('Chain diverges')

            self.loss[1] = self.func(self.x[1])
            self.grad[1].data = torch.autograd.grad(
                self.loss[1], [self.x[1]], create_graph=False)[0].data

            alpha = min([1.0, self.sample_prob()])
            if torch.rand([1]) <= alpha:
                self.grad[0].data = self.grad[1].data
                self.loss[0].data = self.loss[1].data
                self.x[0].data = self.x[1].data
                accepted = True
            else:
                self.x[1].data = self.x[0].data
        self.counter += 1
        return copy.deepcopy(self.x[1].data), self.loss[1].item()

    def proposal_dist(self, idx):
        return (-(.25 / self.lr_fn(self.counter)) *
                torch.norm(self.x[idx] - self.x[idx^1] -
                           self.lr_fn(self.counter)*self.grad[idx^1]/self.P)**2
        )

    def sample_prob(self):
        return torch.exp(-self.loss[1] + self.loss[0]) * \
            torch.exp(self.proposal_dist(0) - self.proposal_dist(1))

    def decay_fn(self, lr=1e-2, lr_final=1e-4, max_itr=1e4):
        gamma = -0.55
        b = max_itr/((lr_final/lr)**(1/gamma) - 1.0)
        a = lr/(b**gamma)
        def lr_fn(t, a=a, b=b, gamma=gamma):
            return a*((b + t)**gamma)
        return lr_fn

    def lr_decay(self):
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr_fn(self.counter)


if __name__ == '__main__':

    import numpy as np

    def decay_fn(lr=1e-2, lr_final=1e-4, max_itr=1e4):
        gamma = -0.55
        b = max_itr/((lr_final/lr)**(1/gamma) - 1.0)
        print("b is {}".format(b))
        a = lr/(b**gamma)
        print("a is {}".format(a))
        def lr_fn(t, a=a, b=b, gamma=gamma):
            return a*((b + t)**gamma)
        return lr_fn

    lr_lis = np.linspace(1e-2,1.3,3)

    for lr in lr_lis:
        print(decay_fn(lr=lr, max_itr=1000))