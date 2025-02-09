import torch
from torch.optim import Optimizer


# SFWBase implements a basis for variations of SFW as neural network optimizers. It is passed a constraint during
# initialization, which has a method LMO(x) = arg max <v,-x> for v within the constraint to calculate the update vector.
# It is also passed a momentum rate which sets the percentage with which to integrate the new gradient
# into the momentum vector. Lastly, it is passed a boolean 'rescaling' which determines whether to rescale the update
# vector to the same length as the current gradient.
# By setting rescaling = True and mr < 1, the MSFW method is obtained.
# By setting rescaling = False and mr = 1, the SFW method is obtained.
class SFWBase(Optimizer):
    def __init__(self, params, constraint, lr=0.01, mr=0.1, rescaling=True):
        defaults = {'lr': lr}
        self.mr = mr
        self.rescaling = rescaling
        self.Constraint = constraint
        super(SFWBase, self).__init__(params, defaults)

        # differentiates the groups of parameters according to their specified learning rates.
        # In this case, there is only one group, as all parameters learn with the same rate.
        for lr_group in self.param_groups:
            # param_group is one group of parameters, e.g. the weights or biases of one layer
            for param_group in lr_group['params']:
                # initialize momentum field for each parameter group to store the current momentum
                self.state[param_group] = {'momentum': torch.zeros_like(param_group.data)}

    def step(self, closure=None):
        for lr_group in self.param_groups:
            lr = lr_group['lr']
            for param_group in lr_group['params']:
                if param_group.grad is None:
                    continue
                # Calculate and store the new momentum.
                curr_momentum = self.state[param_group]['momentum']
                grad_data = param_group.grad.data
                new_momentum = (1 - self.mr) * curr_momentum + self.mr * grad_data
                self.state[param_group]['momentum'].copy_(new_momentum)

                # Calculate changes using the new momentum vector and LMO
                param_data = param_group.data
                changes = self.Constraint.LMO(new_momentum) - param_data

                # Implement update vector rescaling according to 'Deep Neural Network Training with Frank–Wolfe' - Pokutta
                # et al., 2020
                if self.rescaling:
                    rescaling_factor = min(1, lr * torch.norm(grad_data, 2) / (torch.norm(changes, 2)+1e-8))
                else:
                    rescaling_factor = lr
                param_data += rescaling_factor * changes

                # apply update
                param_group.data.copy_(param_data)


# Stochastic Frank-Wolfe method
class SFWOptimizer(SFWBase):
    def __init__(self, params, constraint, lr=0.001):
        super().__init__(params, constraint, lr, mr = 1, rescaling = False)


# Stochastic Frank-Wolfe method with rescaling
class SFWROptimizer(SFWBase):
    def __init__(self, params, constraint, lr=0.01):
        super().__init__(params, constraint, lr, mr = 1, rescaling = True)


# Stochastic Frank-Wolfe method with momentum
class SFWMOptimizer(SFWBase):
    def __init__(self, params, constraint, lr=0.001, mr=0.1):
        super().__init__(params, constraint, lr, mr = mr, rescaling = False)


# Modified Stochastic Frank-Wolfe method
class MSFWOptimizer(SFWBase):
    def __init__(self, params, constraint, lr=0.01, mr=0.1):
        super().__init__(params, constraint, lr, mr = mr, rescaling = True)