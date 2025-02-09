from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.init as init
import math
from initialization import initialize_model


# Abstract class all constraints have to implement. Every constraint has:
# - a method 'initialize' to initialize the parameters within its specific feasible region
# - a method 'LMO' to solve the linear sub-problem of the Frank-Wolfe algorithm
# - a given parameter 'w' which indicates the radius of the constraint (the L2-diameter will be 2*w*E[|x|],
#   where E[|x|] is the expected L2 norm of the parameters after initialization)
# - a parameter 'radius' which stores the radii for all groups of parameters, i.e. one radius the weights and one
#   radius for the biases of each layer
# - a parameter 'curr_group' to track the parameter group currently optimized, which is used to select the correct
#   entry of the 'radius' array
class Constraint(ABC):
    def __init__(self, w):
        self.w = w
        self.radius = []
        self.curr_group = 0

    @abstractmethod
    # Each constraint will implement two functions:
    # - a converter of a given L2-diameter into the radius needed for that constraint to have such an L2-diameter
    # - a method which initializes the weights according to a normal distribution of mean 0 and a given standard
    #   deviation while making sure that the resulting parameters remain within the feasible region
    # They will then call 'initialize_model' from 'initialization.py' with these functions as arguments, where the
    # model will be initialized.
    def initialize(self, model):
        pass

    @abstractmethod
    def LMO(self, grad_data):
        pass


# Given the Hyperparameters p and w, this returns a Lp-Normball constraint with radius calculated from w as described above.
class LpNormball(Constraint):
    def __init__(self, p, w):
        super().__init__(w)
        if not (p == 'inf' or p >= 1):
            raise ValueError("p has to be 'inf' or a numerical value >= 1")
        self.p = p

    def initialize(self, model):
        def diameter_converter(l2diam, num_params):
            if self.p == 'inf':
                return l2diam/2*math.sqrt(num_params)
            elif self.p == 1:
                return l2diam/2
            else:
                return l2diam/(2 * num_params ** (1 / 2 - 1 / self.p))

        # If the Lp-Norm of the parameters is too large after initialization according to the normal distribution, they
        # get projected back into the constraint to ensure a feasible initialization.
        def initializer_lp(layer, std, tau_weights, tau_biases):
            init.normal_(layer.weight, mean=0, std=std)
            # If weights or biases surpass allowed Lp-norm, project back into the Lp-norm-ball
            if torch.norm(layer.weight, float(self.p)) > tau_weights:
                layer.weight.data = tau_weights * layer.weight.data / torch.norm(layer.weight.data, float(self.p))

            # Repeat same procedure on biases, should they exist
            if layer.bias is not None:
                init.normal_(layer.bias, mean=0, std=std)
                if torch.norm(layer.bias, float(self.p)) > tau_biases:
                    layer.bias.data = tau_biases * layer.bias.data / torch.norm(layer.bias.data, float(self.p))

        initialize_model(diameter_converter, initializer_lp, model, self.w, self.radius)

    def LMO(self, grad_data):
        tau = self.radius[self.curr_group]
        self.curr_group = (self.curr_group + 1) % len(self.radius)

        # For p=1 or p='inf', a different formula has to be used.
        if self.p == 1:
            absval_grad = torch.abs(grad_data)
            max_abs_value = absval_grad.max()
            update_entries = (absval_grad == max_abs_value).float()
            change_vector = - tau * torch.sign(grad_data) * update_entries
        elif self.p == 'inf':
            change_vector = - tau * torch.sign(grad_data)
        else:
            q = self.p / (self.p - 1)
            absval_grad_qp = torch.abs(grad_data) ** (q / self.p)
            qnorm_grad_qp = torch.norm(grad_data, q) ** (q / self.p)
            change_vector = - tau * torch.sign(grad_data) * absval_grad_qp / qnorm_grad_qp
        return change_vector


# Given the Hyperparameters K_percentage and w, this returns a K-sparse polytope constraint with radius calculated from w as
# described above. K is calculated as K_percentage of the total parameters.
class KsparsePolytope(Constraint):
    def __init__(self, K_percentage, w):
        super().__init__(w)
        self.K_percentage = K_percentage

    def initialize(self, model):
        def diameter_converter(l2diam, num_params):
            return l2diam/(2 * math.sqrt(num_params * self.K_percentage))

        # By not allowing the parameter absolute values to surpass |tau|/ceil(1/K_percentage), it is guaranteed that
        # the parameters are initialized within the K-sparse polytope.
        def initializer_ksparse(layer, std, tau_weights, tau_biases):
            support_vectors = math.ceil(1/self.K_percentage)
            init.trunc_normal_(layer.weight, mean=0, std=std, a=-tau_weights/support_vectors, b=tau_weights/support_vectors)
            # Repeat same procedure on biases, should they exist
            if layer.bias is not None:
                init.trunc_normal_(layer.bias, mean=0, std=std, a=-tau_biases/support_vectors, b=tau_biases/support_vectors)

        initialize_model(diameter_converter, initializer_ksparse, model, self.w, self.radius)

    def LMO(self, grad_data):
        tau = self.radius[self.curr_group]
        self.curr_group = (self.curr_group + 1) % len(self.radius)

        # calculate a vector which is 1 in the highest k entries of the gradient and 0 otherwise
        top_k_indices = torch.topk(grad_data.flatten().abs(), int(self.K_percentage * grad_data.numel())).indices
        sparse_grad = torch.zeros_like(grad_data)
        sparse_grad.view(-1)[top_k_indices] = 1
        # The sparse vector can now be used to capture the sign of highest k entries, while leaving all other entries 0.
        # Multiplying by tau delivers the desired final vector.
        change_vector = - tau * sparse_grad * torch.sign(grad_data)
        return change_vector


# Given the Hyperparameters K_percentage and w, this returns a K-support Normball constraint with radius calculated from w as
# described above. K is calculated as K_percentage of the total parameters.
class KsupportNormball(Constraint):
    def __init__(self, K_percentage, w):
        super().__init__(w)
        self.K_percentage = K_percentage

    # this code calculating the K-support norm of a vector x stems from
    # https://github.com/ZIB-IOL/compression-aware-SFW/blob/main/optimizers/constraints.py#L559. It is based on the
    # definition of the k-support norm found in 'Sparse Prediction with the k-Support Norm' - Argyriou et al., 2012
    @staticmethod
    def k_support_norm(x, K, tol=1e-8):
        sorted_increasing = torch.sort(torch.abs(x.flatten()), descending=False).values
        running_mean = torch.cumsum(sorted_increasing, 0)  # Compute the entire running_mean since this is optimized
        running_mean = running_mean[-K:]  # Throw away everything but the last entries k entries
        running_mean = running_mean / torch.arange(1, K + 1, device=x.device)
        lower = sorted_increasing[-K:]
        upper = torch.cat([sorted_increasing[-(K - 1):], torch.tensor([float('inf')], device=x.device)])
        relevantIndices = torch.nonzero(torch.logical_and(upper + tol > running_mean, running_mean + tol >= lower))[0]
        r = int(relevantIndices[0])
        d = x.numel()
        x_right = 1 / (r + 1) * torch.sum(sorted_increasing[:d - (K - r) + 1]).pow(2)
        x_left = torch.sum(sorted_increasing[-(K - 1 - r):].pow(2)) if r < K - 1 else 0
        x_norm = torch.sqrt(x_left + x_right)
        return x_norm

    def initialize(self, model):
        def diameter_converter(l2diam, num_params):
            return l2diam/2

        # Similar to the Lp-Normball, parameters are initialized with the normal distribution and projected back into
        # the Normball if necessary.
        def initializer_ksupport(layer, std, tau_weights, tau_biases):
            num_weights = layer.weight.numel()
            init.normal_(layer.weight, mean=0, std=std)
            # If weights surpass allowed K-support-norm, project back into the K-support-Normball
            if KsupportNormball.k_support_norm(layer.weight.data, int(self.K_percentage * num_weights)) > tau_weights:
                layer.weight.data = tau_weights * layer.weight.data / KsupportNormball.k_support_norm(layer.weight.data, int(self.K_percentage * num_weights))

            # Repeat same procedure on biases, should they exist
            if layer.bias is not None:
                num_biases = layer.bias.numel()
                init.normal_(layer.bias, mean=0, std=std)
                if KsupportNormball.k_support_norm(layer.bias.data, int(self.K_percentage * num_biases)) > tau_biases:
                    layer.bias.data = tau_biases * layer.bias.data / KsupportNormball.k_support_norm(layer.bias.data, int(self.K_percentage * num_biases))

        initialize_model(diameter_converter, initializer_ksupport, model, self.w, self.radius)

    def LMO(self, grad_data):
        tau = self.radius[self.curr_group]
        self.curr_group = (self.curr_group + 1) % len(self.radius)

        # The sparse vector is calculated similar to the K-sparse polytope, except this time it is filled with the
        # values and not just the signs.
        top_k_indices = torch.topk(grad_data.flatten().abs(), int(self.K_percentage * grad_data.numel())).indices
        sparse_grad = torch.zeros_like(grad_data)
        sparse_grad.view(-1)[top_k_indices] = grad_data.view(-1)[top_k_indices]
        # The desired vector is obtained by normalizing the L2-Norm properly. Add small quantity to avoid division by 0.
        change_vector = - tau * sparse_grad / (torch.norm(sparse_grad, 2)+1e-8)
        return change_vector


# Given the Hyperparameters K_percentage and w, this returns a Group K-sparse polytope constraint with radius calculated
# from w as described above. K is calculated as K_percentage of the total parameters. On linear layers, K describes the
# number of weights, while on convolutional layers, K describes the number of filters.
class GroupKsparsePolytope(KsparsePolytope):
    def __init__(self, K_percentage, w):
        super().__init__(K_percentage, w)

    # model initialization as in the K-sparse polytope also guarantees initialization within the Group K-sparse polytope
    def initialize(self, model):
        super().initialize(model)

    def LMO(self, grad_data):
        # if current group is not the weights of a convolutional layer, the K-support norm should be utilized
        if grad_data.dim() != 4:
            return super().LMO(grad_data)
        else:
            tau = self.radius[self.curr_group]
            self.curr_group = (self.curr_group + 1) % len(self.radius)

            # find the indices of the K largest filters by magnitude
            l1_norms = torch.norm(grad_data.view(grad_data.size(0), -1), p=1, dim=1)
            top_k_indices = torch.topk(l1_norms, int(self.K_percentage*len(l1_norms))).indices
            # the change vector is calculated as for the K-sparse polytope, only on filter level
            change_vector = torch.zeros_like(grad_data)
            change_vector[top_k_indices] = - tau * torch.sign(grad_data[top_k_indices])
            return change_vector


# Given the Hyperparameters K_percentage and w, this returns a Group K-support Normball constraint with radius calculated
# from w as described above. K is calculated as K_percentage of the total parameters. On linear layers, K describes the
# number of weights, while on convolutional layers, K describes the number of filters.
class GroupKsupportNormball(KsupportNormball):
    def __init__(self, K_percentage, w):
        super().__init__(K_percentage, w)

    # this code calculating the Group K-support norm of a vector x stems from
    # https://github.com/ZIB-IOL/compression-aware-SFW/blob/main/optimizers/constraints.py#L629. It is based on the
    # definition of the Group k-support norm found in 'The group k-support norm for learning with structured sparsity'
    # -  Rao et al., 2020.
    @staticmethod
    def group_k_support_norm(x, K):
        l1_norms = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
        return KsupportNormball.k_support_norm(l1_norms, K)

    # initialization works just as the K-support norm, except the weights of convolutional layers have to be feasible
    # for the Group K-support norm instead
    def initialize(self, model):
        def diameter_converter(l2diam, num_params):
            return l2diam/2

        def initializer_ksupport(layer, std, tau_weights, tau_biases):
            if isinstance(layer, nn.Conv2d):
                # num_weights has to be the number of filters instead for proper calculation of the K-support norm
                norm = GroupKsupportNormball.group_k_support_norm
                num_weights = layer.weight.shape[0]
            else:
                norm = KsupportNormball.k_support_norm
                num_weights = layer.weight.numel()

            init.normal_(layer.weight, mean=0, std=std)

            # If weights and biases surpass allowed (Group) K-support-norm, project back into the (Group) K-support-Normball
            if norm(layer.weight.data, int(self.K_percentage * num_weights)) > tau_weights:
                layer.weight.data = tau_weights * layer.weight.data / norm(layer.weight.data, int(self.K_percentage * num_weights))

            # Repeat same procedure on biases, should they exist
            if layer.bias is not None:
                num_biases = layer.bias.numel()
                init.normal_(layer.bias, mean=0, std=std)
                if KsupportNormball.k_support_norm(layer.bias.data, int(self.K_percentage * num_biases)) > tau_biases:
                    layer.bias.data = tau_biases * layer.bias.data / KsupportNormball.k_support_norm(layer.bias.data, int(self.K_percentage * num_biases))

        initialize_model(diameter_converter, initializer_ksupport, model, self.w, self.radius)

    def LMO(self, grad_data):
        # if current group is not the weights of a convolutional layer, the K-support norm should be utilized
        if grad_data.dim() != 4:
            return super().LMO(grad_data)
        else:
            tau = self.radius[self.curr_group]
            self.curr_group = (self.curr_group + 1) % len(self.radius)

            # find the indices of the K largest filters by L2-norm
            l2_norms = torch.norm(grad_data.view(grad_data.size(0), -1), p=2, dim=1)
            top_k_indices = torch.topk(l2_norms, int(self.K_percentage*len(l2_norms))).indices
            # the change vector is calculated as for the K-support Normball, only on filter level
            sparse_grad = torch.zeros_like(grad_data)
            sparse_grad[top_k_indices] = grad_data[top_k_indices]
            change_vector = - tau * sparse_grad / (torch.norm(sparse_grad, 2)+1e-8)
            return change_vector