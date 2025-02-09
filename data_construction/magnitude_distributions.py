import numpy as np
from torch import optim

from constraints import *
from data_construction.utils import train_or_load, setup_lin, setup_conv, set_seed
from models import ConvolutionalNN, LinearNN
from optimizers import MSFWOptimizer

# Magnitude ranges in which to separate weight absolute ranges and filter L1-norms respectively
weight_magnitude_ranges = [float('inf')] + [10 ** -i for i in range(6)] + [0]
filter_magnitude_ranges = [float('inf')] + [30, 20, 15, 10, 8, 5, 3, 2, 1, 0.5, 0.1] + [0]


# Get the distribution of absolute values of weights into the ranges defined in weight_magnitude_ranges when training
# with different optimizers on the linear model
def weight_magnitude_distribution_linear():
    print('Starting weight magnitude analysis of linear model.')
    return magnitude_distribution(False, False)


# Get the distribution of absolute values of weights into the ranges defined in weight_magnitude_ranges when training
# with different optimizers on the convolutional model
def weight_magnitude_distribution_convolutional():
    print('Starting weight magnitude analysis of convolutional model.')
    return magnitude_distribution(True, False)


# Get the distribution of l1-norms of filters into the ranges defined in filter_magnitude_ranges when training
# with different optimizers on the convolutional model
def filter_magnitude_distribution():
    print('Starting filter L1-norm analysis of convolutional model.')
    return magnitude_distribution(True, True)


def magnitude_distribution(is_convolutional, is_structured):
    seeds = [42,43,44]
    result = {}
    # set device for faster calculation if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define constraints for which the weight distribution data is wanted
    constraints = {
        'L2': LpNormball(2, 15),
        'L5': LpNormball(5, 15),
        'K-sparse polytope': KsparsePolytope(0.15, 15),
        'K-support': KsupportNormball(0.15, 15),
    }
    constraints_conv = {
        'Group K-sparse polytope': GroupKsparsePolytope(0.15, 15),
        'Group K-support': GroupKsupportNormball(0.15, 15),
    }

    # setup method for linear or convolutional model
    if is_convolutional:
        model_name = 'convolutional'
        _, train_loader, test_loader = setup_conv(device)
        def setup_model(device):
            return ConvolutionalNN().to(device)
        # analyze group constraints only on convolutional model
        constraints.update(constraints_conv)
    else:
        model_name = 'linear'
        _, train_loader, test_loader = setup_lin(device)
        def setup_model(device):
            return LinearNN().to(device)

    # setup method for filter or weight magnitude distribution
    if is_structured:
        metric_name = 'Filter'
        magnitude_ranges = filter_magnitude_ranges
        magnitude_analyzer = analyze_filter_l1_norms
    else:
        metric_name = 'Weight'
        magnitude_ranges = weight_magnitude_ranges
        magnitude_analyzer = analyze_weight_magnitudes

    # Always include SGD optimizers, this dict holds their weight decay parameters
    optimizers_SGD = {
        'SGD': 0,
        'SGD weight decay': 1e-2
    }

    num_epochs = 50

    for name, constraint in constraints.items():
        distribution_arrays = []
        for seed in seeds:
            print(f'Analyzing distribution of network optimized with MSFW and constraint {name} (seed {seed}).')
            model = setup_model(device)
            # If model is not already trained, train it here, else load it
            set_seed(seed)
            constraint.initialize(model)
            optimizer = MSFWOptimizer(model.parameters(), constraint=constraint)
            train_or_load(model, optimizer, train_loader, f'./networks/{model_name}-{name}-{seed}', num_epochs, device, seed, test_loader)
            # Get the distribution according to the provided magnitude analyzer function
            distribution_arrays.append(magnitude_analyzer(model, magnitude_ranges))
        # calculate the mean percentage of parameters belonging to each range
        result[name] = np.array(distribution_arrays).mean(axis = 0)

    # separate loop for SGD optimizers as they don't have constraints
    for name, weight_decay in optimizers_SGD.items():
        distribution_arrays = []
        for seed in seeds:
            print(f'Analyzing distribution of network optimized with {name} (seed {seed}).')
            # If model is not already trained, train it here, else load it
            set_seed(seed)
            model = setup_model(device)
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.1, weight_decay=weight_decay)
            train_or_load(model, optimizer, train_loader, f'./networks/{model_name}-{name}-{seed}', num_epochs, device, seed, test_loader)
            # Get the distribution according to the provided magnitude analyzer function
            distribution_arrays.append(magnitude_analyzer(model, magnitude_ranges))
        # calculate the mean percentage of parameters belonging to each range
        result[name] = np.array(distribution_arrays).mean(axis = 0)

    return magnitude_ranges, result, f"{metric_name} magnitude distribution on {model_name} model"


def analyze_weight_magnitudes(model, magnitude_ranges):
    magnitude_counts = [0] * (len(magnitude_ranges) - 1)
    total_weights = 0

    for param in model.parameters():
        # Include only weights
        if param.dim() == 2 or param.dim() == 4:
            weights = param.detach().cpu().numpy().flatten()
            total_weights += len(weights)

            for i in range(len(magnitude_ranges) - 1):
                lower = magnitude_ranges[i + 1]
                upper = magnitude_ranges[i]
                # magnitude_counts[i] is the number of weights smaller than magnitude_ranges[i] and larger than magnitude_ranges[i + 1]
                magnitude_counts[i] += np.sum((np.abs(weights) > lower) & (np.abs(weights) <= upper))

    # return percentages instead of absolute numbers
    magnitude_percentages = [count / total_weights for count in magnitude_counts]
    return magnitude_percentages


def analyze_filter_l1_norms(model, magnitude_ranges):
    magnitude_counts = [0] * (len(magnitude_ranges) - 1)
    total_filters = 0

    for param in model.parameters():
        # Only convolutional layers have filters
        if param.dim() == 4:
            l1_norms = torch.norm(param.detach().cpu().view(param.size(0), -1), p=1, dim=1).numpy()
            total_filters += len(l1_norms)

            for i in range(len(magnitude_ranges) - 1):
                lower = magnitude_ranges[i + 1]
                upper = magnitude_ranges[i]
                # Same logic as for weights
                magnitude_counts[i] += np.sum((l1_norms > lower) & (l1_norms <= upper))

    # return percentages instead of absolute numbers
    magnitude_percentages = [count / total_filters for count in magnitude_counts]
    return magnitude_percentages