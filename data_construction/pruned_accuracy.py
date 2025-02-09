import numpy as np
from torch import optim
from torch.nn.utils import prune

from constraints import *
from models import ResNet18
from optimizers import MSFWOptimizer
from data_construction.utils import test, setup_conv, train_or_load, set_seed

# pruning percentages by which to prune doing unstructured and structured pruning respectively
pruning_percentages_unstructured = [0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
pruning_percentages_structured = [0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]


# Get the accuracy of ResNet18 on CIFAR10 at pruning percentages defined in pruning_percentages_unstructured when doing
# global unstructured pruning
def unstructured_global_pruning_conv():
    print('Starting global unstructured pruning.')
    # Global unstructured pruning: Prune the smallest weights across all layers
    def pruning_method(model, prune_percentage):
        parameters_to_prune = [(layer, 'weight') for layer in model.modules() if isinstance(layer, (nn.Linear, nn.Conv2d))]
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prune_percentage)
    result = prune_and_record_accuracy(pruning_method, False)
    return result, pruning_percentages_unstructured, "Global unstructured pruning"


# Get the accuracy of ResNet18 on CIFAR10 at pruning percentages defined in pruning_percentages_unstructured when doing
# local unstructured pruning
def unstructured_local_pruning_conv():
    print('Starting local unstructured pruning.')
    # Local unstructured pruning: Prune the smallest weights of every layer
    def pruning_method(model, prune_percentage):
        for layer in model.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                prune.l1_unstructured(layer, name="weight", amount=prune_percentage)
    result = prune_and_record_accuracy(pruning_method, False)
    return result, pruning_percentages_unstructured, "Local unstructured pruning"


# Get the accuracy of ResNet18 on CIFAR10 at pruning percentages defined in pruning_percentages_structured when doing
# structured pruning
def structured_pruning():
    print('Starting structured pruning.')
    # Structured pruning: Prune the filters with smallest L1 norm in every layer
    def pruning_method(model, prune_percentage):
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                prune.ln_structured(layer, name="weight", amount=prune_percentage, n=1, dim=0)
    result = prune_and_record_accuracy(pruning_method, True)
    return result, pruning_percentages_structured, "Structured pruning"


def prune_and_record_accuracy(pruning_method, is_structured):
    seeds = [42,43,44]
    result = {}
    # set device for faster calculation if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Different pruning levels depending on structured and unstructured pruning
    if is_structured:
        pruning_percentages = pruning_percentages_structured
    else:
        pruning_percentages = pruning_percentages_unstructured

    standard_path = './networks/resnet'

    # Setup CIFAR10 data and a method to load ResNet18
    _, train_loader, test_loader = setup_conv(device)
    def setup_model(device):
        return ResNet18().to(device)

    # define constraints for which the pruned accuracy data is wanted
    constraints = {
        'K-sparse polytope': KsparsePolytope(0.1, 15),
        'K-support': KsupportNormball(0.1, 15),
        'Group K-sparse polytope': GroupKsparsePolytope(0.1, 15),
        'Group K-support': GroupKsupportNormball(0.1, 15),
    }
    # this dict contains the weight decay parameters of the two SGD optimizers
    optimizers_SGD = {
        'SGD': 0,
        'SGD weight decay': 1e-2
    }

    num_epochs = 50

    for name, constraint in constraints.items():
        print(f'Getting pruned accuracy data for {name} constraint.')
        acc_arrays = []
        for seed in seeds:
            acc_curr_seed = []
            # For every pruning percentage: Prune model to that degree and record accuracy
            for prune_percentage in pruning_percentages:
                print(f'Pruning {name}/MSFW optimized network by {prune_percentage*100:.2f}% (seed {seed}).')
                model = setup_model(device)
                # If model is not already trained, train it here, else load it
                set_seed(seed)
                constraint.initialize(model)
                optimizer = MSFWOptimizer(model.parameters(), constraint=constraint, lr=0.01)
                train_or_load(model, optimizer, train_loader, f'{standard_path}-{name}-{seed}', num_epochs, device, seed, test_loader)
                # Prune and record accuracy
                pruning_method(model, prune_percentage)
                acc = test(model, test_loader, device)
                print(f'Pruned accuracy: {acc}')
                acc_curr_seed.append(acc)
            acc_arrays.append(acc_curr_seed)
        # Calculate average accuracy at each pruning percentage over all three seeds
        result[name] = np.array(acc_arrays).mean(axis = 0)

    # separate loop for SGD optimizers as they don't have constraints
    for name, weight_decay in optimizers_SGD.items():
        print(f'Getting pruned accuracy data for {name} optimizer.')
        acc_arrays = []
        for seed in seeds:
            acc_curr_seed = []
            # same procedure as for constrained networks
            for prune_percentage in pruning_percentages:
                print(f'Pruning {name} optimized network by {prune_percentage*100:.2f}% (seed {seed}).')
                # If model is not already trained, train it here, else load it
                set_seed(seed)
                model = setup_model(device)
                optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.1, weight_decay=weight_decay)
                train_or_load(model, optimizer, train_loader, f'{standard_path}-{name}-{seed}', num_epochs, device, seed, test_loader)
                # Prune and record accuracy
                pruning_method(model, prune_percentage)
                acc = test(model, test_loader, device)
                print(f'Pruned accuracy: {acc}')
                acc_curr_seed.append(acc)
            acc_arrays.append(acc_curr_seed)
        result[name] = np.array(acc_arrays).mean(axis = 0)

    return result