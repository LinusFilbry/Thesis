from collections import defaultdict

from data_construction.utils import setup_conv, setup_lin, train_or_load, test, set_seed
from optimizers import *
from constraints import *


# Creates data of the final test accuracy of SFW and MSFW across various constraints and linear and convolutional model.
def acc_trained_SFW_MSFW():
    seeds = [42, 43, 44]
    data = defaultdict(list)
    # set device for faster calculation if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define constraints for which accuracy data should be recorded
    constraints = {
        'L1': LpNormball(1, 15),
        'L2': LpNormball(2, 15),
        'L5': LpNormball(5, 15),
        'K-sparse polytope': KsparsePolytope(0.15, 15),
        'K-support': KsupportNormball(0.15, 15)
    }
    # Group constraints equal their regular forms on linear models, so they should only be trained on the convolutional model
    constraints_conv = {
        'Group K-sparse polytope': GroupKsparsePolytope(0.15, 15),
        'Group K-support': GroupKsupportNormball(0.15, 15)
    }

    # define models for which accuracy data should be recorded
    models = {
        'linear': setup_lin,
        'convolutional': setup_conv
    }

    # define optimizers for which accuracy data should be recorded
    optimizers = {
        'SFW': SFWOptimizer,
        'MSFW': MSFWOptimizer
    }

    num_epochs = 50

    for model_name in models.keys():
        # setup linear or convolutional model and data
        model, train_loader, test_loader = models[model_name](device)

        # When training on linear models is done, include convolutional models
        if model_name == 'convolutional':
            constraints.update(constraints_conv)
            # Place -1 in the spots for linear models for group constraints so the fields get crossed out in visualization.
            for constraint_name in constraints_conv.keys():
                data[constraint_name].extend([-1, -1])

        # iterate through all constraints
        for constraint_name, constraint in constraints.items():
            # iterate through both optimizers
            for optimizer_name, optimizer_init in optimizers.items():
                # save final networks in different directories
                if optimizer_name == 'MSFW':
                    standard_path = './networks'
                else:
                    standard_path = './networks-SFW'

                # record average accuracy for combination of model, constraint and optimizer
                curr_acc = 0
                for seed in seeds:
                    print(f'Recording accuracy of optimizer {optimizer_name} with constraint {constraint_name} on '
                          f'{model_name} model (seed {seed}).')
                    set_seed(seed)
                    constraint.initialize(model)
                    optimizer = optimizer_init(model.parameters(), constraint)
                    train_or_load(model, optimizer, train_loader, f'{standard_path}/{model_name}-{constraint_name}-{seed}',
                                  num_epochs, device, seed, test_loader)
                    curr_acc += test(model, test_loader, device)
                data[constraint_name].append(curr_acc/len(seeds))

    return data