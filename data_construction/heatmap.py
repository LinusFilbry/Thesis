from torch import optim

from constraints import *
from models import VisualizationNN
from optimizers import *
from data_construction.utils import train_or_load, setup_vis, set_seed, test


# returns the weights of a neural network without hidden layer, i.e. linear regression model, on MNIST for several
# different optimizers/constraints
def weight_data_from_visualization_model():
    seed = 42
    layers_dict = {}
    # set device for faster calculation if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup visualization network and MNIST data
    _, train_loader_vis, test_loader_vis = setup_vis(device)

    # define constraints for which the weight data is wanted
    constraints = {
        'L2': LpNormball(2, 50),
        'Linf': LpNormball('inf', 50),
        'K-sparse polytope': KsparsePolytope(0.1, 50),
        'K-support': KsupportNormball(0.1, 50),
    }
    # this dict contains the weight decay parameters of the two SGD optimizers
    optimizers_SGD = {
        'SGD': 0,
        'SGD weight decay': 1e-2
    }

    num_epochs = 50

    for name, constraint in constraints.items():
        print(f'Training the visualization network with MSFW and constraint {name}.')
        # If model is not already trained, train it here, else load it
        model_vis = VisualizationNN().to(device)
        set_seed(seed)
        constraint.initialize(model_vis)
        optimizer = MSFWOptimizer(model_vis.parameters(), constraint=constraint)
        train_or_load(model_vis, optimizer, train_loader_vis, f'./networks/visualization-{name}', num_epochs, device, seed, test_loader_vis)
        print(f'Accuracy: {test(model_vis, test_loader_vis, device)}%')
        # save the weights to visualize them later
        layers_dict[name] = model_vis.fc1

    # SGD optimizers don't have constraints and initialization functions, so have to be treated in a separate loop
    for name, weight_decay in optimizers_SGD.items():
        print(f'Training the visualization network with {name}.')
        # If model is not already trained, train it here, else load it
        set_seed(seed)
        model_vis = VisualizationNN().to(device)
        optimizer = optim.SGD(model_vis.parameters(), lr=0.01, momentum=0.1, weight_decay=weight_decay)
        train_or_load(model_vis, optimizer, train_loader_vis, f'./networks/visualization-{name}', num_epochs, device, seed, test_loader_vis)
        print(f'Accuracy: {test(model_vis, test_loader_vis, device)}%')
        # save the weights to visualize them later
        layers_dict[name] = model_vis.fc1

    return layers_dict