import torch
from torch import optim

from data_construction.utils import setup_lin, setup_conv, train_or_load, test, set_seed
from models import LinearNN, ConvolutionalNN


# Record accuracy of SGD and SGD with weight decay on linear and convolutional model
def train_SGD():
    seeds = [42,43,44]

    # set device for faster calculation if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define models for which accuracy data should be recorded
    models = ['linear', 'convolutional']

    # define weight decay value for which SGD optimizers should be tested
    optimizers_SGD = {
        'SGD': 0,
        'SGD weight decay': 1e-2
    }

    num_epochs = 50

    for model_name in models:
        # setup linear or convolutional model and data
        if model_name == 'convolutional':
            _, train_loader, test_loader = setup_conv(device)
            def setup_model(device):
                return ConvolutionalNN().to(device)
        else:
            _, train_loader, test_loader = setup_lin(device)
            def setup_model(device):
                return LinearNN().to(device)

        # iterate through both SGD optimizers
        for name, weight_decay in optimizers_SGD.items():
            curr_acc = 0
            # define optimizer with appropriate weight decay and record average accuracy over all seeds
            for seed in seeds:
                print(f'Training and testing {name} on {model_name} model (seed {seed}).')
                set_seed(seed)
                model = setup_model(device)
                optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.1, weight_decay=weight_decay)
                train_or_load(model, optimizer, train_loader, f'./networks/{model_name}-{name}-{seed}', num_epochs, device, seed, test_loader)
                curr_acc += test(model, test_loader, device)
            # Print out observed accuracy
            print(f'{name} accuracy on {model_name} model:{curr_acc/len(seeds):.2f}%')