from data_construction.utils import setup_conv, setup_lin, test, set_seed
from optimizers import *
from constraints import *


# Returns accuracy data of three constraints at various stages of the training process. Points at which the accuracy
# is calculated can be adjusted with the array testing_epochs; during the first three epochs accuracy always collected
# at various points.
def accuracy_during_training():
    seeds = [42, 43, 44]
    result = []
    # set device for faster calculation if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Constraints for which to get data
    constraints = {
        'L2': LpNormball(2, 15),
        'K-sparse polytope': KsparsePolytope(0.15, 15),
        'K-support': KsupportNormball(0.15, 15)
    }

    # epochs at which to record accuracy
    testing_epochs = [4, 5, 7, 10, 15, 20, 30, 40, 50]
    num_epochs = testing_epochs[-1]

    # iterate over all constraints
    for constraint_name, constraint in constraints.items():
        # Use the convolutional model for the K-support norm ball and the linear model otherwise
        if constraint_name != 'K-support':
            model, train_loader, test_loader = setup_lin(device)
            model_name = 'linear'
        else:
            model, train_loader, test_loader = setup_conv(device)
            model_name = 'convolutional'

        # define all optimizers for which accuracy should be recorded: standard SFW, SFW with rescaling, SFW with momentum, and MSFW
        optimizers = {
            'SFW': SFWOptimizer,
            'SFWR': SFWROptimizer,
            'SFWM': SFWMOptimizer,
            'MSFW': MSFWOptimizer
        }

        # iterate over all optimizers
        for optimizer_name, optimizer_init in optimizers.items():
            accuracies = [0 for _ in range(len(testing_epochs)+14)]
            for seed in seeds:
                print(f'Training and testing {optimizer_name} on {model_name} model with {constraint_name} constraint (seed {seed}).')
                # train network and record accuracies at various stages
                set_seed(seed)
                constraint.initialize(model)
                optimizer = optimizer_init(model.parameters(), constraint)
                acc_curr = train_while_testing(model, optimizer, train_loader, num_epochs, test_loader, testing_epochs, device, seed)

                # save models which can be reused in future methods
                if optimizer_name == 'MSFW':
                    torch.save(model.state_dict(), f'./networks/{model_name}-{constraint_name}-{seed}')
                elif optimizer_name == 'SFW':
                    torch.save(model.state_dict(), f'./networks-SFW/{model_name}-{constraint_name}-{seed}')

                # sum up accuracies over all seeds
                accuracies = [x + y for x, y in zip(acc_curr, accuracies)]

            # record average accuracy over all seeds
            result.append([acc/len(seeds) for acc in accuracies])

    return result, testing_epochs

# Train model using optimizer while recording the accuracy achieved on the test data at several stages
def train_while_testing(model, optimizer, train_loader, num_epochs, test_loader, testing_epochs, device, seed):
    set_seed(seed)
    criterion = nn.CrossEntropyLoss()
    accuracies = []

    # record detailed accuracy in the first three epochs (first at every eighth, then at every forth, then at every half)
    for i in range(0,3):
        accuracies += epoch_train_and_test(model, optimizer, criterion, train_loader, test_loader, device, 1/2**(3-i))

    # record accuracy once at every given testing_epoch after the first three epochs
    for epoch in range(3, num_epochs):
        if epoch in testing_epochs:
            acc = test(model, test_loader, device)
            print(f'Accuracy after {epoch} epochs: {acc}')
            accuracies.append(acc)
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # append final accuracy
    acc = test(model, test_loader, device)
    print(f'Final accuracy after {num_epochs} epochs: {acc}')
    accuracies.append(acc)
    return accuracies

# train for one epoch and record test accuracy at every $fraction*100 % of the testing data
def epoch_train_and_test(model, optimizer, criterion, train_loader, test_loader, device, frac):
    total_iterations = len(train_loader)
    interval = int(total_iterations * frac)
    accuracies = []

    for i, (images, labels) in enumerate(train_loader):
        if (i + 1) % interval == 0:
            accuracies.append(test(model, test_loader, device))
        model.train()
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return accuracies