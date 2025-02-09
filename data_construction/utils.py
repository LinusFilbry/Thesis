import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST

from models import ConvolutionalNN, LinearNN, VisualizationNN


# Set seed for maximum reproducibility. Note that complete reproducibility can not be guaranteed in Pytorch, see
# https://pytorch.org/docs/stable/notes/randomness.html.
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Check if there is a pre-trained version of the model. If so, load it, if not, train the model and save it for future use.
def train_or_load(model, optimizer, train_loader, path, num_epochs, device, seed, test_loader=None):
    # set seed for reproducibility
    set_seed(seed)
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        criterion = nn.CrossEntropyLoss()
        # Give regular updates on the progress
        for epoch in range(num_epochs):
            if epoch > 0 and epoch % 10 == 0:
                print(f'Finished epoch {epoch}.')
                if test_loader is not None:
                    curr_acc = test(model, test_loader, device)
                    print(f'Current accuracy: {curr_acc}')
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        if train_loader is not None:
            final_acc = test(model, test_loader, device)
            print(f'Final accuracy: {final_acc}')
        torch.save(model.state_dict(), path)


# Test the accuracy of the model on the test data from the test_loader
def test(model, test_loader, device):
    model.eval()
    num_correct = 0
    num_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            num_total += labels.size(0)
            num_correct += (predicted == labels).sum().item()

    accuracy = 100 * num_correct / num_total
    return accuracy


# Setup model, train_ and test_loader for linear model/Fashion MNIST
def setup_lin(device):
    # prepare normalization transform for Fashion MNIST. Mean and std were calculated over training data.
    transform_FashionMNIST = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3205,))
    ])

    # prepare MNIST data and linear model
    train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform_FashionMNIST)
    test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform_FashionMNIST)
    train_loader_lin = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader_lin = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model_lin = LinearNN().to(device)

    return model_lin, train_loader_lin, test_loader_lin


# Setup model, train_ and test_loader for convolutional model/CIFAR10
def setup_conv(device):
    # prepare normalization transform for CIFAR10. Mean and std were calculated over training data.
    transform_CIFAR10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615))
    ])

    # prepare CIFAR10 data and convolutional model
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform_CIFAR10)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform_CIFAR10)
    train_loader_conv = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader_conv = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model_conv = ConvolutionalNN().to(device)

    return model_conv, train_loader_conv, test_loader_conv


# Setup model, train_ and test_loader for visualization model/MNIST
def setup_vis(device):
    # prepare normalization transform for MNIST. Mean and std were calculated over training data.
    transform_MNIST = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # prepare MNIST data and linear model
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform_MNIST)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform_MNIST)
    train_loader_lin = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader_lin = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model_vis = VisualizationNN().to(device)

    return model_vis, train_loader_lin, test_loader_lin
