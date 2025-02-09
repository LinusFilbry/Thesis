import torch
import torch.nn as nn
import torchvision.models as models


# Fully connected neural network with 1 layer to visualize learned representation. Linear regression model for the
# MNIST dataset.
class VisualizationNN(nn.Module):
    def __init__(self):
        super(VisualizationNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        return x


# Simple fully connected neural network with 2 hidden layers, designed for the FashionMNIST dataset.
class LinearNN(nn.Module):
    def __init__(self):
        super(LinearNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Simple convolutional neural network with three hidden convolutional layers followed by two hidden linear layers.
class ConvolutionalNN(nn.Module):
    def __init__(self):
        super(ConvolutionalNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # First conv + pool
        x = self.pool(torch.relu(self.conv2(x)))  # Second conv + pool
        x = self.pool(torch.relu(self.conv3(x)))  # Third conv + pool
        x = x.view(-1, 64 * 4 * 4)  # Flatten the image for the fully connected layer
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ResNet18 architecture adapted to CIFAR 10 dataset.
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=None)
        # There should be 3 input channels with a kernel_size of 3
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # There should be 10 output classes
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)