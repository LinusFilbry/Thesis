import math
import torch.nn as nn


# initialize the given model using the given methods and hyperparameters
def initialize_model(diam_converter, initializer, model, w, radius):
    for layer in model.modules():
        # only initialize fully connected and convolutional layers, for example not pooling layers
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            std = get_std_he_init(layer)
            num_weights = layer.weight.numel()

            tau_weights = calculate_radius(w, std, num_weights, diam_converter)
            # add the calculated radius to the given array, so it can later be used in the LMO during the
            # optimization process
            radius.append(tau_weights)

            # repeat the procedure for biases, if they exist for the layer
            if layer.bias is not None:
                num_biases = layer.bias.numel()
                tau_biases = calculate_radius(w, std, num_biases, diam_converter)
                radius.append(tau_biases)
            else:
                tau_biases = None

            initializer(layer, std, tau_weights, tau_biases)

        # Batch normalization values get fixed initialization values and radii
        if isinstance(layer, nn.BatchNorm2d):
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)
            radius.append(5)
            radius.append(5)


# The standard deviation with which to initialize parameters is calculated in accordance with He-Initialization.
def get_std_he_init(layer):
    if layer.weight.dim() == 4:
        c_in = layer.in_channels
        filter_height, filter_width = layer.kernel_size
        return (2.0 / (c_in * filter_height * filter_width)) ** 0.5
    else:
        return (2.0 / layer.weight.shape[1]) ** 0.5


# The radius is calculated so that the L2-Diameter will be 2*w*E[|x|] where E[|x|] is the expected L2 norm of the
# parameters after initialization along normal distribution with mean 0 and standard deviation std.
def calculate_radius(w, std, num_params, l2_diameter_converter):
    # The formula for the expected value found on page 6 in ['Deep Neural Network Training with Frank–Wolfe' - Pokutta
    # et al., 2020] leads to overflows, with is why it is approximated using Stirling's approximation. As the number of
    # parameters is always st least 10 and only an approximate result is needed, this is sufficient.
    expected_value = std * math.sqrt(num_params/2)
    l2diam = 2*w*expected_value
    return l2_diameter_converter(l2diam, num_params)