import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# In combination with 'weight_data_from_visualization_model', produces figure 5.4.
def visualize_heatmap(layers_dict):
    # The output neurons/classes for which to visualize the weights
    numbers = [0,2,7]

    # define custom color map: positive values will be green, negative values red and values close to 0 black
    colors = [
        (1, 0, 0),
        (0, 0, 0),
        (0, 1, 0),
    ]
    cmap = LinearSegmentedColormap.from_list("custom_red_black_green", colors, N=256)

    num_layers = len(layers_dict)
    num_numbers = len(numbers)

    plt.figure(figsize=(4 * num_layers, 4 * num_numbers))

    for row_idx, num in enumerate(numbers):
        for col_idx, (layer_name, fc_layer) in enumerate(layers_dict.items()):
            fc_weights = fc_layer.weight.detach().cpu().numpy()

            # reshape so the weights are in the same form as the original picture
            weight_image = fc_weights[num].reshape(28, 28)
            ax = plt.subplot(num_numbers, num_layers, row_idx * num_layers + col_idx + 1)
            ax.imshow(weight_image, cmap=cmap, interpolation='nearest')
            # Column title is the constraint/optimizer whose weights are displayed
            if row_idx == 0:
                ax.set_title(layer_name, fontsize=30)
            # Row title are the numbers/output neurons whose weights are displayed
            if col_idx == 0:
                ax.set_ylabel(f"\'{num}\'", fontsize=30, labelpad=30, rotation=0)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.savefig("./plots/weight-heatmap.png", bbox_inches="tight")
    plt.show()