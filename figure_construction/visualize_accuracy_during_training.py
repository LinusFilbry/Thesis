import matplotlib.pyplot as plt


# In combination with 'accuracy_during_testing', produces figures 5.1 and 5.2.
def visualize_accuracy_curve(data, testing_epochs, full_graph):
    # Assumes that data is created using accuracy_during_testing and therefore adds the corresponding entries during
    # the first three epochs.
    testing_epochs = [i/8 for i in range(1, 9)] + [1+i/4 for i in range(1, 5)] + [2+i/2 for i in range(1,3)] + testing_epochs
    fig = plt.figure(figsize=(15, 5))

    # ax1 and ax2 are of the linear model and can therefore share a y-axis
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3)

    axes = [ax1, ax2, ax3]
    titles = ['L2 constraint - linear model', 'K-sparse polytope constraint - linear model',
              'K-support norm constraint - convolutional model']

    for i, ax in enumerate(axes):
        sfw_data = data[4 * i]
        sfwr_data = data[4 * i + 1]
        sfwm_data = data[4 * i + 2]
        msfw_data = data[4 * i + 3]

        ax.plot(testing_epochs, sfw_data, '-o', label='SFW', color='blue')
        ax.plot(testing_epochs, msfw_data, '-o', label='MSFW', color='red')

        if full_graph:
            ax.plot(testing_epochs, sfwr_data, '-o', label='SFW with rescaling', color='green')
            ax.plot(testing_epochs, sfwm_data, '-o', label='SFW with momentum', color='yellow')

        ax.set_title(titles[i])
        ax.set_xlabel('Epoch')
        if i == 0:
            ax.set_ylabel('Accuracy')
        ax.legend(loc='lower right')
        ax.grid(True)

    plt.tight_layout()
    if full_graph:
        save_path = "./plots/accuracy_curve_during_training_detailed.png"
    else:
        save_path = "./plots/accuracy_curve_during_training.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()