import matplotlib.pyplot as plt


# Plots the provided magnitude distributions by 'prune_and_record_accuracy' against the pruning ratios for which they
# were achieved to produce figures 5.7 and 5.8.
def visualize_pruning_accuracy_curve(data, percentages, title):
    plt.figure(figsize=(10, 6))

    for name, values in data.items():
        plt.plot(percentages, values, '-o', label=name)

    plt.xlabel("Pruning percentage")
    plt.ylabel("Accuracy")
    plt.title(title, fontsize=20)
    plt.legend(loc='lower left')

    ax = plt.gca()
    # Adjust the axis to display correct percentage values
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    plt.grid(visible=True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"./plots/{title}.png", bbox_inches="tight")
    plt.show()