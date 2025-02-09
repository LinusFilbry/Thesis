import matplotlib.pyplot as plt
import numpy as np


# Plots the provided magnitude distributions by 'magnitude_distribution' against the magnitude ranges for which they
# were achieved to produce figures 5.5 and 5.6.
def visualize_magnitude_distribution(ranges, data, title):
    x_positions = np.arange(len(ranges) - 1) + 0.5

    plt.figure(figsize=(10, 6))

    for name, values in data.items():
        plt.plot(x_positions, values, '-o', label=name)

    plt.xticks(np.arange(len(ranges)), ranges)

    plt.xlabel("Ranges")
    plt.ylabel("Percentage")
    plt.title(title, fontsize=20)
    plt.legend()

    ax = plt.gca()
    # Adjust y-axis to display percentage of weights/filters in every range
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))

    plt.grid(visible=True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"./plots/{title}.png", bbox_inches="tight")
    plt.show()
