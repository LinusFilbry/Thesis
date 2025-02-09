import matplotlib.pyplot as plt


# In combination with 'acc_trained_SFW_MSFW', produces figure 5.3.
def visualize_accuracy_table(data):
    rows = []
    for name, values in data.items():
        row = []
        for value in values:
            if value == -1:
                row.append("\u0336\u0336\u0336")  # Crossed out
            else:
                row.append(f"{value:.2f}%")
        rows.append(row)

    columns = ['Acc: SFW - Linear', 'Acc: MSFW - Linear', 'Acc: SFW - Conv', 'Acc: MSFW - Conv']

    fig, ax = plt.subplots(figsize=(10, 1.5+len(data)/4))

    ax.axis('off')

    table = ax.table(cellText=rows,
                     colLabels=columns,
                     rowLabels=list(data.keys()),
                     loc='center',
                     cellLoc='center',
                     rowLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    for i in range(len(data)):
        table[(i+1, -1)].set_text_props(weight='bold')

    for j in range(len(columns)):
        header_cell = table[(0, j)]
        header_cell.set_text_props(weight='bold')
        header_cell.set_fontsize(14)
        header_cell.set_edgecolor('grey')

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_fontsize(14)
        else:
            cell.set_edgecolor('gray')
            cell.set_text_props(fontsize=12)

    plt.savefig("./plots/accuracy_table.png", bbox_inches="tight")

    plt.show()