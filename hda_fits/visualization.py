import matplotlib.pyplot as plt


def show_som(som, figsize=(12, 12), cmap="jet"):
    """
    Function to visualize the SOM nodes in its
    respective grid layout.

    Will currently only support 2D-layouts
    """
    width, height, _ = som.layout
    fig, axes = plt.subplots(height, width, figsize=figsize)
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            ax.axis("off")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.imshow(som.get_node(i, j).T, cmap=cmap)

    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    return fig
