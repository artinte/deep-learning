from matplotlib import pyplot

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(5, 5), cmap='viridis'):
    """
    matrices: (batch, heads, query_len, key_len)
    """
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = pyplot.subplots(
        num_rows, num_cols, figsize=figsize, squeeze=False)

    for i in range(num_rows):
        for j in range(num_cols):
            ax = axes[i][j]
            im = ax.imshow(matrices[i][j].detach().cpu().numpy(), cmap=cmap)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
            fig.colorbar(im, ax=ax)
    pyplot.tight_layout()
    pyplot.show()