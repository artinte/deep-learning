from matplotlib import pyplot
import torch

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return torch.nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return torch.nn.functional.softmax(X.reshape(shape), dim=-1)


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