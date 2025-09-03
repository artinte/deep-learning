import torch
import math
from typing import Optional


def create_mask(src, tgt, pad_token_id, device, batch_first):
    if batch_first:
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
    else:
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

    tgt_mask = (
        torch.nn.Transformer.generate_square_subsequent_mask(tgt_seq_len)
        .to(device)
        .bool()
    )
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    if batch_first:
        src_padding_mask = src == pad_token_id
        tgt_padding_mask = tgt == pad_token_id
    else:
        src_padding_mask = (src == pad_token_id).transpose(0, 1)
        tgt_padding_mask = (tgt == pad_token_id).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def _calculate_fan_in_and_fan_out(tensor: torch.Tensor) -> tuple[int, int]:
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


# These no_grad_* functions are necessary as wrappers around the parts of these
# functions that use `with torch.no_grad()`. The JIT doesn't support context
# managers, so these need to be implemented as builtins. Using these wrappers
# lets us keep those builtins small and reusable.
def _no_grad_uniform_(
    tensor: torch.Tensor,
    a: float,
    b: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    with torch.no_grad():
        return tensor.uniform_(a, b, generator=generator)


def _no_grad_normal_(
    tensor: torch.Tensor,
    mean: float,
    std: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    with torch.no_grad():
        return tensor.normal_(mean, std, generator=generator)


def xavier_uniform_(
    tensor: torch.Tensor,
    gain: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    r"""Fill the input `Tensor` with values using a Xavier uniform distribution.

    The method is described in `Understanding the difficulty of training
    deep feedforward neural networks` - Glorot, X. & Bengio, Y. (2010).
    The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain("relu"))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a, generator)


def xavier_normal_(
    tensor: torch.Tensor,
    gain: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    r"""Fill the input `Tensor` with values using a Xavier normal distribution.

    The method is described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010). The resulting tensor
    will have values sampled from :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    return _no_grad_normal_(tensor, 0.0, std, generator)
