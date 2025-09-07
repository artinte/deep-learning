import torch
import math
from typing import Optional


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def _in_projection_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor] = None,
) -> list[torch.Tensor]:
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            proj = torch.nn.functional.linear(q, w, b)
            # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
            proj = (
                proj.unflatten(-1, (3, E))
                .unsqueeze(0)
                .transpose(0, -2)
                .squeeze(-2)
                .contiguous()
            )
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = torch.nn.functional.linear(q, w_q, b_q)
            kv_proj = torch.nn.functional.linear(k, w_kv, b_kv)
            # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
            kv_proj = (
                kv_proj.unflatten(-1, (2, E))
                .unsqueeze(0)
                .transpose(0, -2)
                .squeeze(-2)
                .contiguous()
            )
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return (
            torch.nn.functional.linear(q, w_q, b_q),
            torch.nn.functional.linear(k, w_k, b_k),
            torch.nn.functional.linear(v, w_v, b_v),
        )


def _canonical_mask(
    mask: Optional[torch.Tensor],
    mask_name: str,
    target_type: torch.dtype,
) -> Optional[torch.Tensor]:
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported"
            )
        if not _mask_is_float:
            mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(
                mask, float("-inf")
            )
    return mask


class MultiheadAttentionScratch(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        assert embed_dim > 0 and num_heads > 0
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        self.in_proj_weight = torch.nn.Parameter(
            torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
        )
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, **factory_kwargs)

        torch.nn.init.xavier_uniform_(self.in_proj_weight)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        key_padding_mask = _canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            target_type=query.dtype,
        )

        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            target_type=query.dtype,
        )

        if self.batch_first:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        attn_output = None
        attn_output_weights = None
        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        if is_causal and attn_mask is None:
            raise RuntimeError(
                "Need attn_mask if specifying the is_causal hint. "
                "You may use the Transformer module method "
                "`generate_square_subsequent_mask` to create this mask."
            )

        if is_causal and key_padding_mask is None and not need_weights:
            # when we have a kpm or need weights, we need attn_mask
            # Otherwise, we use the is_causal hint go as is_causal
            # indicator to SDPA.
            attn_mask = None
        else:
            attn_mask = _canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                target_type=query.dtype,
            )

            if key_padding_mask is not None:
                # We have the attn_mask, and use that to merge kpm into it.
                # Turn off use of is_causal hint, as the merged mask is no
                # longer causal.
                is_causal = False

        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

        q, k, v = _in_projection_packed(query, key, value, self.in_proj_weight)

        # prep attention mask
        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )

        # reshape q, k, v for multihead attention and make them batch first
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(bsz * self.num_heads, 1, src_len)
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        # adjust dropout probability
        if not self.training:
            self.dropout = 0.0

        # (deep breath) calculate attention and out projection
        if need_weights:
            _B, _Nt, E = q.shape
            q_scaled = q * math.sqrt(1.0 / float(E))

            assert not (
                is_causal and attn_mask is None
            ), "FIXME: is_causal not implemented for need_weights"

            if attn_mask is not None:
                attn_output_weights = torch.baddbmm(
                    attn_mask, q_scaled, k.transpose(-2, -1)
                )
            else:
                attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = torch.nn.functional.softmax(
                attn_output_weights, dim=-1
            )
            if self.dropout > 0.0:
                attn_output_weights = torch.nn.functional.dropout(
                    attn_output_weights, p=self.dropout
                )

            attn_output = torch.bmm(attn_output_weights, v)

            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            )
            attn_output = torch.nn.functional.linear(
                attn_output, self.out_proj.weight, self.out_proj.bias
            )
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            # optionally average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
        else:
            # attn_mask can be either (L,S) or (N*num_heads, L, S)
            # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
            # in order to match the input for SDPA of (N, num_heads, L, S)
            if attn_mask is not None:
                if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(0)
                else:
                    attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

            q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
            k = k.view(bsz, self.num_heads, src_len, self.head_dim)
            v = v.view(bsz, self.num_heads, src_len, self.head_dim)

            attn_output = scaled_dot_product_attention(
                q, k, v, attn_mask, self.dropout, is_causal
            )
            attn_output = (
                attn_output.permute(2, 0, 1, 3)
                .contiguous()
                .view(bsz * tgt_len, embed_dim)
            )

            attn_output = torch.nn.functional.linear(
                attn_output, self.out_proj.weight, self.out_proj.bias
            )
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
