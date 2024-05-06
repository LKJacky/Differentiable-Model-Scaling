# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.dist import all_reduce

MASK_THRESHOLD = 0.5


@torch.jit.script
def _dtopk(c: torch.Tensor, a: torch.Tensor, lambda_: float = 1.0):
    y = (c - a) * c.numel() * lambda_
    y = y.sigmoid()
    return y


@torch.jit.script
def differentiable_topk(
    c: torch.Tensor,
    a: torch.Tensor,
    lambda_: float = 1.0,
    normalize: bool = True,
):
    """
    Differentiable top-k operator: Elements with large importance are kept.

    Args:
        c: importance score of elements
        a: pruning ratio of elements
        lambda_: hyper-parameter to control the polarity of the generated mask, default to 1.
        normalize: whether to normalize the importance score, default is True
    Returns:
        soft masks of elements
    """

    if c.numel() == 1:
        return (-(0.5 - a) * lambda_).sigmoid()
    else:
        if normalize:
            c_compare = c.unsqueeze(-1) - c.unsqueeze(-2)  # [N,N]
            c_ = (c_compare >= 0).float().mean(dim=-1)  # normalize to [0,1]
        else:
            c_ = c

        imp = _dtopk(c_, a, lambda_=lambda_)  # [0,1]
        return imp
