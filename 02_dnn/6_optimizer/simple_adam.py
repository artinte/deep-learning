import math
import torch
from typing import Dict, Tuple, Any


class GenericAdaptiveOptimizer(torch.optim.optimizer.Optimizer):
    def __init__(
        self,
        params,
        defaults: Dict[str, Any],
        lr: float,
        betas: Tuple[float, float],
        eps: float,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults.update(dict(lr=lr, betas=betas, eps=eps))
        super().__init__(params, defaults)

    def init_state(
        self, state: Dict[str, any], group: Dict[str, any], param: torch.nn.Parameter
    ):
        pass

    def step_param(
        self,
        state: Dict[str, any],
        group: Dict[str, any],
        grad: torch.Tensor,
        param: torch.Tensor,
    ):
        pass

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "GenericAdaptiveOptimizer does not support sparse gradients,"
                        " please consider SparseAdam instead"
                    )

                state = self.state[param]

                if len(state) == 0:
                    self.init_state(state, group, param)

                self.step_param(state, group, grad, param)

        return loss
