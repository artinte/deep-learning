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


class WeightDecay:
    def __init__(
        self,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        absolute: bool = False,
    ):
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.absolute = absolute
        self.weight_decouple = weight_decouple
        self.weight_decay = weight_decay

    def defaults(self):
        return dict(weight_decay=self.weight_decay)

    def __call__(
        self, param: torch.nn.Parameter, grad: torch.Tensor, group: Dict[str, any]
    ):
        if self.weight_decouple:
            if self.absolute:
                param.data.mul_(1.0 - group["weight_decay"])
            else:
                param.data.mul_(1.0 - group["lr"] * group["weight_decay"])

            return grad
        else:
            if group["weight_decay"] != 0:
                return grad.add(param.data, alpha=group["weight_decay"])
            else:
                return grad
