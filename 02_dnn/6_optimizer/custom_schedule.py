import torch


class CustomSchedule(torch.optim.lr_scheduler._LRScheduler):
    """
    Implements the learning rate schedule described in the Transformer paper.
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(
        self, optimizer, d_model: int, warmup_steps: int = 4000
    ):  # Added type hints
        self.d_model = float(d_model)  # Cast to float immediately
        self.warmup_steps = float(warmup_steps)  # Cast to float
        # Call super().__init__ after all self attributes are set if they are used in get_lr
        super().__init__(optimizer)

    def get_lr(self):
        # self.last_epoch stores the current step/epoch count (0-indexed)
        # We need to add 1 because step is typically 1-indexed in LR schedules.
        step = self.last_epoch + 1
        step_f = float(step)  # Ensure step is float for calculations

        # Handle the case where step is 0 to avoid division by zero in rsqrt(0)
        # For the original formula, step should start from 1.
        # If last_epoch starts at -1 (default for _LRScheduler), step will be 0 on first call.
        # For Transformer's LR, step=0 means LR=0.
        if step_f == 0:
            return [0.0] * len(self.optimizer.param_groups)

        # Calculate arg1 and arg2
        arg1 = torch.rsqrt(torch.tensor(step_f))
        arg2 = torch.tensor(step_f) * (self.warmup_steps**-1.5)

        # Calculate the final learning rate
        lr = torch.rsqrt(torch.tensor(self.d_model)) * torch.min(arg1, arg2)

        # Return a list of learning rates, one for each parameter group
        # Since this schedule computes a single LR, we apply it to all groups.
        return [lr.item()] * len(self.optimizer.param_groups)
