class CustomOptimizer:
    """
    A custom optimizer that implements the learning rate schedule from the
    Transformer paper "Attention Is All You Need."

    It wraps a standard optimizer and dynamically adjusts the learning rate.
    """

    def __init__(self, optimizer, model_dim, warmup_steps):
        self.optimizer = optimizer
        self.model_dim = model_dim
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        """
        Performs a single optimization step and updates the learning rate.
        """
        self.step_num += 1
        lr = self.get_learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_learning_rate(self):
        """
        Calculates the current learning rate based on the paper's formula.
        """
        arg1 = self.step_num**-0.5
        arg2 = self.step_num * (self.warmup_steps**-1.5)
        return (self.model_dim**-0.5) * min(arg1, arg2)
