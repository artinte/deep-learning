import torch


# To see the gradients, we need to enable gradient tracking and perform a backward pass.
input = torch.randn(10, requires_grad=True)
print("Input:", input)


def relu_scratch(input):
    # Custom implementation of ReLU activation function.
    return torch.maximum(torch.tensor(0.0), input)


output_scratch = relu_scratch(input)
print("Output after ReLU activation:", output_scratch)

relu_torch = torch.nn.ReLU()
output = relu_torch(input)
print("Output after ReLU activation:", output)

gradients = torch.autograd.grad(outputs=output.sum(), inputs=input)[0]
print("Gradients:", gradients)
