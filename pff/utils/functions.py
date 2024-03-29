import torch
import torch.nn.functional as F

# loeweX @ https://github.com/loeweX/Forward-Forward/blob/main/src/ff_model.py#L161
class ReLU_full_grad(torch.autograd.Function):
    """ ReLU activation function that passes through the gradient irrespective of its input value."""

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

def my_relu(x):
    return ReLU_full_grad.apply(x)
    
def goodness(z, threshold=3.0, alt=False):
    if alt:
        return threshold - z.square().sum(dim=1)
    
    return z.square().sum(dim=1) - threshold
    