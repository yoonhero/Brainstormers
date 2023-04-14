import torch
import torch.nn as nn

class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Because we are saving one of the inputs use `save_for_backward`
        # Save non-tensors and non-inputs/non-outputs directly on ctx
        ctx.save_for_backward(x)
        return x**2

    @staticmethod
    def backward(ctx, grad_out):
        # A function support double backward automatically if autograd
        # is able to record the computations performed in backward
        x, = ctx.saved_tensors
        return grad_out * 2 * x

class SquareLayer(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return x ** 2

    def backward_pass(self, y, dy, f_args={}, g_args={}):
        torch.autograd.backward(y, dy, retain_graph=True)
        return dy


square = SquareLayer()
x = torch.tensor(3., requires_grad=True).clone()

out = square(x)

out.backward()

print(x)
