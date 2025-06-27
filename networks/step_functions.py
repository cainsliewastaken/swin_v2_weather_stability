import torch
import torch.nn as nn


class Forward_Euler_Step(nn.Module):
    """Wrapper for the forward Euler integration constraint.
    Note, the input and output need to be within the same space"""
    def __init__(self, params, model_handle, dt_tensor = 1):
        super(Forward_Euler_Step, self).__init__()
        self.model = model_handle(params)
        self.dt_tens = dt_tensor

    def forward(self, inp, coszen=None):
        return inp + self.dt_tens * self.model(inp)
