from torch.nn import functional as F
from torch.autograd import Function, gradcheck
import torch

import Ncc3d

class NCC3DFunction(Function):

    @staticmethod
    def forward(ctx, z_block, x_block):

        xc_block = Ncc3d.forward_naive(z_block, x_block)

        ctx.save_for_backward(z_block, x_block)

        return xc_block

    @staticmethod
    def backward(ctx, g_xc_block):

        z_block, x_block = ctx.save_tensors

        g_z_block, g_x_block = Ncc3d.backward_naive(z_block, x_block, g_xc_block)

        return  g_z_block, g_x_block

class ncc3d(torch.nn.Module):
    def __init__(self):
        super(ncc3d, self).__init__()

    def forward(self, z_block, x_block):

        xc_block = NCC3DFunction.apply(z_block, x_block)

        return xc_block