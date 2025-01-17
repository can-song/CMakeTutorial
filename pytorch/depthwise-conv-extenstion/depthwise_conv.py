import math
import collections
from itertools import repeat

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.parameter import Parameter
import depthwise_conv_cuda

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)


class DepthWiseConvFunction(Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, kernel_size, stride=1, padding=0, use_bias=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        assert inputs.size(1) == weight.size(0)

        if use_bias:
            ctx.save_for_backward(*[inputs, weight, bias])
            # ctx.save_for_backward(inputs, weight, bias)
        else:
            bias = torch.zeros(1, dtype=weight.dtype, device=weight.device)
            ctx.save_for_backward(*[inputs, weight, bias])

        ctx.use_bias = use_bias
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding

        output = depthwise_conv_cuda.forward(inputs, weight, bias, kernel_size[0], kernel_size[1],
                                             stride[0], stride[1], padding[0], padding[1], use_bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        use_bias = ctx.use_bias
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding
        if use_bias:
            inputs, weight, bias = ctx.saved_variables
            output_grads = depthwise_conv_cuda.backward(grad_output.contiguous(), inputs, weight, bias, kernel_size[0],
                                                        kernel_size[1], stride[0], stride[1], padding[0],
                                                        padding[1], use_bias)

            inputs_grad, weight_grad, bias_grad = output_grads
            return  inputs_grad, weight_grad, bias_grad, None, None, None, None
        else:
            inputs, weight, bias = ctx.saved_variables
            output_grads = depthwise_conv_cuda.backward(grad_output.contiguous(), inputs, weight, bias, kernel_size[0],
                                                        kernel_size[1], stride[0], stride[1], padding[0],
                                                        padding[1], use_bias)

            inputs_grad, weight_grad = output_grads
            return  inputs_grad, weight_grad, None, None, None, None, None

depthwise_conv = DepthWiseConvFunction.apply


class DepthWiseConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=0, use_bias=False):
        super(DepthWiseConv2d, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.use_bias = use_bias
        self.weight = Parameter(torch.Tensor(in_channels, 1, *self.kernel_size))
        if use_bias:
            self.bias = Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        return depthwise_conv(inputs, self.weight, self.bias, self.kernel_size, self.stride, self.padding, self.use_bias)



def testDepthWiseConvFunction():
    from torch.autograd import Variable
    from torch.nn import functional as F

    device = "cuda:0"
    inputs = torch.randn(4, 3, 7, 7).to(device)
    w = Variable(torch.randn(3, 1, 3, 3), requires_grad=True).to(device)
    b = Variable(torch.randn(3), requires_grad=True).to(device)

    # opt = F.conv2d(inputs, w, bias=None, stride=1, padding=0, dilation=1, groups=3)
    # print(opt.size())

    inp = depthwise_conv(inputs, w, b, 3)
    loss = inp.sum()
    loss.backward()

def testDepthWiseConv2d():
    device = "cuda:0"
    inputs = torch.randn(4, 3, 7, 7).to(device)
    depthwcon = DepthWiseConv2d(3)
    depthwcon.to(device)
    outp = depthwcon(inputs)
    loss = outp.sum()
    loss.backward()
    print(outp.size())


if __name__ == '__main__':
   testDepthWiseConvFunction()
   testDepthWiseConv2d()