import torch
from torch.autograd import Function
import torch.nn as nn
from torch.nn import init
from torch import Tensor
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import math

import os
import os.path as osp
import sys
import importlib

core = importlib.import_module('ssconv.core')
from torch.nn.modules.utils import _pair

INDEX_TYPE = torch.uint8
# INDEX_TYPE = torch.int32


class SSConv2dFunction(Function):
    @staticmethod
    def forward(ctx, 
                input: Tensor,
                in_index: Tensor,
                weight: Tensor,
                bias: Tensor,
                in_block_size: int,
                out_block_size: int,
                stride_h: int,
                stride_w: int,
                pad_h: int,
                pad_w: int):
        # check shape
        ctx.in_block_size = in_block_size
        ctx.out_block_size = out_block_size
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        
        output = input.new_empty(
            SSConv2dFunction._get_output_size(ctx, input, weight))
        out_index = input.new_empty(
            SSConv2dFunction._get_output_size(ctx, input, weight),
            dtype=INDEX_TYPE)
        core.forward(input, in_index, weight, bias, output, out_index,
                     in_block_size, out_block_size, stride_h, stride_w, pad_h, pad_w)

        ctx.save_for_backward(input, in_index, weight, bias, out_index)
        return output, out_index
    
    @staticmethod
    def backward(ctx,
                 grad_output: Tensor,
                 grad_index=None):
        input, in_index, weight, bias, out_index = ctx.saved_variables
        output_grads = core.backward(grad_output, input, in_index, 
                                     weight, bias, out_index,
                                     ctx.in_block_size, ctx.out_block_size,
                                     ctx.stride_h, ctx.stride_w,
                                     ctx.pad_h, ctx.pad_w)
        input_grad, weight_grad, bias_grad = output_grads
        return input_grad, None, weight_grad, bias_grad, None, None, None, None, None, None
    
    @staticmethod
    def _get_output_size(ctx, input, weight):
        batch_size = input.size(0)
        kernel_h, kernel_w = weight.size(2), weight.size(3)
        height = (input.size(2) - 1 + ctx.pad_h + ctx.pad_h - kernel_h + 1) // ctx.stride_h + 1
        width = (input.size(3) - 1 + ctx.pad_w + ctx.pad_w - kernel_w + 1) // ctx.stride_w + 1
        channels =  weight.size(0) // ctx.out_block_size
        return batch_size, channels, height, width
    

class SSConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 in_blocks: int,
                 out_blocks: int,
                 kernel_size=(1, 1),
                 stride=1,
                 pad=0,
                 device=None) -> None:
        super().__init__()
        kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        assert in_channels % in_blocks == 0
        assert out_channels % out_blocks  == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_block_size = in_channels // in_blocks
        self.out_block_size = out_channels // out_blocks
        
        self.weight = Parameter(
            torch.empty((out_channels, in_channels, *kernel_size), device=device))
        self.bias = Parameter(torch.empty(out_channels, device=device))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
    
    def forward(self, inputs):
        input, in_index = inputs
        output = SSConv2dFunction.apply(input, in_index, self.weight, self.bias, 
                                        self.in_block_size, self.out_block_size, *self.stride, *self.pad)
        return output
    
        
class DSTransform2d(nn.Module):
    def forward(self, data):
        index = torch.zeros_like(data, dtype=INDEX_TYPE)
        return data, index


class SDTransform2d(nn.Module):
    def forward(self, data_index_pair):
        data, in_index = data_index_pair
        return data

# class SSLayerNorm2d(nn.Module):
#     def __init__(self, num_channels: int, num_blocks:int, eps: float = 1e-6) -> None:
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(num_channels))
#         self.bias = nn.Parameter(torch.zeros(num_channels))
#         self.eps = eps
#         self.num_blocks = num_blocks

#     def forward(self, data_index_pair) -> torch.Tensor:
#         data, index = data_index_pair
#         u = data.mean(1, keepdim=True)
#         s = (data - u).pow(2).mean(1, keepdim=True)
#         data = (data - u) / torch.sqrt(s + self.eps)
#         data = self.weight[index] * data + self.bias[index]
#         return data, index
    
class SSLayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, num_blocks:int, eps: float = 1e-6) -> None:
        super().__init__()
        assert num_channels % num_blocks == 0
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        self.num_blocks = num_blocks
        # self.base = torch.range(num_blocks, dtype=torch.int32) * num_channels / num_blocks
        
        self.register_buffer('base', torch.arange(num_blocks, dtype=torch.int32) * num_channels // num_blocks)

    def forward(self, data_index_pair) -> torch.Tensor:
        data, index = data_index_pair
        shifted_index = index + self.base[..., None, None]
        # u = data.mean(1, keepdim=True)
        # s = (data - u).pow(2).mean(1, keepdim=True)
        u = data.mean()
        s = (data - u).pow(2).mean()
        data = (data - u) / torch.sqrt(s + self.eps)
        data = self.weight[shifted_index] * data + self.bias[shifted_index]
        return data, index
        
# class SSLayerNorm2d(nn.Module):
#     def __init__(self,
#                  normalized_shape,
#                  eps=1e-5,
#                  elementwise_affine=True,
#                  device=None,
#                  dtype=None) -> None:
#         super().__init__()
#         self.layer_norm = nn.LayerNorm(normalized_shape, eps, 
#                                        elementwise_affine=False, device=device, dtype=dtype)
    
#     def forward(self, data_index_pair):
#         data, index = data_index_pair
#         data = self.layer_norm(data)
#         return data, index
    
    
# class SSBatchNorm2d(nn.Module):
#     def __init__(self,
#                  num_features: int,
#                  eps: float=1e-5,
#                  momentum=0.1,
#                  affine=True,
#                  track_running_stats=True) -> None:
#         super().__init__()
        
#         self.batch_norm = nn.BatchNorm2d(num_features, eps, momentum, 
#                                          affine, track_running_stats)
        
#     def forward(self, data_index_pair):
#         data, index = data_index_pair
#         data = data.permute(0, 3, 1, 2).contiguous()
#         data = self.batch_norm(data)
#         data = data.permute(0, 2, 3, 1).contiguous()
#         return data, index
    

class SSReLU(nn.Module):
    def __init__(self, inplace=False) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace)
        
        
    def forward(self, data_index_pair):
        data, index = data_index_pair
        data = self.relu(data)
        return data, index