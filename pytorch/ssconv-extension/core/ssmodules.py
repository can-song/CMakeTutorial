import torch
from torch.autograd import Function
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

import ssconv
from torch.nn.modules.utils import _pair

INDEX_TYPE = torch.short


class SSConv2dFunction(Function):
    @staticmethod
    def forward(ctx, 
                input: Tensor,
                in_index: Tensor,
                weight: Tensor,
                bias: Tensor,
                in_groups: int,
                out_groups: int,
                stride_h: int,
                stride_w: int):
        # check shape
        ctx.in_groups = in_groups
        ctx.out_groups = out_groups
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = stride_h // 2
        ctx.pad_w = stride_w // 2
        
        output = input.new_empty(
            SSConv2dFunction._get_output_size(ctx, input, weight))
        out_index = input.new_empty(
            SSConv2dFunction._get_output_size(ctx, input, weight), 
            dtype=INDEX_TYPE)
        ssconv.forward(input, in_index, weight, bias, output, out_index, 
                       in_groups, out_groups, stride_h, stride_w)

        ctx.save_for_backward(input, in_index, weight, bias, output, out_index)
        return output, out_index
    
    @staticmethod
    def backward(ctx, 
                 grad_output: Tensor,
                 grad_index=None):
        input, in_index, weight, bias, output, out_index = ctx.saved_variables
        output_grads = ssconv.backward(grad_output, input, in_index, 
                                       weight, bias, out_index,
                                       ctx.in_groups, ctx.out_groups,
                                       ctx.stride_h, ctx.stride_w)
        input_grad, weight_grad, bias_grad = output_grads
        return input_grad, None, weight_grad, bias_grad, None, None, None, None
    
    @staticmethod
    def _get_output_size(ctx, input, weight):
        batch_size = input.size(0)
        height = (input.size(1) - 1) // ctx.stride_h + 1
        width = (input.size(2) -1) // ctx.stride_w + 1
        channels =  weight.size(0) // ctx.out_groups
        return batch_size, height, width, channels
    

class SSConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size=(3, 3),
                 stride=1,
                 in_groups: int=1,
                 out_groups: int=1,
                 device=None) -> None:
        super().__init__()
        kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.in_groups = in_groups
        self.out_groups = out_groups
        
        self.weight = Parameter(
            torch.Tensor(out_channels, in_channels, *kernel_size, device=device))
        self.bias = Parameter(torch.Tensor(out_channels, device=device))
    
    def forward(self, inputs):
        # if isinstance(inputs, torch.Tensor):
        #     data_index_pair = DSTransform2d()(inputs)
        input, in_index = inputs
        output = SSConv2dFunction.apply(input, in_index, self.weight, self.bias, 
                                        self.in_groups, self.out_groups, *self.stride)
        # if self.out_groups == 1:
        #     return SDTransform2d()(output)
        return output
    
        
class DSTransform2d(nn.Module):
    def forward(self, data):
        # if not isinstance(data, torch.Tensor):
        #     return data
        input = data.permute(0, 2, 3, 1).contiguous()
        index = torch.zeros_like(input, dtype=INDEX_TYPE)
        return input, index


class SDTransform2d(nn.Module):
    def forward(self, data_index_pair):
        # if isinstance(data_index_pair, torch.Tensor):
        #     return data_index_pair
        input, in_index = data_index_pair
        input = input.permute(0, 3, 1, 2).contiguous()
        return input
        
        
class SSLayerNorm2d(nn.Module):
    def __init__(self,
                 normalized_shape,
                 eps=1e-5,
                 elementwise_affine=True,
                 device=None,
                 dtype=None) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps, 
                                       elementwise_affine, device, dtype)
    
    def forward(self, data_index_pair):
        data, index = data_index_pair
        data = self.layer_norm(data)
        return data, index
    
    
class SSBatchNorm2d(nn.Module):
    def __init__(self,
                 num_features: int,
                 eps: float=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True) -> None:
        super().__init__()
        
        self.batch_norm = nn.BatchNorm2d(num_features, eps, momentum, 
                                         affine, track_running_stats)
        
    def forward(self, data_index_pair):
        data, index = data_index_pair
        data = input.permute(0, 3, 1, 2).contiguous()
        data = self.batch_norm(data)
        data = data.permute(0, 2, 3, 1).contiguous()
        return data, index
    

class SSReLU(nn.Module):
    def __init__(self, inplace=False) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace)
        
        
    def forward(self, data_index_pair):
        data, index = data_index_pair
        data = self.relu(data)
        return data, index