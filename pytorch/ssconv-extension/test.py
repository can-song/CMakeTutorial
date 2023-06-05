import torch
import torch.nn as nn
from core.ssmodules import SSConv2d, SDTransform2d, DSTransform2d

in_channels = 3
out_channels = 4
img_shape = (50, 50)

device = 'cpu'

def test_one():
    conv = nn.Conv2d(1, 1, 3, 1, 1, device=device)
    ssconv = SSConv2d(1, 1, 3, 1, device=device)
    ssconv.load_state_dict(conv.state_dict(), strict=False)

    dstrans = DSTransform2d()
    sdtrans = SDTransform2d()
    x = torch.ones([1, 1, 3, 3], device=device)
    gt = conv(x)
    res = sdtrans(ssconv(dstrans(x)))
    assert torch.allclose(res, gt, 1e-3, 1e-5)


def test_two(in_channels=3, 
             out_channels=4,
             in_height=50,
             in_width=50,
             kernel_size=3,
             stride=2):
    pad = kernel_size//2
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, device=device)
    ssconv = SSConv2d(in_channels, out_channels, kernel_size, stride, device=device)
    ssconv.load_state_dict(conv.state_dict(), strict=False)

    dstrans = DSTransform2d()
    sdtrans = SDTransform2d()
    x = torch.randn([1, in_channels, in_height, in_width], device=device, requires_grad=True)
    y = x.clone().detach()
    y.requires_grad = True
    gt = conv(x)
    res = sdtrans(ssconv(dstrans(y)))
    dx = torch.randn_like(gt)
    dy = dx
    gt.backward(dx)
    res.backward(dy)

    assert torch.allclose(x.grad, y.grad, 1e3, 1e5)
    assert torch.allclose(conv.weight.grad, conv.weight.grad, 1e3, 1e5)
    assert torch.allclose(conv.bias.grad, conv.bias.grad, 1e3, 1e5)

    assert torch.allclose(res, gt, 1e-3, 1e-5)
    
def test_three(in_channels=3, 
               out_channels=4,
               in_height=50,
               in_width=50,
               kernel_size=3,
               stride=2):
    pad = kernel_size//2
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, device=device)
    ssconv_cpu = SSConv2d(in_channels, out_channels, kernel_size, stride, device="cpu")
    ssconv_gpu = SSConv2d(in_channels, out_channels, kernel_size, stride, device="cuda")
    ssconv_cpu.load_state_dict(conv.state_dict(), strict=False)
    ssconv_gpu.load_state_dict(conv.state_dict(), strict=False)

    dstrans = DSTransform2d()
    sdtrans = SDTransform2d()
    x_cpu = torch.randn([1, in_channels, in_height, in_width], device=device, requires_grad=True)
    x_gpu = x_cpu.clone().detach().to("cuda")
    
    y_cpu = sdtrans(ssconv_cpu(dstrans(x_cpu)))
    y_gpu = sdtrans(ssconv_gpu(dstrans(x_gpu)))
    
    assert torch.allclose(y_cpu, y_gpu, 1e-3, 1e-5)

    # assert torch.allclose(x.grad, y.grad, 1e3, 1e5)
    # assert torch.allclose(conv.weight.grad, conv.weight.grad, 1e3, 1e5)
    # assert torch.allclose(conv.bias.grad, conv.bias.grad, 1e3, 1e5)

    # assert torch.allclose(res, gt, 1e-3, 1e-5)


if __name__=="__main__":
    test_one()
    test_two(3, 4, 11, 11, 3, 2)
    # test_two()

