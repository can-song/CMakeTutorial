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
    x = torch.randn([1, in_channels, in_height, in_width], device=device)
    gt = conv(x)
    res = sdtrans(ssconv(dstrans(x)))

    assert torch.allclose(res, gt, 1e-3, 1e-5)


if __name__=="__main__":
    test_one()
    test_two(3, 4, 11, 11, 3, 2)
    # test_two()

