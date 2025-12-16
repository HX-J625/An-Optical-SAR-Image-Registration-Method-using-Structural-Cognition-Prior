import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['MDSE']


def create_steerable_filter(order=1, theta=0, sigma=1.0, device=None):

    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1

    if device is None:
        device = torch.device('cuda')

    ax = torch.arange(-size // 2 + 1., size // 2 + 1., device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')

    theta_rad = math.radians(theta)
    x_theta = xx * math.cos(theta_rad) + yy * math.sin(theta_rad)

    gauss = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    gauss = gauss / (2 * math.pi * sigma ** 2)

    if order == 1:
        kernel = -x_theta / (sigma ** 2) * gauss
    elif order == 2:
        kernel = ((x_theta ** 2 - sigma ** 2) / (sigma ** 4)) * gauss
    else:
        raise ValueError("Order must be 1 or 2")

    kernel -= kernel.mean()
    kernel /= (kernel.norm() + 1e-6)

    return kernel 


def convolve(img, kernel):

    kernel = kernel.unsqueeze(0).unsqueeze(0)  
    padding = kernel.shape[-1] // 2
    return F.conv2d(img, kernel, padding=padding)


class MDSE(nn.Module):

    def __init__(self):
        super(MDSE, self).__init__()

        self.first_order_dirs = 6
        self.second_order_dirs = 6
        self.first_order_sigmas = [0.6, 0.8, 1.0]
        self.second_order_sigmas = [1.5]
        self.post_norm = nn.InstanceNorm2d(32, affine=False)

    def forward(self, x):

        B, _, H, W = x.shape
        device = x.device
        outputs = []

        for i in range(self.first_order_dirs):
            theta = i * 180 / self.first_order_dirs
            combined = 0
            for sigma in self.first_order_sigmas:
                kernel = create_steerable_filter(order=1, theta=theta, sigma=sigma, device=device)
                feat = convolve(x, kernel)
                feat = (feat - feat.mean(dim=[2, 3], keepdim=True)) / (feat.std(dim=[2, 3], keepdim=True) + 1e-5)
                combined += feat
                
            outputs.append(combined)
            
        for i in range(self.second_order_dirs):
            theta = i * 180 / self.second_order_dirs
            combined = 0
            for sigma in self.second_order_sigmas:
                kernel = create_steerable_filter(order=2, theta=theta, sigma=sigma, device=device)
                feat = convolve(x, kernel)
                feat = (feat - feat.mean(dim=[2, 3], keepdim=True)) / (feat.std(dim=[2, 3], keepdim=True) + 1e-5)
                combined += feat

            outputs.append(combined)

        return self.post_norm(torch.cat(outputs, dim=1))

