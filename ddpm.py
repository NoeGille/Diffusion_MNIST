from unet import UNet, Conditional_UNet
import torch.nn as nn
import torch

class DDPM(nn.Module):
    '''Diffusion model based on UNet architecture.'''
    def __init__(self, in_channels=1, out_channels=1, dim=28,block_out_channels=[64, 128], time_embed_dim=32, device='cpu'):
        '''Initialize a DDPM model.
        in_channels: channels of input images.
        out_channels: channels of output images.
        block_out_channels: number of channels as output of each UNet convolotional block.'''
        super().__init__()
        self.model = UNet(in_channels, out_channels, block_out_channels, time_embed_dim, device)
        self.betas = torch.linspace(10e-4, 0.02, 1000).to(device)
        self.alphas = 1 - self.betas
        self.cumprod_alphas = torch.cumprod(self.alphas, 0)
        self.sqrt_one_minus_cumprod_alphas = torch.sqrt(1 - self.cumprod_alphas)
        self.dim = dim
        self.device = device
        
    def forward(self, x, t, noise=None):
        '''Forward pass of the model.
        x: input image.
        t: timestep of the diffusion process.
        noise: noise to be added to the image.'''
        if noise is None:
            noise = torch.randn_like(x)
        mean = torch.sqrt(self.cumprod_alphas[t])
        std = self.sqrt_one_minus_cumprod_alphas[t]
        x = mean*x + noise*std
        return x, noise
    
    def to(self, device):
        super().to(device)
        self.model.to(device)
        return self
    
    def backward_diffusion(self, x_t, t):
        '''Backward pass of the model.
        x: input image.
        t: timestep of the diffusion process.
        noise: noise to be added to the image.'''
        if t > 0:
            z = torch.randn(1, 1, self.dim, self.dim).to(self.device)
        else:
            z = torch.zeros(1, 1, self.dim, self.dim).to(self.device)
        pred = self.model(x_t, t)
        x_t = (1 / torch.sqrt(self.alphas[t])) * (x_t - (self.betas[t]/self.sqrt_one_minus_cumprod_alphas[t])*pred) + z * torch.sqrt(self.betas[t])
        return x_t
    
    def sampling(self, T):
        '''Sampling from the model.
        t: timestep of the diffusion process.'''
        x_t = torch.randn(1, 1, self.dim, self.dim).to(self.device)
        for t in range(T- 1, 0, -1):
            x_t = self.backward_diffusion(x_t, t)
            x_t = torch.clamp(x_t, -1, 1)
        return x_t
    
class Conditional_DDPM(nn.Module):
    '''Conditional DDPM for Diffusion.'''
    def __init__(self, in_channels=1, out_channels=1, dim=28, block_out_channels=[64, 128], embed_dim=32, n_classes=10, device='cpu'):
        '''Initialize a Conditional DDPM model.
        in_channels: channels of input images.
        out_channels: channels of output images.
        block_out_channels: number of channels as output of each UNet convolotional block.'''
        super().__init__()
        self.model = Conditional_UNet(in_channels, out_channels, block_out_channels, embed_dim, n_classes, device)
        self.betas = torch.linspace(10e-4, 0.02, 1000).to(device)
        self.alphas = 1 - self.betas
        self.cumprod_alphas = torch.cumprod(self.alphas, 0)
        self.sqrt_one_minus_cumprod_alphas = torch.sqrt(1 - self.cumprod_alphas)
        self.dim = dim
        self.device = device
        self.in_channels = in_channels

    def forward(self, x, t, noise=None):
        '''Forward pass of the model.
        x: input image.
        t: timestep of the diffusion process.
        y: conditional information.
        noise: noise to be added to the image.'''
        if noise is None:
            noise = torch.randn_like(x)
        mean = torch.sqrt(self.cumprod_alphas[t])
        std = self.sqrt_one_minus_cumprod_alphas[t]
        x = mean*x + noise*std
        return x, noise
    
    def to(self, device):
        super().to(device)
        self.model.to(device)
        self.device = device
        return self
    
    @torch.no_grad()
    def backward_diffusion(self, x_t, t, c, w):
        '''Backward pass of the model.
        x: input image.
        t: timestep of the diffusion process.
        noise: noise to be added to the image.'''
        assert w >= 0
        if t > 0:
            z = torch.randn(x_t.shape).to(self.device)
        else:
            z = torch.zeros(x_t.shape).to(self.device)
        pred_uncond = self.model(x_t, t)
        pred_cond = self.model(x_t, t, c)
        pred = (w + 1) * pred_cond - w * pred_uncond
        x_t = (1 / torch.sqrt(self.alphas[t])) * (x_t - (self.betas[t]/self.sqrt_one_minus_cumprod_alphas[t])*pred) + z * torch.sqrt(self.betas[t])
        return x_t
    
    @torch.no_grad()
    def sampling(self, T, c, n_samples, w=0.0):
        '''Sampling from the model.
        t: timestep of the diffusion process.'''
        x_t = torch.randn(n_samples, self.in_channels, self.dim, self.dim).to(self.device)
        for t in range(T- 1, 0, -1):
            x_t = self.backward_diffusion(x_t, t, torch.ones(n_samples, dtype=torch.long).to(self.device) * c, w=w)
            x_t = torch.clamp(x_t, -1, 1)
        return x_t
    