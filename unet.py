from torch import nn
import torch
import math

class SinusoidalPositionEmbeddings(nn.Module):
    '''Sinusoidal position embeddings for time parameter used to encode timestep of the forward diffusion process.'''
    def __init__(self, dim, device='cpu'):
        super().__init__()
        self.dim = dim
        self.device = device

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=self.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvBlock(nn.Module):
    '''Convolutional block for U-Net encoder and decoder'''
    def __init__(self, in_channels, out_channels, time_embed_dim=32):
        super().__init__()
        self.time_mlp = nn.Linear(time_embed_dim, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x, t):
        x = self.conv1(x)
        time_emb = self.relu(self.time_mlp(t))
        x = x + time_emb[(..., ) + (None, ) * 2]
        x = self.conv2(x)
        return x
    
class DownBlock(nn.Module):
    '''Down sampling ResNet block'''
    def __init__(self, in_channels,out_channels, time_embed_dim=32):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, time_embed_dim)
        self.maxpool = nn.MaxPool2d(2)
    
    def forward(self, x, t):
        skip_connection = self.conv_block(x, t)
        x = self.maxpool(skip_connection)
        return x, skip_connection
    
    
class UpBlock(nn.Module):
    '''Up sampling ResNet block'''
    def __init__(self, in_channels, out_channels, time_embed_dim=32):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.conv_block = ConvBlock(2*in_channels, out_channels, time_embed_dim)
    def forward(self, x, skip_conn, t):
        x = self.deconv(x)
        x = torch.cat([x, skip_conn], dim=1)
        return self.conv_block(x, t)
    
    
class UNet(nn.Module):
    '''UNet based on ResNet architecture for Diffusion.'''
    def __init__(self, in_channels=1, out_channels=1, block_out_channels=[128, 256, 512], time_embed_dim=32, device='cpu'):
        '''Initialize a UNet model.
        in_channels: channels of input images.
        out_channels: channels of output images.
        block_out_channels: number of channels as output of each UNet convolotional block.'''
        super().__init__()
        assert len(block_out_channels)>=1
        # Intitialize encoder list and decoder list
        self.encoder = nn.ModuleList()
        self.encoder.append(DownBlock(in_channels, block_out_channels[0], time_embed_dim))
        self.decoder = nn.ModuleList()
        self.decoder.append(UpBlock(block_out_channels[0], block_out_channels[0], time_embed_dim))
        self.neck = ConvBlock(block_out_channels[-1], block_out_channels[-1], time_embed_dim)
        for i in range(len(block_out_channels)-1):
            self.encoder.append(DownBlock(block_out_channels[i], block_out_channels[i + 1], time_embed_dim))
            self.decoder.insert(0, UpBlock(block_out_channels[i+1],block_out_channels[i], time_embed_dim))
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim, device=device),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU(),
        )
        self.outConv = nn.Sequential(nn.Conv2d(block_out_channels[0], out_channels, 1), nn.Sigmoid(), nn.Conv2d(out_channels, out_channels, 1))
    
    def forward(self, x, timestep):
        skip_connections = []
        timestep_tensor = torch.ones(1).to(x.device) * timestep
        t = self.time_mlp(timestep_tensor)
        for down_block in self.encoder:
            x, w = down_block(x, t)
            skip_connections.append(w)
        x = self.neck(x, t)
        for w, up_block in zip(skip_connections[::-1], self.decoder):
            x = up_block(x, w, t)
        x = self.outConv(x)
        return x
    
    def to(self, device):
        super().to(device)
        self.time_mlp.to(device)
        return self