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
        self.cond_mlp = nn.Linear(time_embed_dim, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
    def forward(self, x, t=None, cond_embed=None):
        x = self.conv1(x)
        if t is None:
            t = torch.zeros(1).to(x.device)
        else:
            time_emb = self.relu(self.time_mlp(t))
            if cond_embed is not None:
                cond_embed = self.relu(self.cond_mlp(cond_embed))
                x = x * cond_embed[(..., ) + (None, ) * 2]
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
    def forward(self, x, skip_conn, t, cond_embed=None):
        x = self.deconv(x)
        x = torch.cat([x, skip_conn], dim=1)
        return self.conv_block(x, t, cond_embed)
    
    
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
            x, w = down_block(x, None)
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
    
class Conditional_UNet(UNet):
    '''Conditional UNet for Diffusion.'''
    def __init__(self, in_channels=1, out_channels=1, block_out_channels=[128, 256, 512], embed_dim=32, n_classes=10, device='cpu'):
        '''Initialize a Conditional UNet model.
        in_channels: channels of input images.
        out_channels: channels of output images.
        block_out_channels: number of channels as output of each UNet convolotional block.
        embed_dim: dimension of the embedding (time / conditional). Must be greater or equal than n_classes.'''
        super().__init__(in_channels, out_channels, block_out_channels, embed_dim, device)
        assert embed_dim >= n_classes
        self.n_classes = n_classes
        self.cond_embedding = nn.Sequential(
            nn.Linear(n_classes, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
    def forward(self, x, timestep, c=None):
        skip_connections = []
        timestep_tensor = torch.ones(1).to(x.device) * timestep
        t = self.time_mlp(timestep_tensor)
        if c is not None:
            c = nn.functional.one_hot(c, num_classes=self.n_classes).float()
            cond_embed = self.cond_embedding(c)
        else:
            cond_embed = None
        for down_block in self.encoder:
            x, w = down_block(x, None)
            skip_connections.append(w)
        x = self.neck(x, t)
        for w, up_block in zip(skip_connections[::-1], self.decoder):
            x = up_block(x, w, t, cond_embed)
        x = self.outConv(x)
        return x
    
    def to(self, device):
        super().to(device)
        self.time_mlp.to(device)
        self.cond_embedding.to(device)
        return self