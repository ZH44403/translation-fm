import math
import torch
import torch.nn as nn

from abc import abstractmethod



class TimestepBlock(nn.Module):
    
    @abstractmethod
    def forward(self, x, emb):
        """  """
        

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    
    def forward(self, x, emb):
        
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
                
        return x


# 下采样模块
class Downsample(nn.Module):
    
    def __init__(self, in_channels, conv_resample, dims=2, out_channels=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.conv_resample = conv_resample
        self.dims = dims
        
        if self.conv_resample:
            self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            assert self.in_channels == self.out_channels
            self.down = nn.AvgPool2d(kernel_size=2, stride=2)
            
    def forward(self, x):
        
        assert x.shape[1] == self.in_channels, ('Expected input {} channels, but got {} channels', self.in_channels, x.shape[1])
        
        return self.down(x)


# 上采样模块
class Upsample(nn.Module):
    
    def __init__(self, in_channels, conv_resample, dims=2, out_channels=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.conv_resample = conv_resample
        self.dims = dims

        if self.conv_resample:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        
        assert x.shape[1] == self.in_channels, ('Expected input {} channels, but got {} channels', self.in_channels, x.shape[1])
        
        x = self.up(x)
        if self.conv_resample:
            x = self.conv(x)
            
        return x
        

class ResBlock(TimestepBlock):
    
    def __init__(self, in_channels, emb_channels, dropout, out_channels=None, dims=2, up=False, down=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or in_channels
        self.dims = dims
        
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1)
        )
        
        self.updown = up or down
        
        if up:
            self.h_upd = Upsample(in_channels, False, dims)
            self.x_upd = Upsample(in_channels, False, dims)
        elif down:
            self.h_upd = Downsample(in_channels, False, dims)
            self.x_upd = Downsample(in_channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels)
        )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1))
        )
        
        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
            
        
    def forward(self, x, emb):
        
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        
        h = h + emb_out[:, :, None, None]
        h = self.out_layers(h)
        
        return self.skip_connection(x) + h
            
            
class AttentionBlock(nn.Module):
    
    def __init__(self, in_channels, num_heads=1, num_head_channels=-1, new_att_order=False):
        super().__init__()
        
        self.in_channels = in_channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert in_channels % num_head_channels == 0, 'q,k,v channels {in_channels} is not divisible by num_head_channels {num_head_channels}'
            self.num_heads = in_channels // num_head_channels
            
        self.norm = nn.GroupNorm(32, in_channels)
        self.qkv = nn.Conv1d(in_channels, in_channels * 3, kernel_size=1)
        self.attention = QKVAttention(self.num_heads) if new_att_order else QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(in_channels, in_channels, kernel_size=1))

    def forward(self, x):
        
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        qkv = self.qkv(self.norm(x))  # NC3HW
        h = self.attention(qkv)
        h = self.proj_out(h)
        
        return (x + h).reshape(b, c, *spatial)



class QKVAttentionLegacy(nn.Module):

    def __init__(self, n_heads):
        super().__init__()
        
        self.n_heads = n_heads

    def forward(self, qkv):

        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum('bct,bcs->bts', q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum('bts,bcs->bct', weight, v)
        
        return a.reshape(bs, -1, length)



class QKVAttention(nn.Module):

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)

        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum('bct,bcs->bts', 
                              (q * scale).view(bs * self.n_heads, ch, length),
                              (k * scale).view(bs * self.n_heads, ch, length))
        a = torch.einsum('bts,bcs->bct', weight, v.reshape(bs * self.n_heads, ch, length))

        return a.reshape(bs, -1, length)


class AttentionPool2d(nn.Module):
    
    def __init__(self, spatial_dim, embed_dim, num_head_channels, output_dim):
        super().__init__()
        
        self.positional_embedding = nn.Parameter(torch.randn(embed_dim, spatial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = nn.Conv1d(embed_dim, embed_dim * 3, 1)
        self.c_proj = nn.Conv1d(embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_head_channels
        self.attention = QKVAttention(self.num_heads)
        
    def forward(self, x):
        
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        
        return x[:, :, 0]



def timestep_embedding(timesteps, dim, max_period=10000):
    
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    
    return embedding


        
def zero_module(module: nn.Module):
    
    """Zero out the parameters of a module and return it."""
    
    for p in module.parameters():
        p.detach().zero_()
    
    return module