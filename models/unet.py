import torch.nn as nn
from models.modules import *


class UNetModel(nn.Module):
    
    def __init__(self, image_size, in_channels, model_channels, out_channels, 
                 num_res_blocks=2, attention_resolutions=(16, 8), num_heads=1, num_head_channels=-1, num_heads_upsample=-1,
                 channel_mult=(1, 2, 4, 8), dims=2, dropout=0.1, num_classes=None, 
                 conv_resample=True, new_att_order=False):
        super().__init__()
        
        self.image_size     = image_size        # 默认image的H和W相等
        self.in_channels    = in_channels
        self.model_channels = model_channels
        self.out_channels   = out_channels
        
        self.num_res_blocks = num_res_blocks
        
        self.attention_resolutions = attention_resolutions
        self.num_heads             = num_heads
        self.num_head_channels     = num_head_channels
        self.num_head_upsample     = num_heads if num_heads_upsample == -1 else num_heads_upsample
        
        self.channel_mult = channel_mult
        self.dims         = dims
        self.num_classes  = num_classes
        self.dropout      = dropout
        self.dtype        = torch.float32
        
        self.conv_resample           = conv_resample
        self.new_att_order = new_att_order
        
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(nn.Linear(model_channels, time_embed_dim),
                                        nn.SiLU(),
                                        nn.Linear(time_embed_dim, time_embed_dim))

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(in_channels, ch, kernel_size=3, padding=1))])
        
        self._feature_size = ch
        input_block_channels = [ch]
        downsample = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=int(mult * model_channels), dims=dims)]
                ch = int(mult * model_channels)
                if downsample in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads, num_head_channels, new_att_order))
                    
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_channels.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample=conv_resample, 
                                                                            dims=dims, out_channels=out_ch)))
                ch = out_ch
                input_block_channels.append(ch)
                downsample *= 2
                self._feature_size += ch
                
        self.middle_block = TimestepEmbedSequential(ResBlock(ch, time_embed_dim, dropout, dims=dims),
                                                    AttentionBlock(ch, num_heads, num_head_channels, new_att_order),
                                                    ResBlock(ch, time_embed_dim, dropout, dims=dims))
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:     # 倒序遍历
            for i in range(num_res_blocks + 1):
                
                in_ch = input_block_channels.pop()
                layers = [ResBlock(ch + in_ch, time_embed_dim, dropout, out_channels=int(model_channels * mult), dims=dims)]
                
                ch = int(model_channels * mult)
                if downsample in attention_resolutions:
                    layers.append(AttentionBlock(ch, self.num_head_upsample, num_head_channels, new_att_order))
                    
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(Upsample(ch, conv_resample=conv_resample, dims=dims, out_channels=out_ch))
                    ch = out_ch
                    downsample //= 2
                    
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                
        self.out = nn.Sequential(nn.GroupNorm(32, ch),
                                 nn.SiLU(),
                                 zero_module(nn.Conv2d(input_ch, out_channels, kernel_size=3, padding=1)))
        
    def forward(self, t, x, sar=None, y=None):

        # assert (y is not None) == (self.num_classes is not None), 'Must specify y if and only if the model is class-conditional'
        
        # while t.dim() > 1:
        #     t = t[:, 0]
        # if t.dim() == 0:
        #     t = t.repeat(x.shape[0])
        
        hs = []
        emb = self.time_embed(timestep_embedding(t, self.model_channels))
            
        # if self.num_classes is not None:
        #     assert y.shape == (x.shape[0],)
        #     emb = emb + self.label_emb(y)
            
        h = x.type(self.dtype)
        
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        
        h = self.middle_block(h, emb)
        
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
            
        h = h.type(x.dtype)
        
        return self.out(h)