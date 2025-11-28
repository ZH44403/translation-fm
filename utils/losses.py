import torch
import torchvision
from torch import nn
from torch.nn import functional as F

# 约束速度场的L1 loss
class CharbonnierLoss(nn.Module):
    
    def __init__(self, eps: float=1e-3):
        super().__init__()
        
        self.eps = eps
        
    def forward(self, x, y):
        
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        

class PatchGANLoss(nn.Module):
    
    def __init__(self, in_channels=3+3):
        super().__init__()
        
        self.discriminator = PatchGANDiscriminator(in_channels=in_channels)
        self.criterion = nn.MSELoss()
        
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        
    def _make_input(self, x, y):
        return torch.cat([x, y], dim=1)
    
    def _target_tensor(self, pred, is_real: bool):
        
        label = self.real_label if is_real else self.fake_label

        return label.expand_as(pred)
    
    def D_loss(self, x, y_real, y_fake):
        
        real_in     = self._make_input(x, y_real)
        pred_real   = self.discriminator(real_in)
        target_real = self._target_tensor(pred_real, is_real=True)
        loss_real   = self.criterion(pred_real, target_real)
        
        fake_in     = self._make_input(x, y_fake.detach())
        pred_fake   = self.discriminator(fake_in)
        target_fake = self._target_tensor(pred_fake, is_real=False)
        loss_fake   = self.criterion(pred_fake, target_fake)
        
        loss_D = 0.5 * (loss_real + loss_fake)
        
        return loss_D
    
    def G_loss(self, x, y_fake):
        
        fake_in     = self._make_input(x, y_fake)
        pred_fake   = self.discriminator(fake_in)
        target_real = self._target_tensor(pred_fake, is_real=True)
        loss_G      = self.criterion(pred_fake, target_real)
        
        return loss_G
    
    def forward(self, x, y_real=None, y_fake=None, mode='D'):
        
        if mode == 'D':
            assert y_real is not None and y_fake is not None, 'y_real and y_fake must be provided for mode D'
            return self.D_loss(x, y_real, y_fake)
        
        elif mode == 'G':
            assert y_fake is not None, 'y_fake must be provided for mode G'
            return self.G_loss(x, y_fake)
            
        else:
            raise ValueError(f'Unsupported mode: {mode}')
        

class PatchGANDiscriminator(nn.Module):
    
    def __init__(self, in_channels=3+3):    # sar+opt
        super().__init__()
        
        def block(in_channels, out_channels, norm=True):
            
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_channels))
                
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers
        
        self.model = nn.Sequential(
            *block(in_channels, 64, norm=False),
            *block(64, 128),
            *block(128, 256),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)
        )
        
    def forward(self, x):
        return self.model(x)