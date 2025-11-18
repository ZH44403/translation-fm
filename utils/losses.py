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
    

class PerceptualLoss(nn.Module):
    
    def __init__(self, resize: bool=True, layer_weights=None, input_range: str='-1, 1'):
        super().__init__()
        
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
        
        self.stage1 = nn.Sequential(*[vgg[i] for i in range(0, 4)])
        self.stage2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])
        self.stage3 = nn.Sequential(*[vgg[i] for i in range(9, 16)])
        
        # 冻结VGG参数
        for p in self.parameters():
            p.requires_grad = False
        
        if layer_weights is None:
            layer_weights = [1.0, 1.0, 1.0]
            
        self.layer_weights = layer_weights
        self.resize = resize
        
        assert input_range in ['-1, 1', '0, 1']
        self.input_range = input_range
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    # 将输入从[-1, 1]或[0, 1]映射到VGG需要的[0, 1], 并做ImageNet的归一化
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        
        # 保证通道数为3
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        if self.input_range == '-1, 1':
            x = (x + 1.0) / 2.0
            
        x = (x - self.mean) / self.std

        return x
    
    def _vgg_forward(self, x: torch.Tensor) -> torch.Tensor:
        
        h1 = self.stage1(x)
        h2 = self.stage2(h1)
        h3 = self.stage3(h2)
        
        return [h1, h2, h3]
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        
        if self.resize:
            x = F.interpolate(x, size=(224, 224), model='bilinear', align_corners=False)
            y = F.interpolate(y, size=(224, 224), model='bilinear', align_corners=False)
            
        features_x = self._vgg_forward(x)
        features_y = self._vgg_forward(y)

        loss = 0.0
        
        for w, fx, fy in zip(self.layer_weights, features_x, features_y):
            loss += w * F.l1_loss(fx, fy)

        return loss
        

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