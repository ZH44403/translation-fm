import torch
import random
import numpy as np

from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int) -> None:
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    
def get_lr(args: DictConfig, step: int) -> float:
    
    min_lr, max_lr = args.train.min_lr, args.train.max_lr
    warmup_steps, max_steps = args.train.warmup_steps, args.train.max_steps
    
    # warmup
    if step < warmup_steps:
        lr = min_lr + (max_lr - min_lr) * step / warmup_steps
    
    # linear decay
    elif step <= max_steps:
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        lr = max_lr - (max_lr - min_lr) * decay_ratio
    
    # fixed
    else:
        lr = min_lr

    return max(min_lr, min(lr, max_lr))





def save_checkpoint(path, epoch, model, ema_model, optimizer, args, Metrics=None):
    
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'ema_model': ema_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': OmegaConf.to_container(args, resolve=True),
    }

    if Metrics is not None:
        state['Metrics'] = Metrics

    torch.save(state, path)


def load_checkpoint(path, ema_model=None, model=None, optimizer=None):
    
    checkpoint = torch.load(path, map_location='cpu')
    
    epoch = int(checkpoint['epoch'])
    
    if ema_model is not None:
        ema_model.load_state_dict(ema_to_model(checkpoint['ema_model']))
        
    if model is not None:
        model.load_state_dict(checkpoint['model'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])    
        
    return epoch+1, ema_model, model, optimizer


# 将ema_state_dict中的键值转换为model_state_dict的格式
def ema_to_model(ema_state_dict):
    
    new_state_dict = {}
    
    for k, v in ema_state_dict.items():
        
        if k == 'n_averaged':
            continue
        
        if k.startswith('module.'):
            k = k[len('module.'):]
            
        new_state_dict[k] = v

    return new_state_dict


def compute_valid_score(psnr, ssim, lpips):
    return psnr + 50.0 * ssim - 10.0 * lpips


def to_01(x, eps=1e-8):
    
    dims = tuple(range(1, x.ndim))
    x_min = x.amin(dim=dims, keepdim=True)
    x_max = x.amax(dim=dims, keepdim=True)
    scale = (x_max - x_min).clamp_min(eps)
    
    return (x - x_min) / scale