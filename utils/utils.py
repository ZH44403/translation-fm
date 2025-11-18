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


def load_checkpoint(path, model, optimizer=None, scaler=None, ema_model=None):
    
    checkpoint = torch.load(path, weights_only=True)
    
    epoch = int(checkpoint['epoch'])

    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    ema_model.load_state_dict(checkpoint['ema_model'])
    ema_model.eval()
    
    optimizer.load_state_dict(checkpoint['optimizer'])    
        
    return epoch, model, optimizer, ema_model


def compute_valid_score(psnr, ssim, lpips):
    return psnr + 50.0 * ssim - 10.0 * lpips