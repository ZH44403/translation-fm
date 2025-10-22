import torch
import random
import numpy as np

from omegaconf import DictConfig
from models.unet import UNetModel
from models.flow import OptimalTransportFlow


def set_seed(seed: int) -> None:
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    
def loss_func(model: UNetModel, flow: OptimalTransportFlow):
    
    def _loss(batch: torch.Tensor) -> torch.Tensor:
        
        t = torch.rand(batch.shape[0], device=batch.device)
        x_0 = torch.randn_like(batch)
        
        x_t, v_true = flow.step(t, x_0, batch)
        v_pred = model(x_t, t)
    
        return torch.nn.functional.mse_loss(v_pred, v_true)

    return _loss
    
    
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


def load_checkpoint(path, model, optimizer=None, scaler=None, ema_model=None):
    
    checkpoint = torch.load(path, weights_only=True)
    
    step  = int(checkpoint['step'])
    epoch = int(checkpoint['epoch'])

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    if ema_model is not None:
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        ema_model.eval()
        

    return step, epoch, model, optimizer, scaler, ema_model