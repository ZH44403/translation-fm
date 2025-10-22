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
    
    
def loss_func(model: UNetModel, flow: OptimalTransportFlow, batch: torch.Tensor):
    
    t = torch.rand(batch.shape[0], device=batch.device)
    x_0 = torch.randn_like(batch)
    
    x_t, v_true = flow(t, x_0, batch)
    v_pred = model(x_t, t)
    
    return torch.nn.MSELoss(v_pred, v_true)
    
    
def get_lr(args: DictConfig, step: int) -> float:
    
    min_lr, max_lr = args.train.lr_min, args.train.lr_max
    

    return 