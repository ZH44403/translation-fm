import json
import torch

from pathlib import Path
from typing import Dict


def save_last(model: torch.nn.Module, ema_model: torch.nn.Module, optimizer: torch.optim.Optimizer,
              scheduler: torch.optim.lr_scheduler._LRScheduler, scaler: torch.cuda.amp.GradScaler, 
              log_dirL: Path, epoch: int, metrics: Dict[str, float], ):
    
    
    return 