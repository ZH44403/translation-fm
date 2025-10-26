import math
import torch
import torch.nn as nn
import torchdiffeq
from typing import Literal


# original flow matching
class OptimalTransportFlow:
    
    def __init__(self, sigma_min: float = 0.01):
        super().__init__()
        
        self.sigma_min = sigma_min
        
    def step(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        
        t = t.reshape(-1, *([1] * (x_1.ndim - 1))).expand_as(x_1)   # 将t扩展到与x_1相同形状
        
        mu_t = t * x_1
        sigma_t = 1 - (1 - self.sigma_min) * t
        
        # eq (22): phi_t(x) = x_t = (1 - ( 1 - sigma_min) * t) * x_0 + t * x_1 = mu_t + sigma_t * x_0
        x_t = mu_t + sigma_t * x_0
        dx_t = x_1 - (1 - self.sigma_min) * x_0

        return x_t, dx_t
    

# direct translation from sar to optical with Gaussian noise
class GaussianBridgeFlow:
    
    def __init__(self, sigma: float = 0.1):
        super().__init__()

        self.sigma = sigma

    def step(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:

        t = t.reshape(-1, *([1] * (x_1.ndim - 1))).expand_as(x_1)   # 将t扩展到与x_1相同形状
        epsilon = torch.randn_like(x_1)
        
        sigma_t = self.sigma * (1.0 - t)
        
        x_t = (1.0 - t) * x_0 + t * x_1 + sigma_t * epsilon
        dx_t = (x_1 - x_0) - self.sigma * epsilon

        return x_t, dx_t
    

@torch.inference_mode()
def integrate_flow(model: nn.Module, x_0: torch.Tensor, steps: int, device: torch.device,
                   method: Literal['euler', 'heun', 'odeint']='odeint', dt_schedule: Literal['linear', 'cosine']='linear',
                   rtol: float=1e-3, atol: float=1e-4) -> torch.Tensor:

    if method == 'euler':
        return integrate_flow_euler(model, x_0, steps, device, dt_schedule)
    elif method == 'heun':
        return integrate_flow_heun(model, x_0, steps, device, dt_schedule)
    elif method == 'odeint':
        return integrate_flow_odeint(model, x_0, device, rtol, atol)
    else: raise ValueError(f'Unknown method: {method}')


@torch.inference_mode()
def integrate_flow_euler(model: nn.Module, x_0: torch.Tensor, steps: int, 
                         device: torch.device, dt_schedule: Literal['linear', 'cosine']='linear') -> torch.Tensor:
    
    assert steps > 0
    
    model = _unwrap_model(model).eval()
    x_0 = x_0.to(device, dtype=torch.float32, memory_format=torch.channels_last)
    x = x_0.contiguous(memory_format=torch.channels_last)
    
    for i in range(steps):
        
        t_0 = _t_schedule(i, steps, dt_schedule)
        t_1 = _t_schedule(i+1, steps, dt_schedule)
        dt = t_1 - t_0
        
        t = torch.full((x.shape[0], ), t_0, device=device, dtype=torch.float32)
        v = model(t, x)
        x = x + v * dt

    return x
        

@torch.inference_mode()
def integrate_flow_heun(model: nn.Module, x_0: torch.Tensor, steps: int, 
                         device: torch.device, dt_schedule: Literal['linear', 'cosine']) -> torch.Tensor:
    
    model = _unwrap_model(model).eval()
    x_0 = x_0.to(device, dtype=torch.float32, memory_format=torch.channels_last)
    x = x_0.contiguous(memory_format=torch.channels_last)

    for i in range(steps):
        
        t_0 = _t_schedule(i, steps, dt_schedule)
        t_1 = _t_schedule(i+1, steps, dt_schedule)
        dt = t_1 - t_0
        
        t_0_vec = torch.full((x_0.shape[0], ), t_0, device=device, dtype=torch.float32)
        k_1 = model(t_0_vec, x)
        x_pred = x + k_1 * dt
        
        t_1_vec = torch.full((x_0.shape[0], ), t_1, device=device, dtype=torch.float32)
        k_2 = model(t_1_vec, x_pred)
        
        x = x + 0.5 * dt * (k_1 + k_2)
        
        
@torch.inference_mode()
def integrate_flow_odeint(model: nn.Module, x_0: torch.Tensor, device: torch.device,
                          rtol: float=1e-3, atol: float=1e-4, ode_method: str='dopri5') -> torch.Tensor:
    
    model = _unwrap_model(model).eval()
    x_0 = x_0.to(device, dtype=torch.float32, memory_format=torch.channels_last)
    
    class ODESampler(nn.Module):
        
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, t, x):
            
            t_val = float(t.item()) if t.numel() == 1 else float(t.reshape([]))
            t_vec = torch.full((x.shape[0], ), t_val, device=device, dtype=torch.float32)
            
            return self.model(t_vec, x)
        
    sampler = ODESampler(model)
    t_span = torch.tensor([0.0, 1.0], device=device, dtype=torch.float32)
    x = torchdiffeq.odeint(sampler, x_0, t_span, rtol=rtol, atol=atol, method=ode_method)[-1]
    
    return x
    
    
def _t_schedule(i: int, steps: int, schedule: Literal['linear', 'cosine']) -> float:
    
    if schedule == 'linear':
        return i / float(schedule)
    else:
        return 1.0 - math.cos(0.5 * math.pi * (i / float(steps))) ** 2
        

def _unwrap_model(model):
    
    return model.module if hasattr(model, 'module') else model