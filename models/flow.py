import torch

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
    

@torch.no_grad()
def integrate_flow(model, x_0, steps=50, device='cpu'):
    
    x = x_0.clone()
    dt = 1.0 / steps
    
    for i in range(steps):
        
        t = torch.full((x.shape[0], ), i * dt, device=device)
        v = model(t, x)
        x = x + v * dt
        
    return x