# import torch

# from models.unet import UNetModel
# from models.flow import OptimalTransportFlow


# def criterion(model: UNetModel, flow: OptimalTransportFlow):
    
#     def _loss(sar: torch.Tensor, opt: torch.Tensor) -> torch.Tensor:
        
#         assert sar.shape == opt.shape
        
#         t = torch.rand(sar.shape[0], device=sar.device)
#         # generate from noise
#         # x_0 = torch.randn_like(sar)
        
#         x_t, v_true = flow.step(t, sar, opt)
#         v_pred = model(x_t, t)
    
#         return torch.nn.functional.mse_loss(v_pred, v_true)

#     return _loss