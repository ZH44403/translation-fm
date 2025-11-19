# consistency distillation
import torch
import hydra
import pyiqa
import random
import logging

from tqdm import tqdm
from pathlib import Path
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from data import dataset
from models import flow, unet
from utils import dist, losses, utils, sample


# hydra的装饰器，指定配置文件路径和配置文件名
@hydra.main(config_path='configs', config_name='config_distillation', version_base='1.3')
def train(args: DictConfig):
    
    logger = logging.getLogger(__name__)
    
    device = args.device
    utils.set_seed(args.seed)
    
    # consistency distillation log
    log_dir = Path(HydraConfig.get().runtime.output_dir)
    checkpoint_dir = log_dir / 'checkpoint'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_score = float('-inf')
    
    # dataset
    train_set = dataset.SEN12Dataset(root_dir=args.dataset.root_dir, 
                                     data_type='train', split_ratio=args.dataset.split_ratio)
    valid_set = dataset.SEN12Dataset(root_dir=args.dataset.root_dir, 
                                     data_type='valid', split_ratio=args.dataset.split_ratio)
    
    # only use for debugging
    train_set = torch.utils.data.Subset(train_set, range(1280))
    valid_set = torch.utils.data.Subset(valid_set, range(160))
    
    train_loader = DataLoader(train_set, batch_size=args.dataloader.batch_size, 
                              shuffle=True, num_workers=args.dataloader.num_workers)
    valid_loader = DataLoader(valid_set, batch_size=args.dataloader.batch_size,
                              shuffle=False, num_workers=args.dataloader.num_workers)
    
    sar_shape = train_set[0][0].shape           # torch.Size([3, 256, 256])
    opt_shape = train_set[0][1].shape           # torch.Size([3, 256, 256])
    
    # model
    reference_ema = unet.UNetModel(image_size=sar_shape[1], in_channels=sar_shape[0],
                                     model_channels=args.model.num_channels, out_channels=opt_shape[0],
                                     num_res_blocks=args.model.num_res_blocks, ).to(device).float()
    distillation_model = unet.UNetModel(image_size=sar_shape[1], in_channels=sar_shape[0],
                                     model_channels=args.model.num_channels, out_channels=opt_shape[0],
                                     num_res_blocks=args.model.num_res_blocks, ).to(device).float()
    
    # 加载第一阶段的ema模型权重
    current_epoch, reference_ema, _, _ = utils.load_checkpoint(Path(args.distillation.checkpoint_path), reference_ema)
    distillation_model.load_state_dict(reference_ema.state_dict())
    
    # 冻结reference_ema的参数
    reference_ema.eval()
    for p in reference_ema.parameters():
        p.requires_grad = False
        
    distillation_ema = optim.swa_utils.AveragedModel(distillation_model, multi_avg_fn=optim.swa_utils.get_ema_multi_avg_fn(0.9999))
    
    flow_model = flow.PairInterpolantFlow()
    
    # loss
    velocity_loss = losses.CharbonnierLoss(eps=1e-3)
    distillation_loss = losses.CharbonnierLoss(eps=1e-3)
    lpips_loss = pyiqa.create_metric('lpips', device=device, as_loss=True)
    
    optimizer = optim.Adam(distillation_model.parameters(), lr=args.train.min_lr)

    psnr  = pyiqa.create_metric('psnr' , device=device, test_y_channel=False)
    ssim  = pyiqa.create_metric('ssimc', device=device)
    lpips = pyiqa.create_metric('lpips', device=device)
    
    sample_idx_list = sorted(random.sample(range(0, len(valid_set)), k=args.valid.sample_num))
    # accumulate_steps = args.train.accumulate_steps

    for epoch in range(current_epoch, (args.train.epochs+1)):
        
        distillation_model.train()
        distillation_ema.eval()
        
        train_loss              = 0.0
        train_loss_velocity     = 0.0
        train_loss_distillation = 0.0
        train_loss_lpips        = 0.0
        
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                         desc=f'Epoch {epoch}/{args.train.epochs}', dynamic_ncols=True)
        
        for i, (sar, opt) in train_bar:
            
            sar = sar.to(device, dtype=torch.float32)
            opt = opt.to(device, dtype=torch.float32) 
            
            # if i % accumulate_steps == 0:
            optimizer.zero_grad(set_to_none=True)
            
            # velocity loss (可选)
            loss_vel = torch.tensor(0.0, device=device)
            if args.lambdas.velocity > 0.0:
                
                t = torch.rand(sar.shape[0], device=device)
                x_t, v_true = flow_model.step(t, sar, opt)
                v_pred = distillation_model(t, x_t)
                loss_velocity = velocity_loss(v_pred, v_true)
            
            # 阶段一采样, 不反向传播
            with torch.no_grad():
                opt_reference = flow.integrate_flow(reference_ema, sar, args.flow.reference_steps, device=device, 
                                                    method=args.flow.integrate_method, dt_schedule=args.flow.dt_schedule)
            
            # 阶段二采样, 反向传播
            opt_distillation = flow.integrate_flow(distillation_model, sar, args.flow.distillation_steps, device=device, 
                                              method=args.flow.integrate_method, dt_schedule=args.flow.dt_schedule)
            
            loss_distillation = distillation_loss(opt_distillation, opt_reference)
            loss_lpips = lpips_loss(opt_distillation, opt)
            
            loss = (
                args.lambdas.velocity     * loss_velocity + 
                args.lambdas.distillation * loss_distillation + 
                args.lambdas.lpips        * loss_lpips
            )
            
            loss.backward()
            optimizer.step()
            distillation_ema.update_parameters(distillation_model)
            
            train_loss              += loss.item()
            train_loss_velocity     += loss_velocity.item()
            train_loss_distillation += loss_distillation.item()
            train_loss_lpips        += loss_lpips.item()
            
            train_bar.set_postfix(
                loss  = f'{train_loss/(i+1):.4f}',
                vel   = f'{train_loss_velocity/(i+1):.4f}',
                dist  = f'{train_loss_distillation/(i+1):.4f}',
                lpips = f'{train_loss_lpips/(i+1):.4f}'
            )
        
        avg_train_loss              = train_loss              / len(train_loader)
        avg_train_loss_velocity     = train_loss_velocity     / len(train_loader)
        avg_train_loss_distillation = train_loss_distillation / len(train_loader)
        avg_train_loss_lpips        = train_loss_lpips        / len(train_loader)
        
        logger.info(f'[Train] Epoch {epoch}: loss {avg_train_loss:.4f}, vel {avg_train_loss_velocity:.4f}, \
                                             dist {avg_train_loss_distillation:.4f}, lpips {avg_train_loss_lpips:.4f}')
        
        

if __name__ == '__main__':
    
    train()