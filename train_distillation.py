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

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


# hydra装饰器，指定配置文件路径和配置文件名
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
    # train_set = torch.utils.data.Subset(train_set, range(40))
    # valid_set = torch.utils.data.Subset(valid_set, range(20))
    
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
    _, reference_ema, _, _ = utils.load_checkpoint(Path(args.distillation.checkpoint_path), reference_ema)
    distillation_model.load_state_dict(reference_ema.state_dict())
    
    # 冻结reference_ema的参数
    reference_ema.eval()
    for p in reference_ema.parameters():
        p.requires_grad = False
        
    distillation_ema = optim.swa_utils.AveragedModel(distillation_model, multi_avg_fn=optim.swa_utils.get_ema_multi_avg_fn(0.9999))
    
    flow_model = flow.PairInterpolantFlow()
    
    # loss
    velocity_loss     = losses.CharbonnierLoss(eps=1e-3)
    distillation_loss = losses.CharbonnierLoss(eps=1e-3)
    image_loss        = losses.CharbonnierLoss(eps=1e-3)
    lpips_loss        = pyiqa.create_metric('lpips', device=device, as_loss=True)
    ssim_loss         = pyiqa.create_metric('ssimc', device=device, as_loss=True)
    
    optimizer = optim.Adam(distillation_model.parameters(), lr=args.train.min_lr)

    psnr  = pyiqa.create_metric('psnr' , device=device, test_y_channel=False)
    ssim  = pyiqa.create_metric('ssimc', device=device)
    lpips = pyiqa.create_metric('lpips', device=device)
    
    sample_idx_list = sorted(random.sample(range(0, len(valid_set)), k=min(len(valid_set), args.valid.sample_num)))
    # accumulate_steps = args.train.accumulate_steps

    current_epoch = 1
    
    for epoch in range(current_epoch, args.train.epochs+1):
        
        distillation_model.train()
        distillation_ema.eval()
        
        train_loss              = 0.0
        train_loss_velocity     = 0.0
        train_loss_distillation = 0.0
        train_loss_lpips        = 0.0
        train_loss_ssim         = 0.0
        train_loss_image        = 0.0
        
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                         desc=f'Epoch {epoch}/{args.train.epochs}', dynamic_ncols=True)
        
        for i, (sar, opt) in train_bar:
            
            sar = sar.to(device, dtype=torch.float32)
            opt = opt.to(device, dtype=torch.float32) 
            
            # if i % accumulate_steps == 0:
            optimizer.zero_grad(set_to_none=True)
            
            # velocity loss (可选)
            if args.lambdas.velocity > 0.0:
                
                t = torch.rand(sar.shape[0], device=device)
                x_t, v_true = flow_model.step(t, sar, opt)
                v_pred = distillation_model(t, x_t)
                loss_velocity = velocity_loss(v_pred, v_true)
                
            else:
                loss_velocity = torch.tensor(0.0, device=device)
            
            # 阶段一采样, 不反向传播
            if args.lambdas.distillation > 0.0:
                with torch.no_grad():
                    opt_reference = flow.integrate_flow(reference_ema, sar, args.flow.reference_steps, device=device, 
                                                        method=args.flow.integrate_method, dt_schedule=args.flow.dt_schedule)
            
            # 阶段二采样, 反向传播
            opt_distillation = flow.integrate_flow(distillation_model, sar, args.flow.distillation_steps, device=device, 
                                              method=args.flow.integrate_method, dt_schedule=args.flow.dt_schedule)
            
            opt_distillation_01 = utils.to_01(opt_distillation)
            opt_reference_01 = utils.to_01(opt_reference)
            opt_01 = utils.to_01(opt)
            
            loss_distillation = distillation_loss(opt_distillation_01, opt_reference_01)
            loss_lpips = lpips_loss(opt_distillation_01, opt_01)
            loss_ssim  = ssim_loss(opt_distillation_01, opt_01)
            loss_image = image_loss(opt_distillation_01, opt_01)
            
            loss = (
                args.lambdas.velocity     * loss_velocity + 
                args.lambdas.distillation * loss_distillation + 
                args.lambdas.lpips        * loss_lpips + 
                args.lambdas.ssim         * loss_ssim +
                args.lambdas.image        * loss_image
            )
            
            # 反向传播的是加权的loss
            loss.backward()
            optimizer.step()
            distillation_ema.update_parameters(distillation_model)
            
            train_loss              += loss.item()
            train_loss_velocity     += loss_velocity.item()
            train_loss_distillation += loss_distillation.item()
            train_loss_lpips        += loss_lpips.item()
            train_loss_image        += loss_image.item()
            train_loss_ssim         += loss_ssim.item()
            
            train_bar.set_postfix(
                # loss  = f'{train_loss/(i+1):.4f}',
                vel   = f'{train_loss_velocity/(i+1):.4f}',
                img   = f'{train_loss_image/(i+1):.4f}',
                lpips = f'{train_loss_lpips/(i+1):.4f}',
                dist  = f'{train_loss_distillation/(i+1):.4f}',
                ssim  = f'{train_loss_ssim/(i+1):.4f}'
            )
        
        avg_train_loss              = train_loss              / len(train_loader)
        avg_train_loss_velocity     = train_loss_velocity     / len(train_loader)
        avg_train_loss_distillation = train_loss_distillation / len(train_loader)
        avg_train_loss_lpips        = train_loss_lpips        / len(train_loader)
        
        logger.info(f'[Train] Epoch {epoch}: loss {avg_train_loss:.4f}, vel {avg_train_loss_velocity:.4f}, lpips {avg_train_loss_lpips:.4f}, dist {avg_train_loss_distillation:.4f}')
        
        if epoch % args.valid.valid_interval != 0:
            continue
        
        distillation_model.eval()
        distillation_ema.eval()
        
        # valid_loss          = 0.0
        valid_loss_velocity = 0.0
        valid_loss_lpips    = 0.0
        valid_loss_image    = 0.0
        
        valid_psnr  = 0.0
        valid_lpips = 0.0
        valid_ssim  = 0.0
        
        valid_bar = tqdm(enumerate(valid_loader), total=len(valid_loader), 
                         desc=f'Epoch {epoch}/{args.train.epochs}', dynamic_ncols=True)
        
        with torch.no_grad():
            
            for i, (sar, opt) in valid_bar:
                
                sar = sar.to(device, dtype=torch.float32)
                opt = opt.to(device, dtype=torch.float32)
                
                opt_pred = flow.integrate_flow(distillation_ema, sar, args.flow.distillation_steps, device=device,
                                              method=args.flow.integrate_method, dt_schedule=args.flow.dt_schedule)
                
                # velocity loss (可选)
                if args.lambdas.velocity > 0.0:
                    
                    t = torch.rand(sar.shape[0], device=device)
                    x_t, v_true = flow_model.step(t, sar, opt)
                    v_pred = distillation_model(t, x_t)
                    loss_velocity = velocity_loss(v_pred, v_true)
                    
                else:
                    loss_velocity = torch.tensor(0.0, device=device)
                
                opt_pred_01 = utils.to_01(opt_pred)
                opt_01 = utils.to_01(opt)
                
                loss_lpips = lpips_loss(opt_pred_01, opt_01)
                loss_img = image_loss(opt_pred_01, opt_01)
                
                valid_loss_velocity += loss_velocity.item()
                valid_loss_lpips    += loss_lpips.item()
                valid_loss_image += loss_img.item()
                
                valid_psnr  += torch.mean(psnr(opt_pred_01, opt_01)).item()
                valid_lpips += torch.mean(lpips(opt_pred_01, opt_01)).item()
                valid_ssim  += torch.mean(ssim(opt_pred_01, opt_01)).item()

                sample.sample_sen12(sar, opt, opt_pred, epoch, i, sar.shape[0], 
                                    sample_idx_list, valid_set, log_dir, every_n_epochs=args.valid.sample_interval)
                
                valid_bar.set_postfix(l_img=f'{valid_loss_image/(i+1):.4f}',
                                    #   l_lpips=f'{valid_loss_lpips/(i+1):.4f}',
                                      v=f'{valid_loss_velocity/(i+1):.4f}', 
                                      psnr=f'{valid_psnr/(i+1):.2f}', 
                                      lpips=f'{valid_lpips/(i+1):.4f}', 
                                      ssim=f'{valid_ssim/(i+1):.4f}')
                
            avg_valid_loss_velocity = valid_loss_velocity / len(valid_loader)
            avg_valid_loss_lpips    = valid_loss_lpips    / len(valid_loader)
            avg_valid_loss_image    = valid_loss_image    / len(valid_loader)
            
            avg_psnr  = valid_psnr  / len(valid_loader)
            avg_lpips = valid_lpips / len(valid_loader)
            avg_ssim  = valid_ssim  / len(valid_loader)

            valid_score = utils.compute_valid_score(avg_psnr, avg_ssim, avg_lpips)
                
            metrics_epoch = {
                'vel'       : avg_valid_loss_velocity,
                'image'     : avg_valid_loss_image,
                'lpips_loss': avg_valid_loss_lpips,
                'psnr'      : avg_psnr,
                'lpips'     : avg_lpips,
                'ssim'      : avg_ssim,
            }
            logger.info(f'[Valid] Epoch {epoch}: vel {avg_valid_loss_velocity:.4f}, l_lpips {avg_valid_loss_lpips:.4f}, image {avg_valid_loss_image:.4f}, psnr {avg_psnr:.2f}, lpips {avg_lpips:.4f}, ssim {avg_ssim:.4f}')
            
            if valid_score > best_score:
                
                best_score = valid_score
                utils.save_checkpoint(checkpoint_dir/'best_distillation.pth', epoch, 
                                      distillation_model, distillation_ema, optimizer, args, metrics_epoch)
                logger.info(f'New best score: {best_score:.4f}')
                
        utils.save_checkpoint(checkpoint_dir/'last_distillation.pth', epoch, 
                              distillation_model, distillation_ema, optimizer, args, metrics_epoch)

if __name__ == '__main__':
    
    train()