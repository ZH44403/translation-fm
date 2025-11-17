import torch
import hydra
import pyiqa
import random

from tqdm import tqdm
from pathlib import Path
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from data import dataset
from models import flow, unet
from utils import dist, losses, metrics, utils, sample

# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# hydra的装饰器，指定配置文件路径和配置文件名
@hydra.main(config_path='configs', config_name='config', version_base='1.3')
def train(args: DictConfig):
    # cfg即配置文件中的内容
    
    device = args.device
    utils.set_seed(args.seed)
    log_dir = Path(HydraConfig.get().runtime.output_dir)
    
    train_set = dataset.SEN12Dataset(root_dir=args.dataset.root_dir, 
                                     data_type='train', split_ratio=args.dataset.split_ratio)
    valid_set = dataset.SEN12Dataset(root_dir=args.dataset.root_dir, 
                                     data_type='valid', split_ratio=args.dataset.split_ratio)
    # debug
    # train_set = torch.utils.data.Subset(train_set, range(1280))
    # valid_set = torch.utils.data.Subset(valid_set, range(160))
     
    train_loader = DataLoader(train_set, batch_size=args.dataloader.batch_size, 
                              shuffle=True, num_workers=args.dataloader.num_workers)
    valid_loader = DataLoader(valid_set, batch_size=args.dataloader.batch_size,
                              shuffle=False, num_workers=args.dataloader.num_workers)
    
    
    # num_classes = len(set(train_set.classes))   # {'urban', 'barren', 'cropland', 'grassland'}
    sar_shape = train_set[0][0].shape           # torch.Size([3, 256, 256])
    opt_shape = train_set[0][1].shape           # torch.Size([3, 256, 256])
    
    model = unet.UNetModel(image_size=sar_shape[1], in_channels=sar_shape[0], 
                           model_channels=args.model.num_channels, out_channels=opt_shape[0],
                           num_res_blocks=args.model.num_res_blocks, ).to(device).float()
    # model = torch.compile(model)
    
    ema_model = optim.swa_utils.AveragedModel(model, multi_avg_fn=optim.swa_utils.get_ema_multi_avg_fn(0.9999))
    
    # flow_model = flow.GaussianBridgeFlow(args.flow.sigma_min)
    flow_model = flow.PairInterpolantFlow()
    
    # vel_loss = nn.MSELoss()
    vel_loss = losses.CharbonnierLoss(eps=1e-3)
    
    optimizer = optim.Adam(model.parameters(), lr=args.train.min_lr)
    scaler = torch.amp.GradScaler()
    
    psnr = pyiqa.create_metric('psnr', device=device, test_y_channel=False)
    lpips = pyiqa.create_metric('lpips', device=device)
    # msssim = pyiqa.create_metric('ms_ssim', device=device, test_y_channel=False)
    ssim = pyiqa.create_metric('ssim', device=device, test_y_channel=False)

    # checkpoint
    
    sample_idx_list = sorted(random.sample(range(0, len(valid_set)), k=args.valid.sample_num))
    
    # checkpoint
    
    
    current_epoch = 1
    step = 0

    accumulate_steps = args.train.accumulate_steps
    
    for epoch in range(current_epoch, (args.train.epochs+1)):
        
        # training
        model.train()
        ema_model.eval()
        
        train_loss_sum = 0.0
        
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                         desc=f'Epoch {epoch}/{args.train.epochs}', dynamic_ncols=True)
        
        for i, (sar, opt) in train_bar:
            
            sar = sar.to(device, dtype=torch.float32)
            opt = opt.to(device, dtype=torch.float32)
            
            if i % accumulate_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            
            # with torch.amp.autocast(device_type=device):
                
            t = torch.rand(sar.shape[0], device=device)
            
            # generate from noise
            # x_0 = torch.randn_like(sar)
            
            assert sar.shape == opt.shape
            x_t, v_true = flow_model.step(t, sar, opt)
            v_pred = model(t, x_t)
            
            train_loss = vel_loss(v_pred, v_true) / accumulate_steps
            
            # scaler.scale(train_loss).backward()
            train_loss.backward()
            
            if (i+1) % accumulate_steps == 0 or (i+1) == len(train_loader):
                
                # scaler.unscale_(optimizer)
                grad = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # scaler.step(optimizer)
                # scaler.update()
                optimizer.step()
                
                ema_model.update_parameters(model)
                
                # learning rate schedule
                # for pg in optimizer.param_groups:
                #     pg['lr'] = utils.get_lr(args, step) 

                step += 1
            
            train_loss_sum += train_loss.item() * accumulate_steps
            train_bar.set_postfix(loss=f'{train_loss_sum / (i+1):.4f}', )
        
        # 每隔valid_interval个epoch进行一次验证，节省时间
        if epoch % args.valid.valid_interval == 0:
        # validation
            model.eval()
            ema_model.eval()
            
            n_images = 0
            
            valid_loss_sum = 0.0
            valid_psnr_sum = 0.0
            valid_lpips_sum = 0.0
            valid_ssim_sum = 0.0
            
            valid_bar = tqdm(enumerate(valid_loader), total=len(valid_loader), 
                            desc=f'Epoch {epoch}/{args.train.epochs}', dynamic_ncols=True)
            
            with torch.no_grad():
                
                # autocast_ctx = torch.amp.autocast(device_type=device)
                
                for i, (sar, opt) in valid_bar:
                    
                    sar = sar.to(device, dtype=torch.float32)
                    opt = opt.to(device, dtype=torch.float32)

                    batch = sar.shape[0]
                    n_images += batch
                    
                    # 速度场验证
                    t = torch.rand(batch, device=device)
                    x_t, v_true = flow_model.step(t, sar, opt)
                    v_pred = model(t, x_t)
                    
                    valid_loss = vel_loss(v_pred, v_true)
                    valid_loss_sum += valid_loss.item() * batch
                    
                    # 图像质量验证
                    opt_pred  = flow.integrate_flow(ema_model, sar, args.flow.eval_steps, device=device, 
                                                    method=args.valid.integrate_method, dt_schedule=args.valid.dt_schedule)
                    opt_pred  = opt_pred.clamp(0, 1)
                    opt_clamp = opt.clamp(0, 1)
                    
                    valid_psnr_sum   += psnr(opt_pred, opt_clamp).sum().item()
                    valid_lpips_sum  += lpips(opt_pred, opt_clamp).sum().item()
                    valid_ssim_sum   += ssim(opt_pred, opt_clamp).sum().item()
                    
                    avg_loss   = valid_loss_sum / n_images
                    avg_psnr   = valid_psnr_sum / n_images
                    avg_lpips  = valid_lpips_sum / n_images
                    avg_ssim   = valid_ssim_sum / n_images
                    
                    
                    sample.sample_sen12(sar, opt, opt_pred, epoch, i, sar.shape[0], 
                                        sample_idx_list, valid_set, log_dir, every_n_epochs=args.valid.sample_interval)
                    
                    valid_bar.set_postfix(v_loss=f'{avg_loss:.4f}', psnr=f'{avg_psnr:.2f}', 
                                        lpips=f'{avg_lpips:.4f}', ssim=f'{avg_ssim:.4f}')
        
if __name__ == '__main__':
    
    train()
    