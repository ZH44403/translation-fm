import torch
import hydra
import pyiqa

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf

from data import dataset
from models import flow, unet
from utils import dist, losses, metrics, utils

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# hydra的装饰器，指定配置文件路径和配置文件名
@hydra.main(config_path='configs', config_name='config', version_base='1.3')
def train(args: DictConfig):
    # cfg即配置文件中的内容
    
    device = args.device
    utils.set_seed(args.seed)
    
    train_set = dataset.SEN12Dataset(root_dir=args.dataset.root_dir, 
                                     data_type='train', split_ratio=args.dataset.split_ratio)
    valid_set = dataset.SEN12Dataset(root_dir=args.dataset.root_dir, 
                                     data_type='valid', split_ratio=args.dataset.split_ratio)
    # debug
    # train_sub = torch.utils.data.Subset(train_set, range(32))
    # valid_sub = torch.utils.data.Subset(valid_set, range(32))
    
    # train_loader = DataLoader(train_sub, batch_size=args.dataloader.batch_size, 
    #                           shuffle=True, num_workers=args.dataloader.num_workers)
    # valid_loader = DataLoader(valid_sub, batch_size=args.dataloader.batch_size,
    #                           shuffle=False, num_workers=args.dataloader.num_workers)
    
    train_loader = DataLoader(train_set, batch_size=args.dataloader.batch_size, 
                              shuffle=True, num_workers=args.dataloader.num_workers)
    valid_loader = DataLoader(valid_set, batch_size=args.dataloader.batch_size,
                              shuffle=False, num_workers=args.dataloader.num_workers)
    
    
    # num_classes = len(set(train_set.classes))   # {'urban', 'barren', 'cropland', 'grassland'}
    sar_shape = train_set[0][0].shape           # torch.Size([3, 256, 256])
    opt_shape = train_set[0][1].shape           # torch.Size([3, 256, 256])
    
    model = unet.UNetModel(image_size=sar_shape[1], in_channels=sar_shape[0], 
                           model_channels=128, out_channels=opt_shape[0],
                           num_res_blocks=args.model.num_res_blocks, ).to(device).float()
    # model = torch.compile(model)
    
    ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999))
    
    flow_model = flow.GaussianBridgeFlow(args.flow.sigma_min)
    
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.min_lr)
    scaler = torch.amp.GradScaler()
    
    psnr = pyiqa.create_metric('psnr', device=device, test_y_channel=False)
    lpips = pyiqa.create_metric('lpips', device=device)
    msssim = pyiqa.create_metric('ms_ssim', device=device, test_y_channel=False)
    
    # checkpoint
    
    
    current_epoch = 0
    step = 0

    accumulate_steps = args.train.accumulate_steps
    
    for epoch in range(current_epoch, args.train.epochs):
        
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
            
            train_loss = mse(v_pred, v_true) / accumulate_steps
            
            # scaler.scale(train_loss).backward()
            train_loss.backward()
            
            if (i+1) % accumulate_steps == 0 or (i+1) == len(train_loader):
                
                # scaler.unscale_(optimizer)
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        
        
        # validation
        model.eval()
        ema_model.eval()
        
        n_images = 0
        
        valid_loss_sum = 0.0
        valid_psnr_sum = 0.0
        valid_lpips_sum = 0.0
        valid_mmssim_sum = 0.0
        
        valid_bar = tqdm(valid_loader, total=len(valid_loader), 
                         desc=f'Epoch {epoch}/{args.train.epochs}', dynamic_ncols=True)
        
        with torch.no_grad():
            
            # autocast_ctx = torch.amp.autocast(device_type=device)
            
            for sar, opt in valid_bar:
                
                sar = sar.to(device, dtype=torch.float32)
                opt = opt.to(device, dtype=torch.float32)

                batch = sar.shape[0]
                n_images += batch
                
                # 速度场验证
                t = torch.rand(batch, device=device)
                x_t, v_true = flow_model.step(t, sar, opt)
                v_pred = model(t, x_t)
                
                valid_loss = mse(v_pred, v_true)
                valid_loss_sum += valid_loss.item() * batch
                
                # 图像质量验证
                # opt_pred  = flow.integrate_flow(ema_model, sar, args.train.eval_steps, device=device)
                opt_pred  = flow.integrate_flow(ema_model, sar, args.train.eval_steps, device=device)
                opt_pred  = opt_pred.clamp(0, 1)
                opt_clamp = opt.clamp(0, 1)
                
                valid_psnr_sum   += psnr(opt_pred, opt_clamp).sum().item()
                valid_lpips_sum  += lpips(opt_pred, opt_clamp).sum().item()
                valid_mmssim_sum += msssim(opt_pred, opt_clamp).sum().item()
                
                avg_loss   = valid_loss_sum / n_images
                avg_psnr   = valid_psnr_sum / n_images
                avg_lpips  = valid_lpips_sum / n_images
                avg_mmssim = valid_mmssim_sum / n_images
                
                valid_bar.set_postfix(v_loss=f'{avg_loss:.4f}', psnr=f'{avg_psnr:.2f}', 
                                      lpips=f'{avg_lpips:.4f}', mmssim=f'{avg_mmssim:.4f}')
        
if __name__ == '__main__':
    
    train()
    