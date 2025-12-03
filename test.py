import yaml
import time
import torch
import pyiqa
import logging

from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path

from data import dataset
from utils import utils, sample
from models import flow, unet

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def test():
    
    test_root = Path('outputs/sen12/distillation/20251128/211204')
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO) 
    file_handler = logging.FileHandler(test_root / 'test.log', mode='a', encoding='utf-8')
    logger.addHandler(file_handler)
    
    yaml_path       = test_root / '.hydra' / 'config.yaml'
    checkpoint_path = test_root / 'checkpoint' / 'best_distillation.pth'
    save_path = test_root / 'test'
    save_path.mkdir(parents=True, exist_ok=True)
    
    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    device = config['device']
    
    dataset_path = config['dataset']['root_dir']
    split_ratio  = config['dataset']['split_ratio']
    
    distillation_steps = config['flow']['distillation_steps']
    integrate_method   = config['flow']['integrate_method']
    dt_schedule        = config['flow']['dt_schedule']
    
    num_channels   = config['model']['num_channels']
    num_res_blocks = config['model']['num_res_blocks']
    
    if config['dataset']['name'] == 'sen12':
        test_set = dataset.SEN12Dataset(root_dir=dataset_path, data_type='test', json_path=r'./data/sen12.json', split_ratio=split_ratio)
        
    elif config['dataset']['name'] == 'qxs':
        test_set = dataset.QXSDataset(root_dir=dataset_path, data_type='test', json_path=r'./data/qxs.json', split_ratio=split_ratio)
    
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)
    
    sar_shape = test_set[0][0].shape           # torch.Size([3, 256, 256])
    opt_shape = test_set[0][1].shape           # torch.Size([3, 256, 256])
    
    # model
    test_ema = unet.UNetModel(image_size=sar_shape[1], in_channels=sar_shape[0],
                              model_channels=num_channels, out_channels=opt_shape[0],
                              num_res_blocks=num_res_blocks, dropout=0.1).to(device).float()
    
    # 加载蒸馏ema模型权重
    test_ema, _, _, epoch = utils.load_checkpoint(Path(checkpoint_path), test_ema)
    
    # image_loss = losses.CharbonnierLoss(eps=1e-3)
    
    psnr  = pyiqa.create_metric('psnr' , device=device, test_y_channel=False)
    ssim  = pyiqa.create_metric('ssimc', device=device)
    lpips = pyiqa.create_metric('lpips', device=device)
    fid   = pyiqa.create_metric('fid'  , device=device)

    test_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Test', dynamic_ncols=True)
    
    test_psnr  = 0.0
    test_lpips = 0.0
    test_ssim  = 0.0
    test_fid   = 0.0
    
    start_time = time.time()
    with torch.no_grad():
        for i, (sar, opt) in test_bar:
            
            sar = sar.to(device, dtype=torch.float32)
            opt = opt.to(device, dtype=torch.float32) 
            
            opt_pred = flow.integrate_flow(test_ema, sar, distillation_steps, device=device,
                                        method=integrate_method, dt_schedule=dt_schedule)
            
            opt_pred_01 = utils.to_01(opt_pred)
            opt_01 = utils.to_01(opt)
            
            test_psnr  += torch.mean(psnr(opt_pred_01, opt_01)).item()
            test_lpips += torch.mean(lpips(opt_pred_01, opt_01)).item()
            test_ssim  += torch.mean(ssim(opt_pred_01, opt_01)).item()
            
            avg_psnr  = test_psnr  / len(test_loader)
            avg_lpips = test_lpips / len(test_loader)
            avg_ssim  = test_ssim  / len(test_loader)
            
            sample.test_sen12(sar, opt, opt_pred, i, test_set, save_path, need_grid=True)
        
    end_time = time.time()
    speed = (end_time - start_time) / len(test_loader)
    
    test_fid = fid(save_path / 'pred', save_path / 'opt')
    
    logger.info(f'[Test] Epoch {epoch} psnr {avg_psnr:.2f}, lpips {avg_lpips:.4f}, ssim {avg_ssim:.4f}, fid {test_fid:.4f}, speed {speed:.4f}')        
        
if __name__ == '__main__':
    
    test()