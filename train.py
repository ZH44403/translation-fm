import torch
import hydra

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf

from data import dataset
from models import flow, unet
from utils import dist, losses, metrics, utils

# hydra的装饰器，指定配置文件路径和配置文件名
@hydra.main(config_path='configs', config_name='config', version_base='1.3')
def train(args: DictConfig):
    # cfg即配置文件中的内容
    
    device = args.device
    utils.set_seed(args.seed)
    
    train_set = dataset.SEN12Dataset(root_dir=args.dataset.root_dir, data_type='train', split_ratio=args.dataset.split_ratio)
    valid_set = dataset.SEN12Dataset(root_dir=args.dataset.root_dir, data_type='valid', split_ratio=args.dataset.split_ratio)
    
    train_loader = DataLoader(train_set, batch_size=args.dataloader.batch_size, 
                              shuffle=True, num_workers=args.dataloader.num_workers)
    # valid_loader = DataLoader(valid_set, batch_size=args.dataloader.batch_size,
                            #   shuffle=False, num_workers=args.dataloader.num_workers)
    
    # num_classes = len(set(train_set.classes))   # {'urban', 'barren', 'cropland', 'grassland'}
    sar_shape = train_set[0][0].shape           # torch.Size([3, 256, 256])
    opt_shape = train_set[0][1].shape           # torch.Size([3, 256, 256])
    
    model = unet.UNetModel(image_size=sar_shape[1], in_channels=sar_shape[0], model_channels=128, out_channels=opt_shape[0],
                           num_res_blocks=args.model.num_res_blocks, ).to(device)
    # model = torch.compile(model)
    
    ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999))
    
    flow_model = flow.OptimalTransportFlow(args.flow.sigma_min)
    
    loss_func = utils.loss_func(model, flow_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.min_lr)
    scaler = torch.amp.GradScaler()
    
    # checkpoint
    
    
    current_epoch = 0
    step = 0
    # train
    accumulate_steps = args.train.accumulate_steps
    
    for epoch in range(current_epoch, args.train.epochs):
        model.train()
        ema_model.train()
        
        for i, (sar, opt) in enumerate(train_loader):
            sar = sar.to(device)
            
            if i % accumulate_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=device):
                loss = loss_func(sar) / accumulate_steps
            
            scaler.scale(loss).backward()
            
            if (i+1) % accumulate_steps == 0 or (i+1) == len(train_loader):
                scaler.unscale_(optimizer)
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                ema_model.update_parameters(model)
                
                for pg in optimizer.param_groups:
                    pg['lr'] = utils.get_lr(args, step) 
                
                # if (step+1) % args.train.log_interval == 0:
                #     true_loss = loss.item() * accumulate_steps

                step += 1
                
        
if __name__ == '__main__':
    
    train()
    