import torch
import hydra

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf

from models import flow, unet
from utils import dataset, dist, losses, metrics, utils

# hydra的装饰器，指定配置文件路径和配置文件名
@hydra.main(config_path='configs', config_name='config', version_base='1.3')
def train(args: DictConfig):
    # cfg即配置文件中的内容
    
    device = torch.device(args.device)
    utils.set_seed(args.seed)
    
    train_set = dataset.SEN12Dataset(root_dir=args.dataset.root_dir, data_type='train')
    valid_set = dataset.SEN12Dataset(root_dir=args.dataset.root_dir, data_type='valid')
    
    train_loader = DataLoader(train_set, batch_size=args.dataloader.batch_size, 
                              shuffle=True, num_workers=args.dataloader.num_workers)
    valid_loader = DataLoader(valid_set, batch_size=args.dataloader.batch_size,
                              shuffle=False, num_workers=args.dataloader.num_workers)
    
    num_classes = len(set(train_set.classes))   # {'urban', 'barren', 'cropland', 'grassland'}
    sar_shape = train_set[0][0].shape         # torch.Size([3, 256, 256])
    opt_shape = train_set[0][1].shape         # torch.Size([3, 256, 256])
    
    model = unet.UNetModel(image_size=sar_shape[1], in_channels=sar_shape[0], out_channels=opt_shape[0])


if __name__ == '__main__':
    
    train()
    