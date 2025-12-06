import torch
import torchvision

from pathlib import Path
from typing import List, Literal, Sequence


@torch.inference_mode()
def valid_sample(sar: torch.Tensor, opt: torch.Tensor, pred: torch.Tensor,
                 epoch: int, iter_idx: int, batch_size: int, sample_idx_list: Sequence[int],
                 image_meta: torch.utils.data.Dataset, log_dir: Path, dataset_name: Literal['sen12', 'qxs'],
                 every_n_epochs: int=1, layout: Literal['grid', 'folder']='grid', is_master: bool=True):
    
    if (epoch % every_n_epochs) != 0:
        return
    
    if not is_master:
        return
    
    assert sar.ndim == opt.ndim == pred.ndim == 4
    assert sar.shape[:1] == opt.shape[:1] == pred.shape[:1]
    
    
    if isinstance(image_meta, torch.utils.data.dataset.Subset):
        image_pairs = image_meta.dataset.image_pairs
    else:
        image_pairs = image_meta.image_pairs
            
    start = iter_idx * batch_size
    
    select_idx: List[int] = [(start + idx) for idx in range(batch_size) if (start + idx) in sample_idx_list]
    if len(select_idx) == 0:
        return
    
    save_dir = log_dir / 'samples'
    snap_dir = save_dir / f'epoch_{epoch:04d}'
    snap_dir.mkdir(parents=True, exist_ok=True)
    
    # sar, opt, pred生成一张图片
    if layout == 'grid':
        
        for j in range(len(select_idx)):
            
            tiles = [sar[j], opt[j], pred[j]]
            grid = torchvision.utils.make_grid(torch.stack(tiles, dim=0), nrow=len(tiles), padding=2)
            
            p = Path(image_pairs[start+j][0])
            image_name = p.stem
            
            if dataset_name == 'sen12':
                image_class = p.parts[-3]
                prefix = f'{image_class}_{image_name}'
            elif dataset_name == 'qxs':
                prefix = image_name
            
            torchvision.utils.save_image(grid, str(snap_dir / f'{prefix}.png'))
    
    elif layout == 'folder':
        
        sar_dir = snap_dir / 'sar'
        opt_dir = snap_dir / 'opt'
        pred_dir = snap_dir / 'pred'
        
        sar_dir.mkdir(parents=True, exist_ok=True)
        opt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        for j in range(len(select_idx)):
            
            p = Path(image_pairs[start+j][0])
            image_name = p.stem
            
            if dataset_name == 'sen12':
                image_class = p.parts[-3]
                prefix = f'{image_class}_{image_name}'
            elif dataset_name == 'qxs':
                prefix = image_name
            
            torchvision.utils.save_image(sar[j], str(sar_dir / f'{prefix}.png'))
            torchvision.utils.save_image(opt[j], str(opt_dir / f'{prefix}.png'))
            torchvision.utils.save_image(pred[j], str(pred_dir / f'{prefix}.png'))
    
    return 


@torch.inference_mode()
def test_sample(sar: torch.Tensor, opt: torch.Tensor, pred: torch.Tensor, iter_idx: int, image_meta: torch.utils.data.Dataset, 
               save_dir: Path, dataset_name: Literal['sen12', 'qxs'], need_grid=False):
    
    assert sar.ndim == opt.ndim == pred.ndim == 4        # [b, c, h, w]
    assert sar.shape[:1] == opt.shape[:1] == pred.shape[:1]
    
    
    sar_dir = save_dir / 'sar'
    opt_dir = save_dir / 'opt'
    pred_dir = save_dir / 'pred'
    
    sar_dir.mkdir(parents=True, exist_ok=True)
    opt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    p = Path(image_meta.image_pairs[iter_idx][0])
    image_name = p.stem
    
    if dataset_name == 'sen12':
        image_class = p.parts[-3]
        prefix = f'{image_class}_{image_name}'
    elif dataset_name == 'qxs':
        prefix = image_name

    torchvision.utils.save_image(sar, str(sar_dir / f'{prefix}.png'))
    torchvision.utils.save_image(opt, str(opt_dir / f'{prefix}.png'))
    torchvision.utils.save_image(pred, str(pred_dir / f'{prefix}.png'))
    
    if need_grid:
        
        grid_dir = save_dir / 'grid'
        grid_dir.mkdir(parents=True, exist_ok=True)
        
        for j in range(sar.shape[0]):

            tiles = [sar[j], opt[j], pred[j]]
            grid = torchvision.utils.make_grid(torch.stack(tiles, dim=0), nrow=len(tiles), padding=2)
        
        torchvision.utils.save_image(grid, str(grid_dir / f'{prefix}.png'))
    
    return