import torch
import torchvision

from pathlib import Path
from typing import List, Literal, Sequence


@torch.inference_mode()
def sample_sen12(sar: torch.Tensor, opt: torch.Tensor, pred: torch.Tensor,
                 epoch: int, iter_idx: int, batch_size: int, sample_idx_list: Sequence[int],
                 image_meta: torch.utils.data.Dataset, log_dir: Path, 
                 every_n_epochs: int=1, layout: Literal['grid', 'separate']='grid', is_master: bool=True):
    
    if (epoch % every_n_epochs) != 0:
        return
    
    if not is_master:
        return
    
    assert sar.ndim == opt.ndim == pred.ndim == 4
    assert sar.shape[:1] == opt.shape[:1] == pred.shape[:1]
    
    start = iter_idx * batch_size
    
    select_idx: List[int] = [(start + idx) for idx in range(batch_size) if (start + idx) in sample_idx_list]
    if len(select_idx) == 0:
        return
    
    save_dir = log_dir / 'samples'
    snap_dir = save_dir / f"epoch_{epoch:04d}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    
    # sar, opt, pred生成一张图片
    if layout == 'grid':
        for j in range(len(select_idx)):
            
            tiles = [sar[j], opt[j], pred[j]]
            grid = torchvision.utils.make_grid(torch.stack(tiles, dim=0), nrow=len(tiles), padding=2)
            
            p = Path(image_meta.image_pairs[start+j][0])
            image_class = p.parts[-3]
            image_name = p.stem
            
            out_path = snap_dir / f"{image_class}_{image_name}.png"
            torchvision.utils.save_image(grid, str(out_path))
    
    # sar, opt, pred分别存储
    else:
        for j in range(len(select_idx)):
            
            p = Path(image_meta.image_pairs[start+j][0])
            image_class = p.parts[-3]
            image_name = p.stem
            
            p = snap_dir / f'{image_class}_{image_name}_sar.png'
            torchvision.utils.save_image(sar[j], str(p))

            p = snap_dir / f'{image_class}_{image_name}_opt.png'
            torchvision.utils.save_image(opt[j], str(p))

            p = snap_dir / f'{image_class}_{image_name}_pred.png'
            torchvision.utils.save_image(pred[j], str(p))
    
    return 