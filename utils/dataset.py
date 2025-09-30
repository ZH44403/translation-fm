import json
import random
import torch
import torch.utils.data as data
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Literal

class SEN12Dataset(data.Dataset):
    
    def __init__(self, root_dir: Path, data_type : Literal['train, valid, test'], load_from_json: bool=False,
                 image_transform: Optional[Callable]=None, target_transform: Optional[Callable]=None,
                 split_ratio: Tuple=(0.8, 0.1, 0.1), seed: int=42):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        assert self.root_dir.exists(), f'Dataset directory not found {self.root_dir}'
        
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.image_transform = image_transform if image_transform else transform
        self.target_transform = target_transform if target_transform else transform
        
        image_pairs = self._get_image_pairs()
        
        if load_from_json:
            # TODO: load_from_json
            self.image_pairs = self._load_from_json()
        else:
            self.image_pairs = self._random_split(image_pairs, data_type, split_ratio, seed)
        
        
    def _get_image_pairs(self) -> List[Tuple[Path, Path]]:
        
        image_pairs = []
        
        for category in self.root_dir.iterdir():
            
            s1_path = category / 's1'
            # s2_path = category / 's2'
            
            for s1_file in s1_path.glob('*.png'):
                s2_file = Path(str(s1_file).replace('/s1', '/s2').replace('_s1', '_s2'))
                
                assert s1_file.exists(), f'Image not found: {s1_file}'
                assert s2_file.exists(), f'Image not found: {s2_file}'
                
                image_pairs.append((s1_file, s2_file))
                
        return image_pairs
        
        
    def _load_from_json():
        
        return 
    
    def _random_split(image_pair: Tuple[Path, Path], data_type: Literal['train, valid, test'],
                      split_ratio: Tuple[float, float, float], seed: int) -> Tuple[Path, Path]:
        
        assert sum(split_ratio) == 1.0, 'Sum of split ratios must be 1.0'
        
        indices = list(range(len(image_pair)))
        
        random.seed(seed)
        random.shuffle(indices)        
        
        train_end = int(len(indices) * split_ratio[0])
        valid_end = train_end + int(len(indices) * split_ratio[1])
        
        train_indices = indices[:train_end]
        valid_indices = indices[train_end:valid_end]
        test_indices  = indices[valid_end:]


        if data_type == 'train':
            return train_indices
        
        elif data_type == 'valid':
            return valid_indices
        
        else:
            return test_indices
        
        
    def __len__(self):
        
        return len(self.image_pairs)
    
    
    def __getitem__(self, index):
        
        s1_path, s2_path = self.image_pairs[index]
        
        s1_image = Image.open(s1_path).convert('RGB')
        s2_image = Image.open(s2_path).convert('RGB')
        
        s1_image = self.image_transform(s1_image)
        s2_image = self.image_transform(s2_image)
        
        return s1_image, s2_image