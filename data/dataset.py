import json
import random
import torch
import torch.utils.data as data
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Literal

class SEN12Dataset(data.Dataset):
    
    def __init__(self, root_dir: Path, data_type : Literal['train, valid, test'], 
                 json_path: str='./data/sen12.json',
                 image_transform: Optional[Callable]=None, target_transform: Optional[Callable]=None,
                 split_ratio: List=[0.8, 0.1, 0.1], seed: int=114514):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        assert self.root_dir.exists(), f'Dataset directory not found {self.root_dir}'
        
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        
        self.image_transform = image_transform if image_transform else transform
        self.target_transform = target_transform if target_transform else transform
        
        # load_from_json和split_ratio取出来的image_pairs和classes的index不是一致的
        if json_path is not None:
            self.image_pairs, self.classes = self._load_from_json(json_path, data_type)
        else:
            image_pairs, classes = self._get_image_pairs()
            self.image_pairs, self.classes = self._random_split(image_pairs, classes, data_type, split_ratio, seed)
        
        
    @staticmethod
    def _load_from_json(json_path: str, data_type: Literal['train, valid, test']):
        
        image_pairs = []
        classes = []
        
        with open(json_path, 'r') as f:
            dataset = json.load(f)
            
        # 按照data_type取数据集
        for item in dataset[data_type]:
            image_pairs.append((item['path'][0], item['path'][1]))
            classes.append(item['class'])
    
        return image_pairs, classes
        
    
    @staticmethod
    def _random_split(image_pairs: Tuple[Path, Path], 
                      classes: List[str],
                      data_type: Literal['train, valid, test'],
                      split_ratio: Tuple[float, float, float], seed: int) -> Tuple[Tuple[Path, Path], List[str]]:
        
        assert sum(split_ratio) == 1.0, 'Sum of split ratios must be 1.0'
        
        indices = list(range(len(image_pairs)))
        
        random.seed(seed)
        random.shuffle(indices)        
        
        train_end = int(len(indices) * split_ratio[0])
        valid_end = train_end + int(len(indices) * split_ratio[1])
        
        index_map = {
            'train': indices[:train_end],
            'valid': indices[train_end:valid_end],
            'test' : indices[valid_end:]
        }

        try:
            idx = index_map[data_type]
            
        except KeyError:
            raise ValueError(f'data_type must be one of {tuple(index_map)}')

        # 按照data_type取数据集
        return [image_pairs[i] for i in idx], [classes[i] for i in idx]
    
    
    def _get_image_pairs(self) -> Tuple[Tuple[Path, Path], List[str]]:
        
        image_pairs = []
        classes = []
        
        for class_name in sorted(p for p in self.root_dir.iterdir() if p.is_dir()):
            
            s1_path = class_name / 's1'
            # s2_path = class_name / 's2'
            
            for s1_file in sorted(s1_path.glob('*.png')):
                s2_file = Path(str(s1_file).replace('/s1', '/s2').replace('_s1', '_s2'))
                
                assert s1_file.exists(), f'Image not found: {s1_file}'
                assert s2_file.exists(), f'Image not found: {s2_file}'
                
                image_pairs.append((s1_file, s2_file))
                classes.append(class_name.name)
                
        return image_pairs, classes
    
        
    def __len__(self):
        
        return len(self.image_pairs)
    
    
    def __getitem__(self, index):
        
        s1_path = self.image_pairs[index][0]
        s2_path = self.image_pairs[index][1]
        # class_name = self.classes[index]
        
        s1_image = Image.open(s1_path).convert('RGB')
        s2_image = Image.open(s2_path).convert('RGB')
                
        s1_image = self.image_transform(s1_image)
        s2_image = self.image_transform(s2_image)
        
        # return s1_image, s2_image, class_name
        
        return s1_image, s2_image
    
    
if __name__ == '__main__':
    
    dataset = SEN12Dataset('../../dataset/sen12_categorized', 'train', json_path=None)
    print(dataset[0])