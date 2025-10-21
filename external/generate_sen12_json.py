import json
import random
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any

# 在每个类别中按随机种子打散数据，按比例划分训练集、验证集和测试集，并生成json文件
def generate_dataset_json(root_dir: Path, output_dir: Path,
                          split_ratio: Tuple[float, float, float]=(0.8, 0.1, 0.1), 
                          seed: int=114514):

    random.seed(seed)
    class_list: List[str] = []
    image_list: List[str] = []
    
    train_dict: Dict[List[str, str]] = {}
    valid_dict: Dict[List[str, str]] = {}
    test_dict : Dict[List[str, str]] = {}
    
    train_list: List[str]= []
    valid_list = []
    test_list = []
    
    for category in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        
        category_name = category.name
        category_dir = category / 's1'      
        
        class_list.append(category_name)
        image_list = sorted(category_dir.glob('*.png'))
        
        random.shuffle(image_list)
        
        n_train, n_valid, n_test = _alloc_counts(len(image_list), split_ratio)
        
        def make_items(path: List[str]) -> List[Dict[str, str]]:
            return [{'path': [str(p.resolve()), 
                              str(p.resolve()).replace('/s1', '/s2').replace('_s1', '_s2')],
                     'category': category_name} for p in path]
        
        train_list.extend(make_items(image_list[:n_train]))
        valid_list.extend(make_items(image_list[n_train:n_train+n_valid]))
        test_list.extend(make_items(image_list[n_train+n_valid:]))
        
        
    dataset: Dict[str, Any] = {
        'meta': {
            'root': str(root),
            'split_ratio': split_ratio,
            'seed': seed,
            'num_classes': len(class_list),
            'categories': class_list,
            'generate_time': datetime.now().isoformat(timespec='seconds'),
            'count': {
                'train': len(train_list),
                'valid': len(valid_list),
                'test' : len(test_list),
                'total': len(train_list) + len(valid_list) + len(test_list)
            }
        },

        'train': train_list,
        'valid': valid_list,
        'test' : test_list,
    }
    
    with open(output_dir / 'sen12.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)
        
    print(f'Generated dataset json at {output_dir / "sen12.json"}')



def _alloc_counts(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """最大余数法，把 n 个样本按比率分到 3 份并保证总和为 n"""
    
    exact = [n * ratios[0], n * ratios[1], n * ratios[2]]
    floors = [int(x) for x in exact]
    remain = n - sum(floors)
    
    # 余数从大到小分配剩余名额
    remainders = [(i, exact[i] - floors[i]) for i in range(3)]
    remainders.sort(key=lambda x: x[1], reverse=True)
    
    for i in range(remain):
        floors[remainders[i][0]] += 1
    
    return tuple(floors)  # (train_n, val_n, test_n)


if __name__ == '__main__':
    
    root = Path('../../dataset/sen12_categorized')
    output_dir = Path('./configs')
    
    assert root.exists(), 'Dataset not found'
    assert output_dir.exists(), 'Output directory not found'
    
    generate_dataset_json(root, output_dir)