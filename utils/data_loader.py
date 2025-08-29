import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np

# 数据转换器，将图像调整为256x256并转换为Tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class GADFDataset(Dataset):
    """GADF图像数据集类"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []  # 物质类别标签
        self.concentrations = []  # 浓度值
        
        # 物质类别映射
        self.class_map = {'C4': 0, 'C6': 1, 'C10': 2}
        self.class_names = {0: 'C4', 1: 'C6', 2: 'C10'}
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载所有图像路径和对应的标签"""
        for substance in os.listdir(self.data_dir):
            substance_path = os.path.join(self.data_dir, substance)
            if not os.path.isdir(substance_path) or substance not in self.class_map:
                continue
            
            for conc_folder in os.listdir(substance_path):
                conc_path = os.path.join(substance_path, conc_folder)
                if not os.path.isdir(conc_path):
                    continue
                
                try:
                    concentration = float(conc_folder)
                except ValueError:
                    continue
                
                for img_file in os.listdir(conc_path):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(conc_path, img_file)
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_map[substance])
                        self.concentrations.append(concentration)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        concentration = self.concentrations[idx]
        
        # 对浓度进行归一化（0-1000nM -> 0-1）
        normalized_concentration = concentration / 1000.0
        
        return image, torch.tensor(label, dtype=torch.long), torch.tensor(normalized_concentration, dtype=torch.float32)

def get_data_loaders(data_dir, batch_size=32):
    """获取训练、验证和测试数据加载器"""
    # 创建数据集
    dataset = GADFDataset(data_dir, transform=transform)
    
    # 计算数据集分割大小
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # 固定随机种子以确保分割一致
    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, dataset.class_names