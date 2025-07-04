import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import streamlit as st
from art.utils import load_mnist, load_cifar10

try:
    from torchvision import datasets, transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

class DataProcessor:
    """数据处理专用类"""
    
    def __init__(self):
        self.supported_builtin_datasets = ['MNIST', 'CIFAR-10', 'CIFAR-100', 'Fashion-MNIST']
        self.supported_formats = ['numpy', 'csv']
    
    def prepare_dataset(self, dataset_info: Dict, sample_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """准备数据集"""
        try:
            dataset_type = dataset_info.get('data_type', dataset_info.get('type', 'unknown'))
            is_builtin = (dataset_type == 'builtin' or 
                         dataset_info.get('type') == 'builtin' or 
                         not dataset_info.get('file_path'))
            
            if is_builtin:
                x_data, y_data = self._load_builtin_dataset(dataset_info)
            else:
                x_data, y_data = self._load_user_dataset(dataset_info)
            
            if x_data is None:
                return None, None
            
            # 应用采样
            x_data, y_data = self._apply_sampling(x_data, y_data, sample_size)
            
            # 数据预处理 - 传递数据集类型信息
            x_data = self._preprocess_data(x_data, is_builtin=is_builtin)
            
            # 验证数据
            self._validate_data(x_data, y_data)
            
            return x_data, y_data
            
        except Exception as e:
            st.error(f"数据处理失败: {str(e)}")
            return None, None
    
    def _load_builtin_dataset(self, dataset_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """加载内置数据集"""
        dataset_name = dataset_info['name']
        
        if dataset_name == 'MNIST':
            (_, _), (x_test, y_test), _, _ = load_mnist()
            return x_test, y_test
        elif dataset_name == 'CIFAR-10':
            (_, _), (x_test, y_test), _, _ = load_cifar10()
            return x_test, y_test
        elif dataset_name == 'CIFAR-100':
            return self._load_torchvision_dataset('CIFAR100')
        elif dataset_name == 'Fashion-MNIST':
            return self._load_torchvision_dataset('FashionMNIST')
        else:
            raise ValueError(f"不支持的内置数据集: {dataset_name}")
    
    def _load_torchvision_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载torchvision数据集"""
        if not TORCHVISION_AVAILABLE:
            raise ImportError(f"需要安装torchvision来加载{dataset_name}数据集")
        
        transform = transforms.Compose([transforms.ToTensor()])
        
        if dataset_name == 'CIFAR100':
            testset = datasets.CIFAR100(root='./data/datasets/builtin', train=False,
                                       download=True, transform=transform)
        elif dataset_name == 'FashionMNIST':
            testset = datasets.FashionMNIST(root='./data/datasets/builtin', train=False,
                                           download=True, transform=transform)
        else:
            raise ValueError(f"不支持的torchvision数据集: {dataset_name}")
        
        x_data, y_data = [], []
        for i in range(len(testset)):
            img, label = testset[i]
            x_data.append(img.numpy())
            y_data.append(label)
        
        return np.array(x_data), np.array(y_data)
    
    def _load_user_dataset(self, dataset_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """加载用户数据集"""
        dataset_path = dataset_info.get('file_path')
        if not dataset_path:
            raise ValueError("用户数据集缺少文件路径")
        
        data_format = dataset_info.get('data_format', 'numpy')
        
        if data_format == 'numpy':
            return self._load_numpy_data(dataset_path)
        elif data_format == 'csv':
            return self._load_csv_data(dataset_path)
        else:
            raise ValueError(f"不支持的数据格式: {data_format}")
    
    def _load_numpy_data(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载numpy格式数据"""
        data = np.load(dataset_path, allow_pickle=True)
        
        if hasattr(data, 'files'):  # NPZ文件
            available_keys = list(data.files)
            
            # 查找图像数据
            x_data = None
            for key in ['images', 'x', 'data']:
                if key in available_keys:
                    x_data = data[key]
                    break
            if x_data is None and available_keys:
                x_data = data[available_keys[0]]
            
            # 查找标签数据
            y_data = None
            for key in ['labels', 'y', 'targets']:
                if key in available_keys:
                    y_data = data[key]
                    break
            if y_data is None and len(available_keys) > 1:
                y_data = data[available_keys[1]]
            
            data.close()
            return x_data, y_data
            
        elif isinstance(data, dict):
            x_data = data.get('x', data.get('data'))
            y_data = data.get('y', data.get('labels'))
            return x_data, y_data
        else:
            return data, None
    
    def _load_csv_data(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载CSV格式数据"""
        df = pd.read_csv(dataset_path)
        x_data = df.iloc[:, :-1].values
        y_data = df.iloc[:, -1].values
        return x_data, y_data
    
    def _apply_sampling(self, x_data: np.ndarray, y_data: np.ndarray, 
                       sample_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """应用数据采样"""
        if sample_size is None or sample_size >= len(x_data):
            st.info(f"使用全部数据: {len(x_data)} 个样本")
            return x_data, y_data
        
        indices = np.random.choice(len(x_data), sample_size, replace=False)
        x_sampled = x_data[indices]
        y_sampled = y_data[indices] if y_data is not None else None
        
        st.info(f"从 {len(x_data)} 个样本中采样了 {sample_size} 个")
        return x_sampled, y_sampled
    
    def _preprocess_data(self, x_data: np.ndarray, is_builtin: bool = True) -> np.ndarray:
        """数据预处理"""
        # 数据类型转换（所有数据都需要）
        if x_data.dtype != np.float32:
            x_data = x_data.astype(np.float32)
        
        # 对于用户上传的数据集，假设已经预处理过，跳过归一化和维度转换
        if not is_builtin:
            st.info("用户上传的数据集：假设已预处理，跳过归一化和维度转换")
            return x_data
        
        # 仅对内置数据集进行预处理
        st.info("内置数据集：应用标准预处理")
        
        # 归一化
        if x_data.max() > 1.0:
            x_data = x_data / 255.0
        
        # 维度转换：(N, H, W, C) -> (N, C, H, W)
        if len(x_data.shape) == 4 and x_data.shape[-1] in [1, 3]:
            x_data = np.transpose(x_data, (0, 3, 1, 2))
            st.info(f"数据维度已转换为 PyTorch 格式: {x_data.shape}")
        
        return x_data
    
    def _validate_data(self, x_data: np.ndarray, y_data: np.ndarray):
        """验证数据"""
        if y_data is not None:
            st.info(f"数据集信息: 图像形状={x_data.shape}, 标签形状={y_data.shape}")
            st.info(f"标签范围: min={y_data.min()}, max={y_data.max()}")
            
            # 检查标签格式
            if len(y_data.shape) > 1 and y_data.shape[1] > 1:
                st.info("检测到one-hot编码标签")
            else:
                st.info("检测到类别索引标签")
            
            # 标签分布
            unique_labels, counts = np.unique(
                y_data.flatten() if len(y_data.shape) > 1 else y_data, 
                return_counts=True
            )
            st.info(f"标签分布: {dict(zip(unique_labels, counts))}")