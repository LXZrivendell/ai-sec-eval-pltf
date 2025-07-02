import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import hashlib
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# 尝试导入深度学习库
try:
    import torch
    import torchvision
    from torchvision import datasets, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class DatasetManager:
    """数据集管理器"""
    
    def __init__(self):
        self.datasets_dir = Path("data/datasets")
        self.datasets_info_file = Path("data/datasets_info.json")
        self.builtin_datasets_dir = self.datasets_dir / "builtin"
        self.user_datasets_dir = self.datasets_dir / "user"
        
        self.supported_formats = {
            '.csv': 'CSV表格数据',
            '.json': 'JSON数据',
            '.npy': 'NumPy数组',
            '.npz': 'NumPy压缩数组',
            '.pkl': 'Pickle数据',
            '.pickle': 'Pickle数据',
            '.parquet': 'Parquet数据',
            '.jpg': '图像数据',
            '.jpeg': '图像数据',
            '.png': '图像数据',
            '.bmp': '图像数据',
            '.txt': '文本数据'
        }
        
        self.builtin_datasets = {
            'CIFAR-10': {
                'name': 'CIFAR-10',
                'description': '10类自然图像数据集，包含60000张32x32彩色图像',
                'type': 'image',
                'classes': 10,
                'samples': 60000,
                'input_shape': (32, 32, 3),
                'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            },
            'MNIST': {
                'name': 'MNIST',
                'description': '手写数字识别数据集，包含70000张28x28灰度图像',
                'type': 'image',
                'classes': 10,
                'samples': 70000,
                'input_shape': (28, 28, 1),
                'class_names': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            },
            'Fashion-MNIST': {
                'name': 'Fashion-MNIST',
                'description': '时尚物品图像数据集，包含70000张28x28灰度图像',
                'type': 'image',
                'classes': 10,
                'samples': 70000,
                'input_shape': (28, 28, 1),
                'class_names': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            }
        }
        
        self.ensure_directories()
    
    def ensure_directories(self):
        """确保必要的目录存在"""
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.builtin_datasets_dir.mkdir(parents=True, exist_ok=True)
        self.user_datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据集信息文件
        if not self.datasets_info_file.exists():
            with open(self.datasets_info_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
    
    def get_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def detect_dataset_type(self, file_path: Path) -> str:
        """检测数据集类型"""
        file_ext = file_path.suffix.lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            return 'image'
        elif file_ext in ['.csv', '.json', '.parquet']:
            return 'tabular'
        elif file_ext in ['.npy', '.npz', '.pkl', '.pickle']:
            return 'array'
        elif file_ext == '.txt':
            return 'text'
        else:
            return 'unknown'
    
    def validate_dataset_file(self, file_path: Path, dataset_type: str) -> Tuple[bool, str, Dict[str, Any]]:
        """验证数据集文件"""
        try:
            metadata = {}
            
            if dataset_type == 'tabular':
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path)
                    metadata = {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'column_names': df.columns.tolist(),
                        'dtypes': df.dtypes.to_dict()
                    }
                elif file_path.suffix.lower() == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    metadata = {
                        'type': type(data).__name__,
                        'size': len(data) if isinstance(data, (list, dict)) else 1
                    }
                elif file_path.suffix.lower() == '.parquet':
                    df = pd.read_parquet(file_path)
                    metadata = {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'column_names': df.columns.tolist()
                    }
            
            elif dataset_type == 'array':
                if file_path.suffix.lower() in ['.npy']:
                    data = np.load(file_path)
                    metadata = {
                        'shape': data.shape,
                        'dtype': str(data.dtype),
                        'size': data.size
                    }
                elif file_path.suffix.lower() == '.npz':
                    data = np.load(file_path)
                    metadata = {
                        'files': list(data.files),
                        'arrays_info': {name: {'shape': data[name].shape, 'dtype': str(data[name].dtype)} 
                                      for name in data.files}
                    }
                elif file_path.suffix.lower() in ['.pkl', '.pickle']:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    metadata = {
                        'type': type(data).__name__,
                        'shape': getattr(data, 'shape', None),
                        'size': getattr(data, 'size', len(data) if hasattr(data, '__len__') else None)
                    }
            
            elif dataset_type == 'image':
                try:
                    img = Image.open(file_path)
                    metadata = {
                        'width': img.width,
                        'height': img.height,
                        'mode': img.mode,
                        'format': img.format
                    }
                except Exception as e:
                    return False, f"图像文件验证失败: {str(e)}", {}
            
            elif dataset_type == 'text':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                metadata = {
                    'length': len(content),
                    'lines': len(content.split('\n')),
                    'encoding': 'utf-8'
                }
            
            return True, "数据集验证成功", metadata
            
        except Exception as e:
            return False, f"数据集验证失败: {str(e)}", {}
    
    def save_uploaded_dataset(self, uploaded_file, dataset_name: str, description: str, 
                            dataset_type: str, username: str) -> Tuple[bool, str, Optional[str]]:
        """保存上传的数据集文件"""
        try:
            # 生成唯一的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = Path(uploaded_file.name).suffix
            unique_filename = f"{username}_{dataset_name}_{timestamp}{file_extension}"
            file_path = self.user_datasets_dir / unique_filename
            
            # 保存文件
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 自动检测数据集类型（如果未指定）
            if dataset_type == 'auto':
                dataset_type = self.detect_dataset_type(file_path)
            
            # 验证数据集
            is_valid, validation_msg, metadata = self.validate_dataset_file(file_path, dataset_type)
            if not is_valid:
                # 删除无效文件
                file_path.unlink()
                return False, validation_msg, None
            
            # 计算文件哈希
            file_hash = self.get_file_hash(file_path)
            
            # 获取文件信息
            file_size = file_path.stat().st_size
            
            # 保存数据集信息
            dataset_info = {
                "dataset_name": dataset_name,
                "description": description,
                "dataset_type": dataset_type,
                "file_name": unique_filename,
                "file_path": str(file_path),
                "file_size": file_size,
                "file_hash": file_hash,
                "uploaded_by": username,
                "upload_time": datetime.now().isoformat(),
                "validation_status": "valid",
                "validation_message": validation_msg,
                "metadata": metadata,
                "is_builtin": False
            }
            
            # 更新数据集信息文件
            datasets_info = self.load_datasets_info()
            dataset_id = f"{username}_{dataset_name}_{timestamp}"
            datasets_info[dataset_id] = dataset_info
            self.save_datasets_info(datasets_info)
            
            return True, f"数据集上传成功: {validation_msg}", dataset_id
            
        except Exception as e:
            return False, f"数据集保存失败: {str(e)}", None
    
    def load_datasets_info(self) -> Dict[str, Any]:
        """加载数据集信息"""
        try:
            with open(self.datasets_info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def save_datasets_info(self, datasets_info: Dict[str, Any]):
        """保存数据集信息"""
        with open(self.datasets_info_file, 'w', encoding='utf-8') as f:
            json.dump(datasets_info, f, ensure_ascii=False, indent=2)
    
    def get_builtin_datasets(self) -> Dict[str, Any]:
        """获取内置数据集字典"""
        builtin_dict = {}
        for name, info in self.builtin_datasets.items():
            dataset_data = {
                'id': f"builtin_{name}",
                'name': info['name'],
                'type': info['type'],
                'data_type': info['type'],
                'description': info['description'],
                'classes': info.get('classes', 0),
                'samples': info.get('samples', 0),
                'input_shape': info.get('input_shape', []),
                'class_names': info.get('class_names', []),
                'is_builtin': True,
                'uploaded_by': 'system',
                'upload_time': '2024-01-01T00:00:00'
            }
            builtin_dict[name] = dataset_data
        return builtin_dict
    
    def get_user_datasets(self, username: str) -> list:
        """获取用户的数据集列表"""
        datasets_info = self.load_datasets_info()
        user_datasets = []
        
        for dataset_id, info in datasets_info.items():
            if info.get('uploaded_by') == username and not info.get('is_builtin', False):
                dataset_data = {
                    'id': dataset_id,
                    'name': info.get('dataset_name', 'Unknown'),
                    'data_type': info.get('dataset_type', 'Unknown'),
                    'type': info.get('dataset_type', 'Unknown'),
                    'description': info.get('description', ''),
                    'file_size': info.get('file_size', 0),
                    'upload_time': info.get('upload_time', ''),
                    'uploaded_by': info.get('uploaded_by', ''),
                    'file_path': info.get('file_path', ''),
                    'shape': info.get('metadata', {}).get('shape', 'N/A')
                }
                user_datasets.append(dataset_data)
        
        return user_datasets
    
    def get_all_datasets(self) -> list:
        """获取所有数据集列表（包括内置和用户上传）"""
        all_datasets = []
        
        # 添加内置数据集
        all_datasets.extend(self.get_builtin_datasets())
        
        # 添加所有用户数据集
        datasets_info = self.load_datasets_info()
        for dataset_id, info in datasets_info.items():
            if not info.get('is_builtin', False):
                dataset_data = {
                    'id': dataset_id,
                    'name': info.get('dataset_name', 'Unknown'),
                    'data_type': info.get('dataset_type', 'Unknown'),
                    'type': info.get('dataset_type', 'Unknown'),
                    'description': info.get('description', ''),
                    'file_size': info.get('file_size', 0),
                    'upload_time': info.get('upload_time', ''),
                    'uploaded_by': info.get('uploaded_by', ''),
                    'file_path': info.get('file_path', ''),
                    'shape': info.get('metadata', {}).get('shape', 'N/A')
                }
                all_datasets.append(dataset_data)
        
        return all_datasets
    
    def delete_dataset(self, dataset_id: str, username: str, is_admin: bool = False) -> Tuple[bool, str]:
        """删除数据集"""
        try:
            # 不能删除内置数据集
            if dataset_id.startswith('builtin_'):
                return False, "不能删除内置数据集"
            
            datasets_info = self.load_datasets_info()
            
            if dataset_id not in datasets_info:
                return False, "数据集不存在"
            
            dataset_info = datasets_info[dataset_id]
            
            # 权限检查
            if not is_admin and dataset_info.get('uploaded_by') != username:
                return False, "没有权限删除此数据集"
            
            # 删除文件
            file_path = Path(dataset_info['file_path'])
            if file_path.exists():
                file_path.unlink()
            
            # 删除信息记录
            del datasets_info[dataset_id]
            self.save_datasets_info(datasets_info)
            
            return True, "数据集删除成功"
            
        except Exception as e:
            return False, f"删除数据集失败: {str(e)}"
    
    def load_builtin_dataset(self, dataset_name: str) -> Tuple[bool, Any, Any, str]:
        """加载内置数据集"""
        try:
            if not TORCH_AVAILABLE and not TF_AVAILABLE:
                return False, None, None, "需要安装PyTorch或TensorFlow来加载内置数据集"
            
            if dataset_name == 'CIFAR-10':
                if TORCH_AVAILABLE:
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
                    
                    # 下载并加载CIFAR-10
                    trainset = datasets.CIFAR10(root=str(self.builtin_datasets_dir), train=True,
                                               download=True, transform=transform)
                    testset = datasets.CIFAR10(root=str(self.builtin_datasets_dir), train=False,
                                              download=True, transform=transform)
                    
                    return True, trainset, testset, "CIFAR-10数据集加载成功"
                
            elif dataset_name == 'MNIST':
                if TORCH_AVAILABLE:
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])
                    
                    # 下载并加载MNIST
                    trainset = datasets.MNIST(root=str(self.builtin_datasets_dir), train=True,
                                             download=True, transform=transform)
                    testset = datasets.MNIST(root=str(self.builtin_datasets_dir), train=False,
                                            download=True, transform=transform)
                    
                    return True, trainset, testset, "MNIST数据集加载成功"
                
            elif dataset_name == 'Fashion-MNIST':
                if TORCH_AVAILABLE:
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ])
                    
                    # 下载并加载Fashion-MNIST
                    trainset = datasets.FashionMNIST(root=str(self.builtin_datasets_dir), train=True,
                                                    download=True, transform=transform)
                    testset = datasets.FashionMNIST(root=str(self.builtin_datasets_dir), train=False,
                                                   download=True, transform=transform)
                    
                    return True, trainset, testset, "Fashion-MNIST数据集加载成功"
            
            return False, None, None, f"不支持的内置数据集: {dataset_name}"
            
        except Exception as e:
            return False, None, None, f"加载内置数据集失败: {str(e)}"
    
    def load_user_dataset(self, dataset_id: str) -> Tuple[bool, Any, str]:
        """加载用户数据集"""
        try:
            datasets_info = self.load_datasets_info()
            
            if dataset_id not in datasets_info:
                return False, None, "数据集不存在"
            
            dataset_info = datasets_info[dataset_id]
            file_path = Path(dataset_info['file_path'])
            
            if not file_path.exists():
                return False, None, "数据集文件不存在"
            
            dataset_type = dataset_info['dataset_type']
            
            # 根据类型加载数据集
            if dataset_type == 'tabular':
                if file_path.suffix.lower() == '.csv':
                    data = pd.read_csv(file_path)
                elif file_path.suffix.lower() == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                elif file_path.suffix.lower() == '.parquet':
                    data = pd.read_parquet(file_path)
                else:
                    return False, None, f"不支持的表格数据格式: {file_path.suffix}"
            
            elif dataset_type == 'array':
                if file_path.suffix.lower() == '.npy':
                    data = np.load(file_path)
                elif file_path.suffix.lower() == '.npz':
                    data = np.load(file_path)
                elif file_path.suffix.lower() in ['.pkl', '.pickle']:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                else:
                    return False, None, f"不支持的数组数据格式: {file_path.suffix}"
            
            elif dataset_type == 'image':
                data = Image.open(file_path)
            
            elif dataset_type == 'text':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = f.read()
            
            else:
                return False, None, f"不支持的数据集类型: {dataset_type}"
            
            return True, data, "数据集加载成功"
            
        except Exception as e:
            return False, None, f"数据集加载失败: {str(e)}"
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """获取数据集详细信息"""
        if dataset_id.startswith('builtin_'):
            dataset_name = dataset_id.replace('builtin_', '')
            if dataset_name in self.builtin_datasets:
                info = self.builtin_datasets[dataset_name].copy()
                info['dataset_id'] = dataset_id
                info['is_builtin'] = True
                return info
        else:
            datasets_info = self.load_datasets_info()
            return datasets_info.get(dataset_id)
        
        return None
    
    def update_dataset_info(self, dataset_id: str, updates: Dict[str, Any], 
                          username: str, is_admin: bool = False) -> Tuple[bool, str]:
        """更新数据集信息"""
        try:
            # 不能修改内置数据集
            if dataset_id.startswith('builtin_'):
                return False, "不能修改内置数据集信息"
            
            datasets_info = self.load_datasets_info()
            
            if dataset_id not in datasets_info:
                return False, "数据集不存在"
            
            dataset_info = datasets_info[dataset_id]
            
            # 权限检查
            if not is_admin and dataset_info.get('uploaded_by') != username:
                return False, "没有权限修改此数据集信息"
            
            # 更新允许的字段
            allowed_fields = ['dataset_name', 'description']
            for field in allowed_fields:
                if field in updates:
                    datasets_info[dataset_id][field] = updates[field]
            
            # 更新修改时间
            datasets_info[dataset_id]['last_modified'] = datetime.now().isoformat()
            datasets_info[dataset_id]['modified_by'] = username
            
            self.save_datasets_info(datasets_info)
            
            return True, "数据集信息更新成功"
            
        except Exception as e:
            return False, f"更新数据集信息失败: {str(e)}"
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        datasets_info = self.load_datasets_info()
        builtin_datasets = self.get_builtin_datasets()
        
        total_user_datasets = len(datasets_info)
        total_builtin_datasets = len(builtin_datasets)
        total_size = sum(info.get('file_size', 0) for info in datasets_info.values())
        
        # 按类型统计
        type_stats = {}
        for info in datasets_info.values():
            dataset_type = info.get('dataset_type', 'Unknown')
            if dataset_type not in type_stats:
                type_stats[dataset_type] = {'count': 0, 'size': 0}
            type_stats[dataset_type]['count'] += 1
            type_stats[dataset_type]['size'] += info.get('file_size', 0)
        
        # 按用户统计
        user_stats = {}
        for info in datasets_info.values():
            username = info.get('uploaded_by', 'Unknown')
            if username not in user_stats:
                user_stats[username] = {'count': 0, 'size': 0}
            user_stats[username]['count'] += 1
            user_stats[username]['size'] += info.get('file_size', 0)
        
        return {
            'total_user_datasets': total_user_datasets,
            'total_builtin_datasets': total_builtin_datasets,
            'total_datasets': total_user_datasets + total_builtin_datasets,
            'total_size': total_size,
            'type_stats': type_stats,
            'user_stats': user_stats
        }
    
    def preview_dataset(self, dataset_id: str, max_samples: int = 10) -> Tuple[bool, Any, str]:
        """预览数据集"""
        try:
            if dataset_id.startswith('builtin_'):
                dataset_name = dataset_id.replace('builtin_', '')
                success, trainset, testset, message = self.load_builtin_dataset(dataset_name)
                if success:
                    # 返回少量样本用于预览
                    preview_data = {
                        'type': 'builtin',
                        'dataset_name': dataset_name,
                        'train_samples': min(len(trainset), max_samples),
                        'test_samples': min(len(testset), max_samples),
                        'info': self.builtin_datasets[dataset_name]
                    }
                    return True, preview_data, message
                else:
                    return False, None, message
            else:
                success, data, message = self.load_user_dataset(dataset_id)
                if success:
                    dataset_info = self.get_dataset_info(dataset_id)
                    dataset_type = dataset_info.get('dataset_type', 'unknown')
                    
                    preview_data = {
                        'type': 'user',
                        'dataset_type': dataset_type,
                        'data': data,
                        'info': dataset_info
                    }
                    
                    # 根据数据类型处理预览
                    if dataset_type == 'tabular' and hasattr(data, 'head'):
                        preview_data['preview'] = data.head(max_samples)
                    elif dataset_type == 'array' and hasattr(data, 'shape'):
                        preview_data['shape'] = data.shape
                        preview_data['dtype'] = str(data.dtype)
                    
                    return True, preview_data, message
                else:
                    return False, None, message
                    
        except Exception as e:
            return False, None, f"预览数据集失败: {str(e)}"