import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from datetime import datetime

def download_cifar100_with_proper_preprocessing():
    """下载CIFAR-100数据集并应用与训练时相同的预处理"""
    
    print('正在下载CIFAR-100数据集...')
    
    # 与训练脚本完全相同的预处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # 下载测试集
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data/datasets/cifar100_normalized', 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    print(f'测试集大小: {len(test_dataset)}')
    
    # 转换为numpy数组
    x_test = []
    y_test = []
    
    print('正在转换数据格式...')
    for i, (img, label) in enumerate(test_dataset):
        x_test.append(img.numpy())
        y_test.append(label)
        
        if (i + 1) % 1000 == 0:
            print(f'已处理 {i + 1}/{len(test_dataset)} 个样本')
    
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    print(f'数据形状: x_test={x_test.shape}, y_test={y_test.shape}')
    print(f'数据范围: min={x_test.min():.3f}, max={x_test.max():.3f}')
    print(f'标签范围: min={y_test.min()}, max={y_test.max()}')
    
    # 保存预处理后的数据
    os.makedirs('data/datasets/processed', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'data/datasets/processed/cifar100_normalized_{timestamp}.npz'
    
    np.savez_compressed(
        save_path,
        images=x_test,
        labels=y_test,
        preprocessing_info={
            'mean': [0.5071, 0.4867, 0.4408],
            'std': [0.2675, 0.2565, 0.2761],
            'normalized': True,
            'format': 'CHW',
            'dtype': 'float32'
        }
    )
    
    print(f'\n已保存预处理后的CIFAR-100数据集到: {save_path}')
    
    # 同时保存原始数据（仅ToTensor，无归一化）用于对比
    raw_transform = transforms.Compose([transforms.ToTensor()])
    raw_dataset = torchvision.datasets.CIFAR100(
        root='./data/datasets/cifar100_raw', 
        train=False, 
        download=True, 
        transform=raw_transform
    )
    
    x_raw = []
    y_raw = []
    
    print('\n正在保存原始数据用于对比...')
    for i, (img, label) in enumerate(raw_dataset):
        x_raw.append(img.numpy())
        y_raw.append(label)
        
        if (i + 1) % 1000 == 0:
            print(f'已处理 {i + 1}/{len(raw_dataset)} 个样本')
    
    x_raw = np.array(x_raw)
    y_raw = np.array(y_raw)
    
    raw_save_path = f'data/datasets/processed/cifar100_raw_{timestamp}.npz'
    np.savez_compressed(
        raw_save_path,
        images=x_raw,
        labels=y_raw,
        preprocessing_info={
            'mean': None,
            'std': None,
            'normalized': False,
            'format': 'CHW',
            'dtype': 'float32'
        }
    )
    
    print(f'已保存原始CIFAR-100数据集到: {raw_save_path}')
    
    return save_path, raw_save_path

def verify_preprocessing_difference():
    """验证预处理差异"""
    print('\n=== 验证预处理差异 ===')
    
    # 加载一个样本进行对比
    raw_transform = transforms.Compose([transforms.ToTensor()])
    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    dataset_raw = torchvision.datasets.CIFAR100(
        root='./data/datasets/temp', train=False, download=True, transform=raw_transform
    )
    dataset_norm = torchvision.datasets.CIFAR100(
        root='./data/datasets/temp', train=False, download=True, transform=norm_transform
    )
    
    # 获取第一个样本
    img_raw, _ = dataset_raw[0]
    img_norm, _ = dataset_norm[0]
    
    print(f'原始数据范围: [{img_raw.min():.3f}, {img_raw.max():.3f}]')
    print(f'归一化后范围: [{img_norm.min():.3f}, {img_norm.max():.3f}]')
    print(f'平台预处理 (raw/255): [{img_raw.min():.3f}, {img_raw.max():.3f}]')
    print(f'训练预处理 (normalized): [{img_norm.min():.3f}, {img_norm.max():.3f}]')
    
if __name__ == '__main__':
    print('=== CIFAR-100数据集下载器（匹配训练预处理）===')
    
    # 验证预处理差异
    verify_preprocessing_difference()
    
    # 下载并保存数据集
    normalized_path, raw_path = download_cifar100_with_proper_preprocessing()
    
    print('\n=== 完成 ===')
    print(f'标准化数据集: {normalized_path}')
    print(f'原始数据集: {raw_path}')
    print('\n现在你可以使用标准化的数据集进行测试，应该能获得与训练时相近的准确率！')