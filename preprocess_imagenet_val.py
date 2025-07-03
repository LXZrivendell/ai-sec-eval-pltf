#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageNet验证集预处理脚本
将ILSVRC2012_img_val.tar解压并转换为适合平台使用的.npz格式
"""

import os
import tarfile
import numpy as np
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

def extract_tar_file(tar_path, extract_dir):
    """
    解压tar文件
    """
    print(f"🔄 正在解压 {tar_path} 到 {extract_dir}...")
    
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_dir)
    
    print(f"✅ 解压完成！")
    return extract_dir

def load_imagenet_labels():
    """
    加载ImageNet类别标签映射
    这里使用简化的标签映射，您可以根据需要调整
    """
    # 简化版本：使用文件名中的数字作为标签
    # 实际使用时可以加载完整的ImageNet标签映射
    return {}

def preprocess_images(image_dir, output_path, max_images=None, target_size=(224, 224)):
    """
    预处理图像并保存为npz格式
    
    Args:
        image_dir: 图像目录路径
        output_path: 输出npz文件路径
        max_images: 最大处理图像数量（None表示处理所有）
        target_size: 目标图像尺寸
    """
    print(f"🖼️  开始预处理图像...")
    
    # 获取所有JPEG文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.JPEG', '*.JPG']:
        image_files.extend(Path(image_dir).glob(ext))
    
    print(f"📊 找到 {len(image_files)} 张图像")
    
    if max_images:
        image_files = image_files[:max_images]
        print(f"🎯 将处理前 {max_images} 张图像")
    
    images = []
    labels = []
    valid_files = []
    
    # 预处理参数
    mean = np.array([0.485, 0.456, 0.406])  # ImageNet标准化参数
    std = np.array([0.229, 0.224, 0.225])
    
    print(f"🔄 开始处理图像...")
    
    for i, img_path in enumerate(tqdm(image_files, desc="处理图像")):
        try:
            # 加载图像
            img = Image.open(img_path).convert('RGB')
            
            # 调整大小
            img = img.resize(target_size, Image.LANCZOS)
            
            # 转换为numpy数组并标准化
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # 标准化（使用ImageNet的均值和标准差）
            img_array = (img_array - mean) / std
            
            # 转换为CHW格式（通道在前）
            img_array = np.transpose(img_array, (2, 0, 1))
            
            images.append(img_array)
            
            # 从文件名提取标签（简化版本）
            # ImageNet验证集文件名格式通常是 ILSVRC2012_val_00000001.JPEG
            filename = img_path.stem
            if 'val_' in filename:
                # 提取序号作为临时标签
                label = int(filename.split('_')[-1]) % 1000  # 限制在0-999范围内
            else:
                label = i % 1000  # 备用方案
            
            labels.append(label)
            valid_files.append(str(img_path))
            
        except Exception as e:
            print(f"⚠️  处理图像 {img_path} 时出错: {e}")
            continue
    
    # 转换为numpy数组
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"📊 成功处理 {len(images)} 张图像")
    print(f"📏 图像数组形状: {images.shape}")
    print(f"📏 标签数组形状: {labels.shape}")
    
    # 保存为npz格式
    print(f"💾 保存到 {output_path}...")
    
    # 创建元数据
    metadata = {
        'dataset_name': 'ImageNet_Validation_Subset',
        'num_samples': len(images),
        'image_shape': list(images.shape[1:]),
        'num_classes': len(np.unique(labels)),
        'preprocessing': {
            'resize': target_size,
            'normalization': 'ImageNet_standard',
            'mean': mean.tolist(),
            'std': std.tolist(),
            'format': 'CHW'
        },
        'created_at': datetime.now().isoformat(),
        'source_files': valid_files[:100]  # 只保存前100个文件名作为示例
    }
    
    np.savez_compressed(
        output_path,
        images=images,
        labels=labels,
        metadata=json.dumps(metadata, indent=2)
    )
    
    print(f"✅ 数据集已保存到 {output_path}")
    print(f"📊 文件大小: {Path(output_path).stat().st_size / (1024*1024):.2f} MB")
    
    return images.shape, labels.shape, len(np.unique(labels))

def main():
    parser = argparse.ArgumentParser(description='ImageNet验证集预处理')
    parser.add_argument('--tar_file', default='ILSVRC2012_img_val.tar', help='tar文件路径')
    parser.add_argument('--extract_dir', default='imagenet_val_extracted', help='解压目录')
    parser.add_argument('--output', default='imagenet_val_dataset.npz', help='输出npz文件路径')
    parser.add_argument('--max_images', type=int, default=5000, help='最大处理图像数量')
    parser.add_argument('--image_size', type=int, default=224, help='目标图像尺寸')
    
    args = parser.parse_args()
    
    print("🚀 ImageNet验证集预处理开始")
    print(f"📁 输入文件: {args.tar_file}")
    print(f"📂 解压目录: {args.extract_dir}")
    print(f"💾 输出文件: {args.output}")
    print(f"🎯 最大图像数: {args.max_images}")
    print(f"📏 图像尺寸: {args.image_size}x{args.image_size}")
    print("-" * 50)
    
    try:
        # 第一步：解压tar文件
        if not os.path.exists(args.extract_dir):
            extract_tar_file(args.tar_file, args.extract_dir)
        else:
            print(f"📂 目录 {args.extract_dir} 已存在，跳过解压")
        
        # 第二步：预处理图像
        img_shape, label_shape, num_classes = preprocess_images(
            args.extract_dir,
            args.output,
            max_images=args.max_images,
            target_size=(args.image_size, args.image_size)
        )
        
        print("\n🎉 预处理完成！")
        print(f"📊 图像形状: {img_shape}")
        print(f"🏷️  标签形状: {label_shape}")
        print(f"🎯 类别数量: {num_classes}")
        print(f"\n📋 下一步：在平台中上传 {args.output} 文件")
        
    except Exception as e:
        print(f"❌ 预处理失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())