#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证生成的npz数据集
"""

import numpy as np
import json
import argparse

def verify_dataset(npz_path):
    """
    验证npz数据集的格式和内容
    """
    print(f"🔍 验证数据集: {npz_path}")
    
    try:
        # 加载数据
        data = np.load(npz_path)
        
        print(f"📋 数据集包含的键: {list(data.keys())}")
        
        if 'images' in data:
            images = data['images']
            print(f"🖼️  图像数组形状: {images.shape}")
            print(f"📊 图像数据类型: {images.dtype}")
            print(f"📈 图像数值范围: [{images.min():.3f}, {images.max():.3f}]")
        
        if 'labels' in data:
            labels = data['labels']
            print(f"🏷️  标签数组形状: {labels.shape}")
            print(f"📊 标签数据类型: {labels.dtype}")
            print(f"📈 标签数值范围: [{labels.min()}, {labels.max()}]")
            print(f"🎯 唯一标签数量: {len(np.unique(labels))}")
        
        if 'metadata' in data:
            metadata = json.loads(data['metadata'].item())
            print(f"📋 元数据:")
            for key, value in metadata.items():
                if key != 'source_files':  # 跳过文件列表
                    print(f"   {key}: {value}")
        
        print("✅ 数据集验证通过！")
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='验证npz数据集')
    parser.add_argument('npz_file', help='npz文件路径')
    
    args = parser.parse_args()
    
    verify_dataset(args.npz_file)

if __name__ == '__main__':
    main()