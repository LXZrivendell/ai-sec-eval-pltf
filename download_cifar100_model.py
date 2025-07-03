import torch
import torchvision.models as models
import torch.nn as nn
import os
from pathlib import Path

def download_cifar100_resnet18(save_path="./data/models/pretrained/cifar100_resnet18.pth", pretrained=True):
    """
    下载并修改ResNet-18模型以适配CIFAR-100数据集
    
    Args:
        save_path (str): 保存路径
        pretrained (bool): 是否使用预训练权重
    
    Returns:
        bool: 下载是否成功
    """
    try:
        print("正在下载并修改ResNet-18模型以适配CIFAR-100...")
        
        # 下载预训练的ResNet-18模型
        model = models.resnet18(pretrained=pretrained)
        
        # 修改最后一层以适配CIFAR-100的100个类别
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 100)  # CIFAR-100有100个类别
        
        # 由于CIFAR-100图像尺寸为32x32，修改第一层卷积
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # 移除maxpool以适应小尺寸图像
        
        # 确保保存目录存在
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 🔧 修改：保存完整模型对象而不是字典
        torch.save(model, save_path)
        
        # 同时保存详细信息到单独文件
        info_path = save_path.replace('.pth', '_info.json')
        import json
        model_info = {
            'model_architecture': 'resnet18_cifar100',
            'num_classes': 100,
            'input_size': [3, 32, 32],
            'pretrained': pretrained,
            'parameter_count': sum(p.numel() for p in model.parameters())
        }
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ CIFAR-100 ResNet-18模型已保存到: {save_path}")
        print(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"📋 模型信息已保存到: {info_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        return False

def load_cifar100_resnet18(model_path="./data/models/pretrained/cifar100_resnet18.pth"):
    """
    加载CIFAR-100 ResNet-18模型
    
    Args:
        model_path (str): 模型文件路径
    
    Returns:
        torch.nn.Module: 加载的模型
    """
    try:
        # 🔧 修改：直接加载完整模型，添加weights_only=False参数
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        model.eval()
        print(f"✅ 成功加载CIFAR-100模型: {model_path}")
        
        return model
        
    except Exception as e:
        print(f"❌ 加载模型失败: {str(e)}")
        return None

def main():
    """
    主函数 - 下载CIFAR-100适配模型
    """
    print("=== CIFAR-100 ResNet-18 模型下载器 ===")
    
    # 下载模型
    success = download_cifar100_resnet18()
    
    # 测试加载
    if success:
        print("\n测试加载模型...")
        loaded_model = load_cifar100_resnet18()
        if loaded_model:
            print(f"模型类型: {type(loaded_model)}")
            print(f"输出类别数: {loaded_model.fc.out_features}")
            
            # 测试前向传播
            test_input = torch.randn(1, 3, 32, 32)
            with torch.no_grad():
                output = loaded_model(test_input)
                print(f"输出形状: {output.shape}")
                print("✅ 模型测试通过")
    
    print("\n🎯 模型已准备就绪，可用于CIFAR-100安全评估")

if __name__ == "__main__":
    main()