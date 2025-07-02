import torch
import torchvision.models as models
import os
from pathlib import Path

def download_resnet18(save_path="./data/models/pretrained/resnet18.pth", pretrained=True):
    """
    下载ResNet-18模型并保存为.pth文件
    
    Args:
        save_path (str): 保存路径，默认为data/models/pretrained/resnet18.pth
        pretrained (bool): 是否下载预训练权重，默认为True
    
    Returns:
        bool: 下载是否成功
    """
    try:
        print("正在下载ResNet-18模型...")
        
        # 下载ResNet-18模型
        model = models.resnet18(pretrained=pretrained)
        
        # 确保保存目录存在
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整模型（包含架构和权重）
        torch.save(model, save_path)
        
        # 获取文件大小
        file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
        
        print(f"✅ ResNet-18模型下载成功！")
        print(f"📁 保存路径: {save_path}")
        print(f"📊 文件大小: {file_size:.2f} MB")
        print(f"🔧 预训练权重: {'是' if pretrained else '否'}")
        print(f"📦 保存类型: 完整模型（架构+权重）")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        return False

def load_resnet18(model_path="./data/models/pretrained/resnet18.pth"):
    """
    加载保存的ResNet-18完整模型
    
    Args:
        model_path (str): 模型文件路径
    
    Returns:
        torch.nn.Module: 加载的模型
    """
    try:
        # 直接加载完整模型，设置weights_only=False以兼容完整模型
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print(f"✅ 模型加载成功: {model_path}")
        return model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return None

def download_full_model(save_path="./data/models/pretrained/resnet18_full.pth", pretrained=True):
    """
    下载完整的ResNet-18模型（包含架构和权重）
    
    Args:
        save_path (str): 保存路径
        pretrained (bool): 是否使用预训练权重
    
    Returns:
        bool: 下载是否成功
    """
    try:
        print("正在下载完整ResNet-18模型...")
        
        # 下载模型
        model = models.resnet18(pretrained=pretrained)
        
        # 确保保存目录存在
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整模型
        torch.save(model, save_path)
        
        # 获取文件大小
        file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
        
        print(f"✅ 完整ResNet-18模型下载成功！")
        print(f"📁 保存路径: {save_path}")
        print(f"📊 文件大小: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        return False

def main():
    """
    主函数 - 下载ResNet18模型到指定目录
    """
    print("=== ResNet-18 模型下载器 ===")
    
    # 下载完整模型到data/models/pretrained目录
    print("\n正在下载ResNet18模型到data/models/pretrained目录...")
    success = download_resnet18("./data/models/pretrained/resnet18.pth", pretrained=True)
    
    # 测试加载
    if success:
        print("\n测试加载模型...")
        loaded_model = load_resnet18("./data/models/pretrained/resnet18.pth")
        if loaded_model:
            print(f"模型类型: {type(loaded_model)}")
            print(f"模型参数数量: {sum(p.numel() for p in loaded_model.parameters()):,}")
    
    # 显示模型信息
    if success:
        print("\n📋 模型信息:")
        print("- 模型名称: ResNet-18")
        print("- 框架: PyTorch")
        print("- 输入尺寸: (3, 224, 224)")
        print("- 分类数: 1000 (ImageNet)")
        print("- 参数量: ~11.7M")
        print("- 保存位置: data/models/pretrained/resnet18.pth")
        print("- 保存格式: 完整模型（包含架构和权重）")

if __name__ == "__main__":
    # 运行主程序
    main()