import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import os
from datetime import datetime

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 数据预处理
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# 加载CIFAR-100数据集
print('正在下载和加载CIFAR-100数据集...')
train_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=train_transform
)
test_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

print(f'训练集大小: {len(train_dataset)}')
print(f'测试集大小: {len(test_dataset)}')

# 使用torchvision的标准ResNet18（修改最后一层适配CIFAR-100）
def create_resnet18_cifar100():
    """创建适配CIFAR-100的ResNet18模型"""
    model = models.resnet18(pretrained=False)
    
    # 修改第一层卷积以适配CIFAR-100的32x32输入
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # 移除maxpool层
    
    # 修改最后一层以适配100个类别
    model.fc = nn.Linear(model.fc.in_features, 100)
    
    return model

# 创建模型
model = create_resnet18_cifar100().to(device)
print(f'模型参数数量: {sum(p.numel() for p in model.parameters()):,}')

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 训练函数
def train_epoch(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.3f}, Acc: {100.*correct/total:.2f}%')
    
    return train_loss/len(train_loader), 100.*correct/total

# 测试函数
def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100.*correct/total
    print(f'测试集准确率: {acc:.2f}%')
    return test_loss/len(test_loader), acc

# 训练模型
print('开始训练...')
num_epochs = 200
best_acc = 0

# 创建保存目录
os.makedirs('data/models', exist_ok=True)

for epoch in range(num_epochs):
    print(f'\n=== Epoch {epoch+1}/{num_epochs} ===')
    train_loss, train_acc = train_epoch(epoch)
    test_loss, test_acc = test()
    scheduler.step()
    
    print(f'训练损失: {train_loss:.3f}, 训练准确率: {train_acc:.2f}%')
    print(f'测试损失: {test_loss:.3f}, 测试准确率: {test_acc:.2f}%')
    print(f'学习率: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'data/models/resnet18_cifar100_best_{timestamp}.pth'
        
        # 保存完整模型（使用标准架构，无自定义类）
        torch.save({
            'model': model,  # 标准ResNet18模型（无自定义类）
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
            'model_info': {
                'name': 'resnet18-cifar100',
                'framework': 'PyTorch',
                'model_type': 'PyTorch',
                'dataset': 'CIFAR-100',
                'num_classes': 100,
                'input_shape': [3, 32, 32],
                'architecture': 'torchvision.models.resnet18'
            }
        }, model_path)
        print(f'保存最佳模型: {model_path} (准确率: {best_acc:.2f}%)')
    
    # 每10个epoch保存一次检查点
    if (epoch + 1) % 10 == 0:
        checkpoint_path = f'data/models/resnet18_cifar100_epoch_{epoch+1}.pth'
        torch.save({
            'model': model,  # 标准ResNet18模型（无自定义类）
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'test_acc': test_acc,
        }, checkpoint_path)
        print(f'保存检查点: {checkpoint_path}')

print(f'\n训练完成！最佳测试准确率: {best_acc:.2f}%')

# 最终测试
print('\n最终测试结果:')
final_loss, final_acc = test()
print(f'最终测试准确率: {final_acc:.2f}%')

# 保存最终模型
final_model_path = f'data/models/resnet18_cifar100_final.pth'
torch.save({
    'model': model,  # 标准ResNet18模型（无自定义类）
    'model_state_dict': model.state_dict(),
    'model_info': {
        'name': 'resnet18-cifar100',
        'framework': 'PyTorch',
        'model_type': 'PyTorch',
        'dataset': 'CIFAR-100',
        'num_classes': 100,
        'input_shape': [3, 32, 32],
        'final_accuracy': final_acc,
        'architecture': 'torchvision.models.resnet18'
    }
}, final_model_path)
print(f'保存最终模型: {final_model_path}')