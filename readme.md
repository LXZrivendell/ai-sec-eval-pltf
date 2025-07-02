## 📊 推荐的测试参数
针对ResNet-18模型和5000张ImageNet图片，推荐以下参数配置：

### FGSM攻击
- eps : 0.01-0.03 (预期攻击成功率30-60%)
- norm : "inf"
### PGD攻击
- eps : 0.01-0.03
- eps_step : 0.003-0.007
- max_iter : 10-20
### DeepFool攻击
- max_iter : 20-50
- nb_grads : 3-5
### 评估参数
- sample_size : 1000-2000 (从5000张中采样)
- batch_size : 32-64
这些修复应该能解决攻击成功率始终100%的问题，并提供更准确和合理的评估结果。