# AI模型安全评估平台项目结构

```plaintext
ai-sec-eval-pltf/
├── .gitattributes                  # Git属性配置
├── .gitignore                      # Git忽略文件配置
├── app.py                          # Streamlit主应用入口
├── config.py                       # 全局配置文件
├── requirements.txt                # 项目依赖
├── readme.md                       # 项目说明文档
├── project_structure.txt           # 项目结构说明（本文件）
│
├── pages/                          # Streamlit页面模块
│   ├── __init__.py
│   ├── 1_🏠_Home.py                # 首页
│   ├── 2_🔐_Login.py               # 用户注册和登录页
│   ├── 3_📤_Model_Upload.py        # 模型上传页面
│   ├── 4_📊_Dataset_Manager.py     # 数据集管理页面
│   ├── 5_⚔️_Attack_Config.py       # 攻击配置页面
│   ├── 6_🛡️_Security_Evaluation.py # 安全评估页面
│   ├── 7_📊_Report_Manager.py      # 报告管理页面
│   ├── 8_📥_Model_Download.py      # 模型下载页面
│   └── 9_🛡️_Defense_Config.py     # 防御配置页面
│
├── core/                           # 核心功能模块
│   ├── __init__.py
│   ├── attack_manager.py           # 攻击管理器
│   ├── auth_manager.py             # 认证管理器
│   ├── dataset_manager.py          # 数据集管理器
│   ├── model_loader.py             # 模型加载器
│   ├── model_downloader.py         # 模型下载器
│   ├── security_evaluator.py       # 安全评估器 ---> evaluation/
│   │
│   ├── defense/                    # 防御模块（实现ing）
│   │   ├── __init__.py
│   │   ├── defense_manager.py      # 防御管理器
│   │   ├── defense_metrics.py      # 防御指标计算
│   │   └── purification_methods.py # 净化方法
│   │
│   ├── evaluation/                 # 评估模块
│   │   ├── __init__.py
│   │   ├── art_estimator_manager.py # ART估计器管理
│   │   ├── attack_executor.py      # 攻击执行器
│   │   ├── config.py               # 评估配置
│   │   ├── data_processor.py       # 数据处理器
│   │   ├── defense_evaluator.py    # 防御评估器
│   │   ├── memory_manager.py       # 内存管理器
│   │   ├── metrics_calculator.py   # 指标计算器
│   │   └── result_manager.py       # 结果管理器
│   │
│   ├── reporting/                  # 报告模块
│   │   ├── __init__.py
│   │   └── report_generator.py     # 报告生成器
│   │
│   └── visualization/              # 可视化模块
│       ├── __init__.py
│       └── chart_generator.py      # 图表生成器
│
├── data/                           # 数据存储目录
│   ├── attack_configs/             # 攻击配置文件
│   │   ├── 1_admin.json
│   │   ├── 2_admin.json
│   │   └── 333_admin.json
│   ├── attacks/                    # 攻击数据
│   ├── datasets_info.json          # 数据集信息
│   └── models_info.json            # 模型信息
│
├── static/                         # 静态资源
│   ├── css/                        # 样式文件
│   ├── images/                     # 图片资源
│   └── js/                         # JavaScript文件
│
├── templates/                      # 模板文件
│
├── tests/                          # 测试文件
│   ├── __init__.py
│   ├── test_core.py                # 核心模块测试
│   └── test_refactored_evaluator.py # 重构评估器测试
│
├── download_cifar100_dataset.py    # CIFAR-100数据集下载脚本
├── download_resnet18.py            # ResNet-18模型下载脚本
├── download_resnet50.py            # ResNet-50模型下载脚本
└── train_cifar100_model.py         # CIFAR-100模型训练脚本
```

# 功能模块说明：

1. 主应用 (app.py)
   - Streamlit应用入口
   - 侧边栏导航
   - 全局配置和状态管理

2. 页面模块 (pages/)
   - 模块化的页面组件，共9个功能页面
   - 涵盖完整的AI安全评估流程
   - 支持模型上传、下载、攻击配置、防御配置等

3. 核心模块 (core/)
   - attack_manager: 攻击算法管理和执行
   - auth_manager: 用户认证和权限管理
   - dataset_manager: 数据集加载和预处理
   - model_loader: 多框架模型加载支持
   - model_downloader: 预训练模型下载
   - security_evaluator: 安全性评估主控制器

4. 防御模块 (core/defense/)
   - defense_manager: 防御策略管理
   - defense_metrics: 防御效果评估指标
   - purification_methods: 输入净化方法

5. 评估模块 (core/evaluation/)
   - art_estimator_manager: ART框架估计器管理
   - attack_executor: 攻击执行引擎
   - defense_evaluator: 防御评估引擎
   - memory_manager: 内存优化管理
   - metrics_calculator: 评估指标计算
   - result_manager: 结果存储和管理

6. 报告模块 (core/reporting/)
   - report_generator: 自动化报告生成

7. 可视化模块 (core/visualization/)
   - chart_generator: 图表和可视化生成

8. 数据目录 (data/)
   - 攻击配置、数据集信息、模型信息的结构化存储
   - 支持多用户配置管理

9. 工具脚本
   - 数据集和模型的自动下载脚本
   - 模型训练脚本

# 技术栈：
- 前端: Streamlit + HTML/CSS/JS
- 后端: Python + ART (Adversarial Robustness Toolbox)
- 深度学习: PyTorch + TensorFlow
- 数据处理: NumPy + Pandas
- 可视化: Matplotlib + Plotly + Streamlit Charts
- 安全评估: ART + 自定义攻击/防御算法

# 使用说明：
1. 安装依赖: pip install -r requirements.txt
2. 下载数据集: python download_cifar100_dataset.py
3. 下载预训练模型: python download_resnet18.py 或 python download_resnet50.py
4. 运行应用: streamlit run app.py
5. 访问地址: http://localhost:8501

# 项目特色：
- 🔐 完整的用户认证系统
- 📤 支持多种模型格式上传
- 📥 预训练模型一键下载
- ⚔️ 集成多种对抗攻击算法
- 🛡️ 提供多种防御策略
- 📊 全面的安全评估指标
- 📈 丰富的可视化展示
- 📋 自动化报告生成
- 🧠 内存优化管理
- 🔄 模块化架构设计

# 开发状态：
### 该项目已实现完整的AI模型安全评估平台功能，包括：
- ✅ 用户认证和权限管理
- ✅ 模型上传和下载管理
- ✅ 数据集管理
- ✅ 攻击配置和执行
- ✅ 防御策略配置
- ✅ 安全评估和指标计算
- ✅ 结果可视化
- ✅ 报告生成和管理
- ✅ 内存优化和性能管理
