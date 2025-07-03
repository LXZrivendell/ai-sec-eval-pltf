# Core模块初始化文件
"""
AI模型安全评估平台 - 核心模块

本模块包含平台的核心功能组件：
- AuthManager: 用户认证管理
- ModelLoader: 模型加载管理
- DatasetManager: 数据集管理
- AttackManager: 攻击配置管理
- SecurityEvaluator: 安全评估器
- ReportGenerator: 报告生成器
"""

# 导入核心类，方便外部使用
from .auth_manager import AuthManager
from .model_loader import ModelLoader
from .dataset_manager import DatasetManager
from .attack_manager import AttackManager
from .security_evaluator import SecurityEvaluator
from .reporting import ReportGenerator

# 定义模块版本
__version__ = "1.0.0"

# 定义公开的API
__all__ = [
    'AuthManager',
    'ModelLoader', 
    'DatasetManager',
    'AttackManager',
    'SecurityEvaluator',
    'ReportGenerator'
]

# 模块级别的配置
DEFAULT_CONFIG = {
    'data_dir': 'data',
    'reports_dir': 'reports',
    'models_dir': 'data/models',
    'datasets_dir': 'data/datasets',
    'attack_configs_dir': 'data/attack_configs',
    'evaluation_results_dir': 'data/evaluation_results',
    'logs_dir': 'data/logs'
}

# 支持的模型框架
SUPPORTED_FRAMEWORKS = [
    'pytorch',
    'tensorflow',
    'keras',
    'sklearn'
]

# 支持的攻击算法
SUPPORTED_ATTACKS = [
    'FGSM',
    'PGD',
    'C&W',
    'DeepFool',
    'BIM',
    'JSMA'
]

# 支持的数据集格式
SUPPORTED_DATASET_FORMATS = [
    'numpy',
    'csv',
    'json',
    'pickle'
]