import os
from pathlib import Path

# 基础配置
class Config:
    # 项目根目录
    BASE_DIR = Path(__file__).parent
    
    # 数据目录
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = DATA_DIR / "models"
    DATASETS_DIR = DATA_DIR / "datasets"
    RESULTS_DIR = DATA_DIR / "results"
    REPORTS_DIR = DATA_DIR / "reports"
    
    # 静态资源目录
    STATIC_DIR = BASE_DIR / "static"
    CSS_DIR = STATIC_DIR / "css"
    IMAGES_DIR = STATIC_DIR / "images"
    JS_DIR = STATIC_DIR / "js"
    
    # 支持的模型格式
    SUPPORTED_MODEL_FORMATS = [
        '.pth', '.pt',  # PyTorch
        '.h5', '.keras',  # Keras/TensorFlow
        '.pb',  # TensorFlow
        '.onnx',  # ONNX
        '.pkl', '.pickle'  # Scikit-learn
    ]
    
    # 支持的数据集格式
    SUPPORTED_DATASET_FORMATS = [
        '.csv', '.json', '.npy', '.npz',
        '.jpg', '.jpeg', '.png', '.bmp',
        '.txt', '.parquet'
    ]
    
    # 内置数据集
    BUILTIN_DATASETS = {
        'CIFAR-10': {
            'name': 'CIFAR-10',
            'description': '10类自然图像数据集',
            'type': 'image',
            'classes': 10,
            'samples': 60000
        },
        'MNIST': {
            'name': 'MNIST',
            'description': '手写数字识别数据集',
            'type': 'image',
            'classes': 10,
            'samples': 70000
        },
        'ImageNet-Sample': {
            'name': 'ImageNet-Sample',
            'description': 'ImageNet样本数据集',
            'type': 'image',
            'classes': 1000,
            'samples': 1000
        }
    }
    
    # 攻击算法配置
    ATTACK_ALGORITHMS = {
        'FGSM': {
            'name': 'Fast Gradient Sign Method',
            'description': '快速梯度符号攻击',
            'params': {
                'eps': {'type': 'float', 'default': 0.3, 'min': 0.0, 'max': 1.0},
                'norm': {'type': 'select', 'options': ['inf', '1', '2'], 'default': 'inf'}
            }
        },
        'PGD': {
            'name': 'Projected Gradient Descent',
            'description': '投影梯度下降攻击',
            'params': {
                'eps': {'type': 'float', 'default': 0.3, 'min': 0.0, 'max': 1.0},
                'eps_step': {'type': 'float', 'default': 0.1, 'min': 0.0, 'max': 1.0},
                'max_iter': {'type': 'int', 'default': 10, 'min': 1, 'max': 100}
            }
        },
        'C&W': {
            'name': 'Carlini & Wagner',
            'description': 'C&W攻击算法',
            'params': {
                'confidence': {'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 10.0},
                'learning_rate': {'type': 'float', 'default': 0.01, 'min': 0.001, 'max': 0.1},
                'max_iter': {'type': 'int', 'default': 1000, 'min': 100, 'max': 5000}
            }
        },
        'DeepFool': {
            'name': 'DeepFool',
            'description': 'DeepFool攻击算法',
            'params': {
                'max_iter': {'type': 'int', 'default': 50, 'min': 10, 'max': 200},
                'epsilon': {'type': 'float', 'default': 1e-6, 'min': 1e-8, 'max': 1e-4}
            }
        }
    }
    
    # 评估指标
    EVALUATION_METRICS = [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'auc_roc',
        'robustness_score',
        'attack_success_rate',
        'perturbation_magnitude'
    ]
    
    # 用户认证配置
    AUTH_CONFIG = {
        'cookie_name': 'ai_sec_eval_auth',
        'cookie_key': 'ai_sec_eval_platform_2024',
        'cookie_expiry_days': 30
    }
    
    # 文件上传限制
    MAX_FILE_SIZE_MB = 500
    MAX_BATCH_SIZE = 1000
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 开发环境配置
class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

# 生产环境配置
class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = 'WARNING'

# 根据环境变量选择配置
config = DevelopmentConfig() if os.getenv('ENVIRONMENT') == 'development' else ProductionConfig()