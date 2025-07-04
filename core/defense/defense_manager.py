import numpy as np
from typing import Dict, Any, Optional
from .purification_methods import PurificationMethods

class DefenseManager:
    def __init__(self):
        self.defense_methods = {
            # 输入预处理防御
            "gaussian_noise": {
                "name": "高斯噪声",
                "type": "preprocessing",
                "params": {"std": 0.1}
            },
            "jpeg_compression": {
                "name": "JPEG压缩",
                "type": "preprocessing", 
                "params": {"quality": 75}
            },
            "bit_depth_reduction": {
                "name": "位深度降低",
                "type": "preprocessing",
                "params": {"bits": 4}
            },
            
            # 对抗训练防御
            "adversarial_training": {
                "name": "对抗训练",
                "type": "training",
                "params": {"eps": 0.3, "alpha": 0.01}
            },
            
            # 检测防御
            "statistical_detection": {
                "name": "统计检测",
                "type": "detection",
                "params": {"threshold": 0.5}
            },
            
            # 净化防御
            "autoencoder_purification": {
                "name": "自编码器净化",
                "type": "purification",
                "params": {"compression_ratio": 0.1}
            }
        }
        self.purification_methods = PurificationMethods()
    
    def create_defense_instance(self, method_name: str, params: Dict, model: Any) -> Optional['DefenseInstance']:
        """创建防御实例"""
        if method_name not in self.defense_methods:
            return None
        
        method_info = self.defense_methods[method_name]
        return DefenseInstance(method_name, method_info, params, self.purification_methods)
    
    def get_available_methods(self) -> Dict:
        """获取可用的防御方法"""
        return self.defense_methods

class DefenseInstance:
    """防御实例"""
    
    def __init__(self, method_name: str, method_info: Dict, params: Dict, purification_methods: PurificationMethods):
        self.method_name = method_name
        self.method_info = method_info
        self.params = params
        self.purification_methods = purification_methods
    
    def defend(self, x_data: np.ndarray) -> np.ndarray:
        """应用防御"""
        if self.method_name == "gaussian_noise":
            return self.purification_methods.gaussian_denoising(x_data, self.params.get('std', 0.1))
        elif self.method_name == "jpeg_compression":
            return self.purification_methods.jpeg_compression_purification(x_data, self.params.get('quality', 75))
        else:
            # 默认返回原始数据
            return x_data