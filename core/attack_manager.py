import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
from art.attacks.evasion import (
    FastGradientMethod, ProjectedGradientDescent, 
    CarliniL2Method, DeepFool, BoundaryAttack,
    HopSkipJump, SquareAttack, AutoAttack
)
from art.attacks.poisoning import (
    PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor
)
from art.estimators.classification import (
    PyTorchClassifier, TensorFlowV2Classifier, KerasClassifier
)
import numpy as np

class AttackManager:
    """攻击配置管理器"""
    
    def __init__(self):
        self.attacks_dir = "data/attacks"
        self.configs_dir = "data/attack_configs"
        self._ensure_directories()
        
        # 支持的攻击算法配置
        self.attack_algorithms = {
            "FGSM": {
                "name": "Fast Gradient Sign Method",
                "class": FastGradientMethod,
                "type": "evasion",
                "description": "快速梯度符号方法，通过在梯度方向添加扰动生成对抗样本",
                "params": {
                    "eps": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "description": "扰动幅度"},
                    "norm": {"type": "select", "options": ["inf", 1, 2], "default": "inf", "description": "范数类型"},
                    "eps_step": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0, "description": "步长"},
                    "targeted": {"type": "bool", "default": False, "description": "是否为目标攻击"}
                }
            },
            "PGD": {
                "name": "Projected Gradient Descent",
                "class": ProjectedGradientDescent,
                "type": "evasion",
                "description": "投影梯度下降攻击，FGSM的迭代版本",
                "params": {
                    "eps": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "description": "扰动幅度"},
                    "eps_step": {"type": "float", "default": 0.01, "min": 0.0, "max": 1.0, "description": "每步扰动大小"},
                    "max_iter": {"type": "int", "default": 100, "min": 1, "max": 1000, "description": "最大迭代次数"},
                    "targeted": {"type": "bool", "default": False, "description": "是否为目标攻击"},
                    "num_random_init": {"type": "int", "default": 0, "min": 0, "max": 10, "description": "随机初始化次数"}
                }
            },
            "C&W": {
                "name": "Carlini & Wagner L2",
                "class": CarliniL2Method,
                "type": "evasion",
                "description": "C&W L2攻击，优化L2距离的对抗攻击",
                "params": {
                    "confidence": {"type": "float", "default": 0.0, "min": 0.0, "max": 100.0, "description": "置信度参数"},
                    "targeted": {"type": "bool", "default": False, "description": "是否为目标攻击"},
                    "learning_rate": {"type": "float", "default": 0.01, "min": 0.001, "max": 1.0, "description": "学习率"},
                    "max_iter": {"type": "int", "default": 1000, "min": 1, "max": 10000, "description": "最大迭代次数"},
                    "initial_const": {"type": "float", "default": 0.01, "min": 0.001, "max": 100.0, "description": "初始常数"}
                }
            },
            "DeepFool": {
                "name": "DeepFool",
                "class": DeepFool,
                "type": "evasion",
                "description": "DeepFool攻击，寻找最小扰动使样本跨越决策边界",
                "params": {
                    "max_iter": {"type": "int", "default": 100, "min": 1, "max": 1000, "description": "最大迭代次数"},
                    "epsilon": {"type": "float", "default": 1e-6, "min": 1e-10, "max": 1e-2, "description": "数值稳定性参数"},
                    "nb_grads": {"type": "int", "default": 10, "min": 1, "max": 100, "description": "梯度计算批次大小"}
                }
            },
            "Boundary": {
                "name": "Boundary Attack",
                "class": BoundaryAttack,
                "type": "evasion",
                "description": "边界攻击，基于决策边界的黑盒攻击",
                "params": {
                    "targeted": {"type": "bool", "default": False, "description": "是否为目标攻击"},
                    "delta": {"type": "float", "default": 0.01, "min": 0.001, "max": 0.1, "description": "步长参数"},
                    "epsilon": {"type": "float", "default": 0.01, "min": 0.001, "max": 0.1, "description": "扰动幅度"},
                    "max_iter": {"type": "int", "default": 5000, "min": 100, "max": 50000, "description": "最大迭代次数"}
                }
            },
            "HopSkipJump": {
                "name": "HopSkipJump",
                "class": HopSkipJump,
                "type": "evasion",
                "description": "HopSkipJump攻击，改进的边界攻击算法",
                "params": {
                    "targeted": {"type": "bool", "default": False, "description": "是否为目标攻击"},
                    "norm": {"type": "select", "options": [2, "inf"], "default": 2, "description": "范数类型"},
                    "max_iter": {"type": "int", "default": 50, "min": 1, "max": 1000, "description": "最大迭代次数"},
                    "max_eval": {"type": "int", "default": 10000, "min": 100, "max": 100000, "description": "最大评估次数"}
                }
            },
            "Square": {
                "name": "Square Attack",
                "class": SquareAttack,
                "type": "evasion",
                "description": "Square攻击，基于随机搜索的黑盒攻击",
                "params": {
                    "norm": {"type": "select", "options": ["inf", 2], "default": "inf", "description": "范数类型"},
                    "max_iter": {"type": "int", "default": 5000, "min": 100, "max": 50000, "description": "最大迭代次数"},
                    "eps": {"type": "float", "default": 0.05, "min": 0.001, "max": 1.0, "description": "扰动幅度"},
                    "p_init": {"type": "float", "default": 0.8, "min": 0.1, "max": 1.0, "description": "初始扰动比例"}
                }
            },
            "AutoAttack": {
                "name": "AutoAttack",
                "class": AutoAttack,
                "type": "evasion",
                "description": "AutoAttack，组合多种攻击算法的强力攻击",
                "params": {
                    "norm": {"type": "select", "options": ["inf", 2], "default": "inf", "description": "范数类型"},
                    "eps": {"type": "float", "default": 0.3, "min": 0.001, "max": 1.0, "description": "扰动幅度"},
                    "version": {"type": "select", "options": ["standard", "plus", "rand"], "default": "standard", "description": "攻击版本"}
                }
            }
        }
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        os.makedirs(self.attacks_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)
    
    def get_attack_algorithms(self) -> Dict[str, Dict]:
        """获取所有支持的攻击算法"""
        return self.attack_algorithms
    
    def get_attack_by_type(self, attack_type: str) -> Dict[str, Dict]:
        """根据类型获取攻击算法"""
        return {k: v for k, v in self.attack_algorithms.items() 
                if v["type"] == attack_type}
    
    def save_attack_config(self, config_name: str, config_data: Dict, 
                          user_id: str) -> bool:
        """保存攻击配置"""
        try:
            config_info = {
                "name": config_name,
                "config": config_data,
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            config_file = os.path.join(self.configs_dir, f"{config_name}_{user_id}.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_info, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            st.error(f"保存攻击配置失败: {str(e)}")
            return False
    
    def load_attack_config(self, config_name: str, user_id: str) -> Optional[Dict]:
        """加载攻击配置"""
        try:
            config_file = os.path.join(self.configs_dir, f"{config_name}_{user_id}.json")
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            st.error(f"加载攻击配置失败: {str(e)}")
            return None
    
    def get_all_configs(self) -> List[Dict]:
        """获取所有攻击配置（管理员用）"""
        configs = []
        try:
            for filename in os.listdir(self.configs_dir):
                if filename.endswith('.json'):
                    config_file = os.path.join(self.configs_dir, filename)
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        configs.append(config)
        except Exception as e:
            st.error(f"获取所有配置失败: {str(e)}")
        
        return sorted(configs, key=lambda x: x.get('updated_at', ''), reverse=True)
    
    def delete_attack_config(self, config_name: str, user_id: str) -> bool:
        """删除攻击配置"""
        try:
            config_file = os.path.join(self.configs_dir, f"{config_name}_{user_id}.json")
            if os.path.exists(config_file):
                os.remove(config_file)
                return True
            return False
        except Exception as e:
            st.error(f"删除攻击配置失败: {str(e)}")
            return False
    
    def create_attack_instance(self, algorithm: str, params: Dict, 
                             estimator) -> Optional[Any]:
        """创建攻击实例"""
        try:
            if algorithm not in self.attack_algorithms:
                raise ValueError(f"不支持的攻击算法: {algorithm}")
            
            attack_class = self.attack_algorithms[algorithm]["class"]
            
            # 处理特殊参数
            processed_params = self._process_attack_params(algorithm, params)
            
            # 创建攻击实例
            attack_instance = attack_class(estimator=estimator, **processed_params)
            
            return attack_instance
        except Exception as e:
            st.error(f"创建攻击实例失败: {str(e)}")
            return None
    
    def _process_attack_params(self, algorithm: str, params: Dict) -> Dict:
        """处理攻击参数"""
        processed = {}
        algorithm_config = self.attack_algorithms[algorithm]
        
        for param_name, param_value in params.items():
            if param_name in algorithm_config["params"]:
                param_config = algorithm_config["params"][param_name]
                
                # 类型转换
                if param_config["type"] == "float":
                    processed[param_name] = float(param_value)
                elif param_config["type"] == "int":
                    processed[param_name] = int(param_value)
                elif param_config["type"] == "bool":
                    processed[param_name] = bool(param_value)
                elif param_config["type"] == "select":
                    # 处理特殊值
                    if param_value == "inf":
                        processed[param_name] = np.inf
                    else:
                        processed[param_name] = param_value
                else:
                    processed[param_name] = param_value
        
        return processed
    
    def validate_attack_params(self, algorithm: str, params: Dict) -> Tuple[bool, str]:
        """验证攻击参数"""
        try:
            if algorithm not in self.attack_algorithms:
                return False, f"不支持的攻击算法: {algorithm}"
            
            algorithm_config = self.attack_algorithms[algorithm]
            
            for param_name, param_value in params.items():
                if param_name not in algorithm_config["params"]:
                    return False, f"未知参数: {param_name}"
                
                param_config = algorithm_config["params"][param_name]
                
                # 验证参数类型和范围
                if param_config["type"] == "float":
                    try:
                        value = float(param_value)
                        if "min" in param_config and value < param_config["min"]:
                            return False, f"参数 {param_name} 小于最小值 {param_config['min']}"
                        if "max" in param_config and value > param_config["max"]:
                            return False, f"参数 {param_name} 大于最大值 {param_config['max']}"
                    except ValueError:
                        return False, f"参数 {param_name} 必须是数字"
                
                elif param_config["type"] == "int":
                    try:
                        value = int(param_value)
                        if "min" in param_config and value < param_config["min"]:
                            return False, f"参数 {param_name} 小于最小值 {param_config['min']}"
                        if "max" in param_config and value > param_config["max"]:
                            return False, f"参数 {param_name} 大于最大值 {param_config['max']}"
                    except ValueError:
                        return False, f"参数 {param_name} 必须是整数"
                
                elif param_config["type"] == "select":
                    if param_value not in param_config["options"]:
                        return False, f"参数 {param_name} 必须是 {param_config['options']} 中的一个"
            
            return True, "参数验证通过"
        
        except Exception as e:
            return False, f"参数验证失败: {str(e)}"
    
    def get_attack_info(self, algorithm: str) -> Optional[Dict]:
        """获取攻击算法信息"""
        return self.attack_algorithms.get(algorithm)
    
    def get_storage_stats(self) -> Dict:
        """获取存储统计信息"""
        stats = {
            "total_configs": 0,
            "total_size": 0,
            "by_user": {}
        }
        
        try:
            for filename in os.listdir(self.configs_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.configs_dir, filename)
                    file_size = os.path.getsize(file_path)
                    
                    stats["total_configs"] += 1
                    stats["total_size"] += file_size
                    
                    # 提取用户ID
                    user_id = filename.split('_')[-1].replace('.json', '')
                    if user_id not in stats["by_user"]:
                        stats["by_user"][user_id] = {"count": 0, "size": 0}
                    
                    stats["by_user"][user_id]["count"] += 1
                    stats["by_user"][user_id]["size"] += file_size
        
        except Exception as e:
            st.error(f"获取存储统计失败: {str(e)}")
        
        return stats
    
    def get_user_configs(self, user_id: str) -> List[Dict]:
        """获取指定用户的攻击配置"""
        configs = []
        try:
            for filename in os.listdir(self.configs_dir):
                if filename.endswith('.json') and filename.endswith(f'_{user_id}.json'):
                    config_file = os.path.join(self.configs_dir, filename)
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        # 确保配置属于指定用户
                        if config.get('user_id') == user_id:
                            configs.append(config)
        except Exception as e:
            st.error(f"获取用户配置失败: {str(e)}")
        
        return sorted(configs, key=lambda x: x.get('updated_at', ''), reverse=True)