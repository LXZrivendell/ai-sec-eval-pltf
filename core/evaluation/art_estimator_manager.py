import torch
import tensorflow as tf
from typing import Dict, Tuple, Optional, Any
import streamlit as st
from art.estimators.classification import (
    PyTorchClassifier, TensorFlowV2Classifier, KerasClassifier
)

class ARTEstimatorManager:
    """ART估计器管理类"""
    
    def __init__(self):
        self.supported_frameworks = ['pytorch', 'tensorflow', 'keras']
    
    def create_estimator(self, model_info: Dict, input_shape: Tuple) -> Optional[Any]:
        """创建ART估计器"""
        try:
            model_type = model_info['model_type'].lower()
            
            if model_type == 'pytorch':
                return self._create_pytorch_estimator(model_info, input_shape)
            elif model_type in ['tensorflow', 'keras', 'keras/tensorflow']:
                return self._create_tensorflow_estimator(model_info, input_shape)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
                
        except Exception as e:
            st.error(f"创建ART估计器失败: {str(e)}")
            return None
    
    def _create_pytorch_estimator(self, model_info: Dict, input_shape: Tuple) -> PyTorchClassifier:
        """创建PyTorch估计器"""
        model_path = model_info['file_path']
        
        # 安全加载模型
        try:
            model_data = torch.load(model_path, map_location='cpu', weights_only=True)
        except Exception:
            model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 检查模型类型并提取模型对象
        if isinstance(model_data, dict):
            if 'model' in model_data:
                # 字典中包含完整模型对象
                model = model_data['model']
            elif 'model_state_dict' in model_data:
                # 只有权重，需要重建模型架构
                raise ValueError("检测到权重文件，但缺少模型架构。请上传完整的模型文件。")
            else:
                raise ValueError("无法识别的模型文件格式。")
        else:
            # 直接是模型对象
            model = model_data
        
        model.eval()
        
        # 创建损失函数和优化器
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # 创建ART分类器
        estimator = PyTorchClassifier(
            model=model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=input_shape,
            nb_classes=model_info.get('num_classes', 10)
        )
        
        return estimator
    
    def _create_tensorflow_estimator(self, model_info: Dict, input_shape: Tuple) -> TensorFlowV2Classifier:
        """创建TensorFlow估计器"""
        model_path = model_info['file_path']
        model = tf.keras.models.load_model(model_path)
        
        estimator = TensorFlowV2Classifier(
            model=model,
            nb_classes=model_info.get('num_classes', 10),
            input_shape=input_shape
        )
        
        return estimator
    
    def validate_estimator(self, estimator: Any, x_sample: Any) -> bool:
        """验证估计器是否正常工作"""
        try:
            # 测试预测
            test_input = x_sample[:1] if len(x_sample) > 0 else x_sample
            predictions = estimator.predict(test_input)
            return predictions is not None and len(predictions) > 0
        except Exception as e:
            st.error(f"估计器验证失败: {str(e)}")
            return False