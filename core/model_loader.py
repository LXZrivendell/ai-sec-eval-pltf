import streamlit as st
import torch
import tensorflow as tf
import numpy as np
import pickle
import joblib
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import hashlib
import shutil

class ModelLoader:
    """模型加载和管理器"""
    
    def __init__(self):
        self.models_dir = Path("data/models")
        self.models_info_file = Path("data/models_info.json")
        self.supported_formats = {
            '.pth': 'PyTorch',
            '.pt': 'PyTorch', 
            '.h5': 'Keras/TensorFlow',
            '.keras': 'Keras',
            '.pb': 'TensorFlow',
            '.onnx': 'ONNX',
            '.pkl': 'Scikit-learn',
            '.pickle': 'Scikit-learn',
            '.joblib': 'Scikit-learn'
        }
        self.ensure_directories()
    
    def ensure_directories(self):
        """确保必要的目录存在"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型信息文件
        if not self.models_info_file.exists():
            with open(self.models_info_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
    
    def get_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def validate_model_file(self, file_path: Path, model_type: str) -> Tuple[bool, str]:
        """验证模型文件"""
        try:
            if model_type == 'PyTorch':
                # 验证PyTorch模型
                model = torch.load(file_path, map_location='cpu')
                if hasattr(model, 'state_dict'):
                    return True, "PyTorch模型验证成功"
                elif isinstance(model, dict):
                    return True, "PyTorch状态字典验证成功"
                else:
                    return True, "PyTorch模型文件验证成功"
            
            elif model_type in ['Keras/TensorFlow', 'Keras']:
                # 验证Keras/TensorFlow模型
                model = tf.keras.models.load_model(file_path)
                return True, "Keras/TensorFlow模型验证成功"
            
            elif model_type == 'TensorFlow':
                # 验证TensorFlow SavedModel
                if file_path.suffix == '.pb':
                    return True, "TensorFlow模型文件验证成功"
                else:
                    model = tf.saved_model.load(str(file_path))
                    return True, "TensorFlow SavedModel验证成功"
            
            elif model_type == 'Scikit-learn':
                # 验证Scikit-learn模型
                if file_path.suffix in ['.pkl', '.pickle']:
                    with open(file_path, 'rb') as f:
                        model = pickle.load(f)
                elif file_path.suffix == '.joblib':
                    model = joblib.load(file_path)
                
                # 检查是否有predict方法
                if hasattr(model, 'predict'):
                    return True, "Scikit-learn模型验证成功"
                else:
                    return False, "文件不是有效的Scikit-learn模型"
            
            elif model_type == 'ONNX':
                # ONNX模型验证（需要安装onnx库）
                try:
                    import onnx
                    model = onnx.load(str(file_path))
                    onnx.checker.check_model(model)
                    return True, "ONNX模型验证成功"
                except ImportError:
                    return True, "ONNX模型文件验证成功（未安装onnx库，跳过详细验证）"
            
            return True, "模型文件格式验证通过"
            
        except Exception as e:
            return False, f"模型验证失败: {str(e)}"
    
    def save_uploaded_model(self, uploaded_file, model_name: str, description: str, 
                          model_type: str, username: str) -> Tuple[bool, str, Optional[str]]:
        """保存上传的模型文件"""
        try:
            # 生成唯一的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = Path(uploaded_file.name).suffix
            unique_filename = f"{username}_{model_name}_{timestamp}{file_extension}"
            file_path = self.models_dir / unique_filename
            
            # 保存文件
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 验证模型
            is_valid, validation_msg = self.validate_model_file(file_path, model_type)
            if not is_valid:
                # 删除无效文件
                file_path.unlink()
                return False, validation_msg, None
            
            # 计算文件哈希
            file_hash = self.get_file_hash(file_path)
            
            # 获取文件信息
            file_size = file_path.stat().st_size
            
            # 保存模型信息
            model_info = {
                "model_name": model_name,
                "description": description,
                "model_type": model_type,
                "file_name": unique_filename,
                "file_path": str(file_path),
                "file_size": file_size,
                "file_hash": file_hash,
                "uploaded_by": username,
                "upload_time": datetime.now().isoformat(),
                "validation_status": "valid",
                "validation_message": validation_msg
            }
            
            # 更新模型信息文件
            models_info = self.load_models_info()
            model_id = f"{username}_{model_name}_{timestamp}"
            models_info[model_id] = model_info
            self.save_models_info(models_info)
            
            return True, f"模型上传成功: {validation_msg}", model_id
            
        except Exception as e:
            return False, f"模型保存失败: {str(e)}", None
    
    def load_models_info(self) -> Dict[str, Any]:
        """加载模型信息"""
        try:
            with open(self.models_info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def save_models_info(self, models_info: Dict[str, Any]):
        """保存模型信息"""
        with open(self.models_info_file, 'w', encoding='utf-8') as f:
            json.dump(models_info, f, ensure_ascii=False, indent=2)
    
    def get_user_models(self, username: str) -> Dict[str, Any]:
        """获取用户的模型列表"""
        models_info = self.load_models_info()
        user_models = {}
        
        for model_id, info in models_info.items():
            if info.get('uploaded_by') == username:
                # 保持原有的模型信息结构，添加必要的字段
                model_data = info.copy()
                model_data.update({
                    'model_name': info.get('model_name', 'Unknown'),
                    'model_type': info.get('model_type', 'Unknown'),
                    'file_size': info.get('file_size', 0),
                    'upload_time': info.get('upload_time', ''),
                    'uploaded_by': info.get('uploaded_by', ''),
                    'description': info.get('description', ''),
                    'file_path': info.get('file_path', ''),
                    'validation_status': info.get('validation_status', 'unknown'),
                    'validation_message': info.get('validation_message', '')
                })
                user_models[model_id] = model_data
        
        return user_models
    
    def get_all_models(self) -> Dict[str, Any]:
        """获取所有模型列表（管理员用）"""
        models_info = self.load_models_info()
        all_models = {}
        
        for model_id, info in models_info.items():
            # 保持原有的模型信息结构，添加必要的字段
            model_data = info.copy()
            model_data.update({
                'model_name': info.get('model_name', 'Unknown'),
                'model_type': info.get('model_type', 'Unknown'),
                'file_size': info.get('file_size', 0),
                'upload_time': info.get('upload_time', ''),
                'uploaded_by': info.get('uploaded_by', ''),
                'description': info.get('description', ''),
                'file_path': info.get('file_path', ''),
                'validation_status': info.get('validation_status', 'unknown'),
                'validation_message': info.get('validation_message', '')
            })
            all_models[model_id] = model_data
        
        return all_models
    
    def delete_model(self, model_id: str, username: str, is_admin: bool = False) -> Tuple[bool, str]:
        """删除模型"""
        try:
            models_info = self.load_models_info()
            
            if model_id not in models_info:
                return False, "模型不存在"
            
            model_info = models_info[model_id]
            
            # 权限检查
            if not is_admin and model_info.get('uploaded_by') != username:
                return False, "没有权限删除此模型"
            
            # 删除文件或目录
            # 检查是上传的模型还是下载的模型
            if 'file_path' in model_info:
                # 上传的模型，删除单个文件
                file_path = Path(model_info['file_path'])
                if file_path.exists():
                    file_path.unlink()
            elif 'model_dir' in model_info:
                # 下载的模型，删除整个目录
                model_dir = Path(model_info['model_dir'])
                if model_dir.exists():
                    shutil.rmtree(model_dir)
            else:
                return False, "模型文件路径信息缺失"
            
            # 删除信息记录
            del models_info[model_id]
            self.save_models_info(models_info)
            
            return True, "模型删除成功"
            
        except Exception as e:
            return False, f"删除模型失败: {str(e)}"
    
    def load_model(self, model_id: str) -> Tuple[bool, Any, str]:
        """加载模型"""
        try:
            models_info = self.load_models_info()
            
            if model_id not in models_info:
                return False, None, "模型不存在"
            
            model_info = models_info[model_id]
            file_path = Path(model_info['file_path'])
            
            if not file_path.exists():
                return False, None, "模型文件不存在"
            
            model_type = model_info['model_type']
            
            # 根据类型加载模型
            if model_type == 'PyTorch':
                model = torch.load(file_path, map_location='cpu')
            elif model_type in ['Keras/TensorFlow', 'Keras']:
                model = tf.keras.models.load_model(file_path)
            elif model_type == 'TensorFlow':
                model = tf.saved_model.load(str(file_path))
            elif model_type == 'Scikit-learn':
                if file_path.suffix in ['.pkl', '.pickle']:
                    with open(file_path, 'rb') as f:
                        model = pickle.load(f)
                elif file_path.suffix == '.joblib':
                    model = joblib.load(file_path)
            else:
                return False, None, f"不支持的模型类型: {model_type}"
            
            return True, model, "模型加载成功"
            
        except Exception as e:
            return False, None, f"模型加载失败: {str(e)}"
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型详细信息"""
        models_info = self.load_models_info()
        return models_info.get(model_id)
    
    def update_model_info(self, model_id: str, updates: Dict[str, Any], 
                         username: str, is_admin: bool = False) -> Tuple[bool, str]:
        """更新模型信息"""
        try:
            models_info = self.load_models_info()
            
            if model_id not in models_info:
                return False, "模型不存在"
            
            model_info = models_info[model_id]
            
            # 权限检查
            if not is_admin and model_info.get('uploaded_by') != username:
                return False, "没有权限修改此模型信息"
            
            # 更新允许的字段
            allowed_fields = ['model_name', 'description']
            for field in allowed_fields:
                if field in updates:
                    models_info[model_id][field] = updates[field]
            
            # 更新修改时间
            models_info[model_id]['last_modified'] = datetime.now().isoformat()
            models_info[model_id]['modified_by'] = username
            
            self.save_models_info(models_info)
            
            return True, "模型信息更新成功"
            
        except Exception as e:
            return False, f"更新模型信息失败: {str(e)}"
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        models_info = self.load_models_info()
        
        total_models = len(models_info)
        total_size = sum(info.get('file_size', 0) for info in models_info.values())
        
        # 按类型统计
        type_stats = {}
        for info in models_info.values():
            model_type = info.get('model_type', 'Unknown')
            if model_type not in type_stats:
                type_stats[model_type] = {'count': 0, 'size': 0}
            type_stats[model_type]['count'] += 1
            type_stats[model_type]['size'] += info.get('file_size', 0)
        
        # 按用户统计
        user_stats = {}
        for info in models_info.values():
            username = info.get('uploaded_by', 'Unknown')
            if username not in user_stats:
                user_stats[username] = {'count': 0, 'size': 0}
            user_stats[username]['count'] += 1
            user_stats[username]['size'] += info.get('file_size', 0)
        
        return {
            'total_models': total_models,
            'total_size': total_size,
            'type_stats': type_stats,
            'user_stats': user_stats
        }