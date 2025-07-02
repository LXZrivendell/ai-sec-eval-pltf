import os
import json
import shutil
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from urllib.parse import urlparse
import hashlib
import zipfile
import tarfile
import pickle
import joblib

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    from transformers.utils import cached_file
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import tensorflow_hub as hub
    import tensorflow as tf
    TF_HUB_AVAILABLE = True
except ImportError:
    TF_HUB_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class ModelDownloader:
    """官方模型下载器 - 支持多种格式保存"""
    
    def __init__(self):
        self.pretrained_dir = Path("data/models/pretrained")
        self.models_info_file = Path("data/models_info.json")
        self.download_cache = Path("data/models/.cache")
        
        # 支持的文件格式映射
        self.format_extensions = {
            'pytorch': ['.pth', '.pt'],
            'tensorflow': ['.pb', '.h5'],
            'keras': ['.h5', '.keras'],
            'onnx': ['.onnx'],
            'scikit-learn': ['.pkl', '.pickle', '.joblib'],
            'transformers': ['.bin', '.safetensors']  # Hugging Face 格式
        }
        
        # 支持的官方源
        self.supported_sources = {
            'huggingface': 'Hugging Face Hub',
            'tensorflow_hub': 'TensorFlow Hub',
            'pytorch_hub': 'PyTorch Hub',
            'custom_url': 'Custom URL'
        }
        
        self.ensure_directories()
    
    def ensure_directories(self):
        """确保必要的目录存在"""
        self.pretrained_dir.mkdir(parents=True, exist_ok=True)
        self.download_cache.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型信息文件
        if not self.models_info_file.exists():
            with open(self.models_info_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
    
    def _convert_and_save_model(self, model, model_dir: Path, model_type: str, 
                               model_name: str, **kwargs) -> List[str]:
        """将模型转换并保存为指定格式"""
        saved_files = []
        
        try:
            if model_type.lower() == 'pytorch' and TORCH_AVAILABLE:
                # 保存为 .pth 格式
                model_path = model_dir / f"{model_name}.pth"
                torch.save(model.state_dict() if hasattr(model, 'state_dict') else model, model_path)
                saved_files.append(str(model_path))
                
                # 如果是完整模型，也保存为 .pt 格式
                if hasattr(model, 'state_dict'):
                    full_model_path = model_dir / f"{model_name}_full.pt"
                    torch.save(model, full_model_path)
                    saved_files.append(str(full_model_path))
            
            elif model_type.lower() == 'tensorflow' and TF_HUB_AVAILABLE:
                # 保存为 SavedModel 格式 (.pb)
                saved_model_dir = model_dir / "saved_model"
                tf.saved_model.save(model, str(saved_model_dir))
                saved_files.append(str(saved_model_dir))
                
                # 如果是 Keras 模型，也保存为 .h5 格式
                if hasattr(model, 'save'):
                    h5_path = model_dir / f"{model_name}.h5"
                    model.save(str(h5_path))
                    saved_files.append(str(h5_path))
            
            elif model_type.lower() == 'keras':
                # 保存为 .h5 和 .keras 格式
                h5_path = model_dir / f"{model_name}.h5"
                keras_path = model_dir / f"{model_name}.keras"
                
                if hasattr(model, 'save'):
                    model.save(str(h5_path))
                    saved_files.append(str(h5_path))
                    
                    # 尝试保存为新的 .keras 格式
                    try:
                        model.save(str(keras_path))
                        saved_files.append(str(keras_path))
                    except Exception:
                        pass
            
            elif model_type.lower() == 'scikit-learn':
                # 保存为多种 scikit-learn 支持的格式
                pkl_path = model_dir / f"{model_name}.pkl"
                joblib_path = model_dir / f"{model_name}.joblib"
                
                # 使用 pickle 保存
                with open(pkl_path, 'wb') as f:
                    pickle.dump(model, f)
                saved_files.append(str(pkl_path))
                
                # 使用 joblib 保存（更适合大型数组）
                try:
                    joblib.dump(model, joblib_path)
                    saved_files.append(str(joblib_path))
                except Exception:
                    pass
            
            else:
                # 默认使用 pickle 保存
                default_path = model_dir / f"{model_name}.pkl"
                with open(default_path, 'wb') as f:
                    pickle.dump(model, f)
                saved_files.append(str(default_path))
        
        except Exception as e:
            print(f"模型格式转换警告: {e}")
            # 如果转换失败，尝试默认保存
            try:
                default_path = model_dir / f"{model_name}.pkl"
                with open(default_path, 'wb') as f:
                    pickle.dump(model, f)
                saved_files.append(str(default_path))
            except Exception:
                pass
        
        return saved_files
    
    def download_huggingface_model(self, model_name: str, model_alias: str = None, 
                                 include_tokenizer: bool = True, 
                                 save_format: str = 'transformers') -> Tuple[bool, str, Optional[str]]:
        """从 Hugging Face Hub 下载完整模型"""
        if not HF_AVAILABLE:
            return False, "transformers 库未安装，无法下载 Hugging Face 模型", None
        
        try:
            model_alias = model_alias or model_name.replace('/', '_')
            model_dir = self.pretrained_dir / f"huggingface_{model_alias}"
            
            # 创建模型目录
            model_dir.mkdir(exist_ok=True)
            
            print(f"正在下载模型: {model_name}...")
            
            # 下载模型配置
            print("下载配置文件...")
            config = AutoConfig.from_pretrained(model_name)
            config.save_pretrained(model_dir)
            
            # 下载模型权重
            print("下载模型权重...")
            model = AutoModel.from_pretrained(model_name)
            
            # 保存为 Transformers 原生格式
            model.save_pretrained(model_dir)
            
            saved_files = []
            
            # 根据指定格式额外保存
            if save_format.lower() == 'pytorch' and TORCH_AVAILABLE:
                # 转换为 PyTorch 格式
                additional_files = self._convert_and_save_model(
                    model, model_dir, 'pytorch', model_alias
                )
                saved_files.extend(additional_files)
            
            # 下载 tokenizer（如果需要）
            if include_tokenizer:
                try:
                    print("下载 tokenizer...")
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    tokenizer.save_pretrained(model_dir)
                except Exception as e:
                    print(f"警告: tokenizer 下载失败: {e}")
            
            # 计算目录大小
            total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
            
            # 保存模型信息
            model_info = {
                "model_name": model_alias,
                "original_name": model_name,
                "description": f"从 Hugging Face Hub 下载的 {model_name} 模型",
                "source": "huggingface",
                "model_type": "Transformers",
                "model_dir": str(model_dir),
                "total_size": total_size,
                "include_tokenizer": include_tokenizer,
                "save_format": save_format,
                "saved_files": saved_files,
                "download_time": datetime.now().isoformat(),
                "config_info": config.to_dict() if hasattr(config, 'to_dict') else str(config)
            }
            
            # 更新模型信息文件
            self._save_model_info(f"hf_{model_alias}", model_info)
            
            return True, f"模型 {model_name} 下载成功，保存到 {model_dir}", f"hf_{model_alias}"
            
        except Exception as e:
            return False, f"下载 Hugging Face 模型失败: {str(e)}", None
    
    def download_tensorflow_hub_model(self, model_url: str, model_alias: str, 
                                     save_format: str = 'tensorflow') -> Tuple[bool, str, Optional[str]]:
        """从 TensorFlow Hub 下载模型"""
        if not TF_HUB_AVAILABLE:
            return False, "tensorflow_hub 库未安装，无法下载 TensorFlow Hub 模型", None
        
        try:
            model_dir = self.pretrained_dir / f"tfhub_{model_alias}"
            model_dir.mkdir(exist_ok=True)
            
            print(f"正在从 TensorFlow Hub 下载模型: {model_url}...")
            
            # 下载模型
            model = hub.load(model_url)
            
            # 转换并保存为指定格式
            saved_files = self._convert_and_save_model(
                model, model_dir, save_format, model_alias
            )
            
            # 计算目录大小
            total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
            
            # 保存模型信息
            model_info = {
                "model_name": model_alias,
                "original_url": model_url,
                "description": f"从 TensorFlow Hub 下载的模型",
                "source": "tensorflow_hub",
                "model_type": "TensorFlow",
                "model_dir": str(model_dir),
                "total_size": total_size,
                "save_format": save_format,
                "saved_files": saved_files,
                "download_time": datetime.now().isoformat()
            }
            
            self._save_model_info(f"tfhub_{model_alias}", model_info)
            
            return True, f"TensorFlow Hub 模型下载成功，保存到 {model_dir}", f"tfhub_{model_alias}"
            
        except Exception as e:
            return False, f"下载 TensorFlow Hub 模型失败: {str(e)}", None
    
    def download_pytorch_hub_model(self, repo: str, model_name: str, model_alias: str = None, 
                                  save_format: str = 'pytorch', **kwargs) -> Tuple[bool, str, Optional[str]]:
        """从 PyTorch Hub 下载模型"""
        if not TORCH_AVAILABLE:
            return False, "torch 库未安装，无法下载 PyTorch Hub 模型", None
        
        try:
            import torch
            print(f"使用 PyTorch 版本: {torch.__version__}")
            
            model_alias = model_alias or f"{repo}_{model_name}".replace('/', '_').replace(':', '_')
            model_dir = self.pretrained_dir / f"pytorch_{model_alias}"
            model_dir.mkdir(exist_ok=True)
            
            print(f"正在从 PyTorch Hub 下载模型: {repo}:{model_name}...")
            print(f"参数: {kwargs}")
            print(f"保存目录: {model_dir}")
            
            # 下载模型
            model = torch.hub.load(repo, model_name, **kwargs)
            print(f"模型下载成功，类型: {type(model)}")
            
            # 转换并保存为指定格式
            saved_files = self._convert_and_save_model(
                model, model_dir, save_format, model_alias
            )
            print(f"模型保存完成，文件: {saved_files}")
            
            # 保存模型信息到 JSON
            info_path = model_dir / "model_info.json"
            model_info_dict = {
                "repo": repo,
                "model_name": model_name,
                "kwargs": kwargs,
                "download_time": datetime.now().isoformat()
            }
            
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info_dict, f, ensure_ascii=False, indent=2)
            
            # 计算目录大小
            total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
            
            # 保存模型信息
            model_info = {
                "model_name": model_alias,
                "repo": repo,
                "original_model_name": model_name,
                "description": f"从 PyTorch Hub 下载的 {repo}:{model_name} 模型",
                "source": "pytorch_hub",
                "model_type": "PyTorch",
                "model_dir": str(model_dir),
                "total_size": total_size,
                "save_format": save_format,
                "saved_files": saved_files,
                "download_time": datetime.now().isoformat(),
                "kwargs": kwargs
            }
            
            self._save_model_info(f"pytorch_{model_alias}", model_info)
            
            return True, f"PyTorch Hub 模型下载成功，保存到 {model_dir}", f"pytorch_{model_alias}"
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"PyTorch Hub 下载失败: {error_details}")
            return False, f"下载 PyTorch Hub 模型失败: {str(e)}", None
    
    def download_from_url(self, url: str, model_alias: str, model_type: str = 'custom',
                         extract: bool = True) -> Tuple[bool, str, Optional[str]]:
        """从自定义 URL 下载模型"""
        try:
            model_dir = self.pretrained_dir / f"custom_{model_alias}"
            model_dir.mkdir(exist_ok=True)
            
            print(f"正在从 URL 下载模型: {url}...")
            
            # 下载文件
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # 获取文件名
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name or "model_file"
            
            # 根据文件扩展名确定模型类型
            file_ext = Path(filename).suffix.lower()
            detected_type = self._detect_model_type_from_extension(file_ext)
            if detected_type:
                model_type = detected_type
            
            # 确保文件有正确的扩展名
            if not self._has_valid_extension(filename, model_type):
                # 添加适当的扩展名
                valid_ext = self._get_default_extension(model_type)
                filename = f"{Path(filename).stem}{valid_ext}"
            
            file_path = model_dir / filename
            
            # 保存文件
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            saved_files = [str(file_path)]
            
            # 如果是压缩文件且需要解压
            if extract and filename.endswith(('.zip', '.tar.gz', '.tar', '.tgz')):
                print("正在解压文件...")
                if filename.endswith('.zip'):
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(model_dir)
                elif filename.endswith(('.tar.gz', '.tar', '.tgz')):
                    with tarfile.open(file_path, 'r:*') as tar_ref:
                        tar_ref.extractall(model_dir)
                
                # 删除压缩文件
                file_path.unlink()
                saved_files = [str(f) for f in model_dir.rglob('*') if f.is_file()]
            
            # 计算目录大小
            total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
            
            # 保存模型信息
            model_info = {
                "model_name": model_alias,
                "original_url": url,
                "description": f"从自定义 URL 下载的模型",
                "source": "custom_url",
                "model_type": model_type,
                "model_dir": str(model_dir),
                "total_size": total_size,
                "extracted": extract,
                "saved_files": saved_files,
                "download_time": datetime.now().isoformat()
            }
            
            self._save_model_info(f"custom_{model_alias}", model_info)
            
            return True, f"自定义 URL 模型下载成功，保存到 {model_dir}", f"custom_{model_alias}"
            
        except Exception as e:
            return False, f"从 URL 下载模型失败: {str(e)}", None
    
    def _detect_model_type_from_extension(self, extension: str) -> Optional[str]:
        """根据文件扩展名检测模型类型"""
        for model_type, extensions in self.format_extensions.items():
            if extension in extensions:
                return model_type
        return None
    
    def _has_valid_extension(self, filename: str, model_type: str) -> bool:
        """检查文件名是否有有效的扩展名"""
        file_ext = Path(filename).suffix.lower()
        valid_extensions = self.format_extensions.get(model_type, [])
        return file_ext in valid_extensions
    
    def _get_default_extension(self, model_type: str) -> str:
        """获取模型类型的默认扩展名"""
        extensions = self.format_extensions.get(model_type, ['.pkl'])
        return extensions[0]
    
    def _save_model_info(self, model_id: str, model_info: Dict[str, Any]):
        """保存模型信息到文件"""
        try:
            # 加载现有信息
            with open(self.models_info_file, 'r', encoding='utf-8') as f:
                all_models_info = json.load(f)
            
            # 添加新模型信息
            all_models_info[model_id] = model_info
            
            # 保存回文件
            with open(self.models_info_file, 'w', encoding='utf-8') as f:
                json.dump(all_models_info, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"保存模型信息失败: {e}")
    
    def list_downloaded_models(self) -> Dict[str, Any]:
        """列出已下载的预训练模型"""
        try:
            with open(self.models_info_file, 'r', encoding='utf-8') as f:
                all_models = json.load(f)
            
            # 过滤出预训练模型
            pretrained_models = {}
            for model_id, info in all_models.items():
                if info.get('source') in self.supported_sources:
                    pretrained_models[model_id] = info
            
            return pretrained_models
            
        except Exception:
            return {}
    
    def delete_downloaded_model(self, model_id: str) -> Tuple[bool, str]:
        """删除已下载的模型"""
        try:
            # 加载模型信息
            with open(self.models_info_file, 'r', encoding='utf-8') as f:
                all_models = json.load(f)
            
            if model_id not in all_models:
                return False, "模型不存在"
            
            model_info = all_models[model_id]
            
            # 删除模型目录
            model_dir = Path(model_info['model_dir'])
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # 从信息文件中删除
            del all_models[model_id]
            
            # 保存更新后的信息
            with open(self.models_info_file, 'w', encoding='utf-8') as f:
                json.dump(all_models, f, ensure_ascii=False, indent=2)
            
            return True, "模型删除成功"
            
        except Exception as e:
            return False, f"删除模型失败: {str(e)}"
    
    def get_popular_models(self) -> Dict[str, List[str]]:
        """获取热门模型列表"""
        return {
            "huggingface": [
                "bert-base-uncased",
                "bert-base-chinese",
                "gpt2",
                "distilbert-base-uncased",
                "roberta-base",
                "xlm-roberta-base",
                "t5-small",
                "facebook/bart-base"
            ],
            "tensorflow_hub": [
                "https://tfhub.dev/google/universal-sentence-encoder/4",
                "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/4",
                "https://tfhub.dev/tensorflow/resnet_50/classification/1",
                "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5"
            ],
            "pytorch_hub": [
                "pytorch/vision:v0.10.0:resnet18",
                "pytorch/vision:v0.10.0:resnet50",
                "pytorch/vision:v0.10.0:mobilenet_v2",
                "ultralytics/yolov5:v6.0:yolov5s"
            ]
        }


# 使用示例函数
def download_model_example():
    """模型下载示例"""
    downloader = ModelDownloader()
    
    print("=== 模型下载器示例 ===")
    
    # 示例1: 下载 Hugging Face 模型
    print("\n1. 下载 Hugging Face 模型...")
    success, message, model_id = downloader.download_huggingface_model(
        "distilbert-base-uncased", 
        "distilbert_base",
        include_tokenizer=True
    )
    print(f"结果: {message}")
    
    # 示例2: 下载 PyTorch Hub 模型
    print("\n2. 下载 PyTorch Hub 模型...")
    success, message, model_id = downloader.download_pytorch_hub_model(
        "pytorch/vision:v0.10.0", 
        "resnet18",
        "resnet18_pretrained",
        pretrained=True
    )
    print(f"结果: {message}")
    
    # 示例3: 列出已下载的模型
    print("\n3. 已下载的模型:")
    models = downloader.list_downloaded_models()
    for model_id, info in models.items():
        print(f"  - {model_id}: {info['model_name']} ({info['source']})")
        if 'saved_files' in info:
            print(f"    保存的文件: {info['saved_files']}")

if __name__ == "__main__":
    download_model_example()