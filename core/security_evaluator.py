import os
import json
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
from art.estimators.classification import (
    PyTorchClassifier, TensorFlowV2Classifier, KerasClassifier
)
from art.utils import load_mnist, load_cifar10
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .model_loader import ModelLoader
from .dataset_manager import DatasetManager
from .attack_manager import AttackManager

class SecurityEvaluator:
    """安全评估引擎"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.dataset_manager = DatasetManager()
        self.attack_manager = AttackManager()
        self.results_dir = "data/evaluation_results"
        self.reports_dir = "data/reports"
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def create_art_estimator(self, model_info: Dict, input_shape: Tuple) -> Optional[Any]:
        """创建ART估计器"""
        try:
            model_path = model_info['file_path']
            model_type = model_info['model_type'].lower()  # 转换为小写进行比较
            
            if model_type == 'pytorch':
                # 加载PyTorch模型
                try:
                    # 首先尝试 weights_only=True（安全模式）
                    model = torch.load(model_path, map_location='cpu', weights_only=True)
                except Exception:
                    # 如果失败，尝试 weights_only=False（兼容完整模型）
                    model = torch.load(model_path, map_location='cpu', weights_only=False)
                
                # 如果加载的是 state_dict，需要先创建模型架构
                if isinstance(model, dict):
                    st.error("检测到权重文件，但缺少模型架构。请上传完整的模型文件。")
                    return None
                
                model.eval()
                
                # 创建损失函数
                criterion = torch.nn.CrossEntropyLoss()
                
                # 创建优化器（用于对抗训练，这里使用默认参数）
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                
                # 创建ART分类器
                estimator = PyTorchClassifier(
                    model=model,
                    loss=criterion,
                    optimizer=optimizer,
                    input_shape=input_shape,
                    nb_classes=model_info.get('num_classes', 10)
                )
                
            elif model_type in ['tensorflow', 'keras', 'keras/tensorflow']:
                # 加载TensorFlow/Keras模型
                model = tf.keras.models.load_model(model_path)
                
                # 创建ART分类器
                estimator = TensorFlowV2Classifier(
                    model=model,
                    nb_classes=model_info.get('num_classes', 10),
                    input_shape=input_shape
                )
                
            else:
                st.error(f"不支持的模型类型: {model_info['model_type']}")
                return None
            
            return estimator
            
        except Exception as e:
            st.error(f"创建ART估计器失败: {str(e)}")
            return None
    
    def prepare_dataset(self, dataset_info: Dict, sample_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """准备数据集"""
        try:
            # 修复键名不匹配问题：统一使用 data_type，兼容 type 字段
            dataset_type = dataset_info.get('data_type', dataset_info.get('type', 'unknown'))
            
            # 判断是否为内置数据集：检查 type 字段或者没有 file_path
            is_builtin = (dataset_type == 'builtin' or 
                         dataset_info.get('type') == 'builtin' or 
                         not dataset_info.get('file_path'))
            
            if is_builtin:
                # 内置数据集
                dataset_name = dataset_info['name']
                
                if dataset_name == 'MNIST':
                    (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
                    x_data, y_data = x_test, y_test
                elif dataset_name == 'CIFAR-10':
                    (x_train, y_train), (x_test, y_test), _, _ = load_cifar10()
                    x_data, y_data = x_test, y_test
                elif dataset_name == 'CIFAR-100':
                    (x_train, y_train), (x_test, y_test), _, _ = load_cifar100()
                    x_data, y_data = x_test, y_test
                elif dataset_name == 'Fashion-MNIST':
                    (x_train, y_train), (x_test, y_test), _, _ = load_fashion_mnist()
                    x_data, y_data = x_test, y_test
                else:
                    st.error(f"不支持的内置数据集: {dataset_name}")
                    return None, None
                    
            else:
                # 用户上传的数据集
                dataset_path = dataset_info.get('file_path')
                if not dataset_path:
                    st.error("用户数据集缺少文件路径")
                    return None, None
                    
                data_format = dataset_info.get('data_format', 'numpy')  # 添加默认值
                
                if data_format == 'numpy':
                    data = np.load(dataset_path, allow_pickle=True)
                    
                    # 正确处理 NpzFile 对象
                    if hasattr(data, 'files'):  # 检查是否为 NpzFile 对象
                        # NPZ 文件包含多个数组
                        available_keys = list(data.files)
                        
                        # 尝试找到图像数据
                        if 'images' in available_keys:
                            x_data = data['images']
                        elif 'x' in available_keys:
                            x_data = data['x']
                        elif 'data' in available_keys:
                            x_data = data['data']
                        else:
                            # 使用第一个可用的键
                            x_data = data[available_keys[0]]
                        
                        # 尝试找到标签数据
                        if 'labels' in available_keys:
                            y_data = data['labels']
                        elif 'y' in available_keys:
                            y_data = data['y']
                        elif 'targets' in available_keys:
                            y_data = data['targets']
                        else:
                            # 如果只有一个数组，假设没有标签
                            if len(available_keys) == 1:
                                y_data = None
                            else:
                                # 使用第二个可用的键作为标签
                                y_data = data[available_keys[1]] if len(available_keys) > 1 else None
                                
                        # 关闭 NpzFile 对象
                        data.close()
                        
                    elif isinstance(data, dict):
                        # 处理字典格式
                        x_data = data['x'] if 'x' in data else data['data']
                        y_data = data['y'] if 'y' in data else data['labels']
                    else:
                        # 单个numpy数组
                        x_data = data
                        y_data = None
                        
                elif data_format == 'csv':
                    df = pd.read_csv(dataset_path)
                    # 假设最后一列是标签
                    x_data = df.iloc[:, :-1].values
                    y_data = df.iloc[:, -1].values
                    
                else:
                    st.error(f"不支持的数据格式: {data_format}")
                    return None, None
            
            # 采样 - 修复索引越界问题
            if sample_size:
                actual_sample_size = min(sample_size, len(x_data))
                if actual_sample_size < len(x_data):
                    indices = np.random.choice(len(x_data), actual_sample_size, replace=False)
                    x_data = x_data[indices]
                    if y_data is not None:
                        y_data = y_data[indices]
                    st.info(f"从 {len(x_data)} 个样本中采样了 {actual_sample_size} 个")
                else:
                    st.info(f"数据集大小 ({len(x_data)}) 小于等于采样大小 ({sample_size})，使用全部数据")
            
            # 数据预处理
            if x_data.dtype != np.float32:
                x_data = x_data.astype(np.float32)
            
            # 归一化到[0,1]
            if x_data.max() > 1.0:
                x_data = x_data / 255.0
            
            # 添加维度格式转换：从 (N, H, W, C) 转换为 (N, C, H, W)
            if len(x_data.shape) == 4 and x_data.shape[-1] in [1, 3]:  # 检测是否为图像数据
                # 如果最后一个维度是通道数（1或3），则转换为 PyTorch 格式
                x_data = np.transpose(x_data, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
                st.info(f"数据维度已转换为 PyTorch 格式: {x_data.shape}")
            
            # 在返回前添加数据验证和调试信息
            if y_data is not None:
                st.info(f"数据集信息: 图像形状={x_data.shape}, 标签形状={y_data.shape}")
                st.info(f"标签范围: min={y_data.min()}, max={y_data.max()}")
                st.info(f"标签类型: {y_data.dtype}")
                
                # 检查标签是否为one-hot编码
                if len(y_data.shape) > 1 and y_data.shape[1] > 1:
                    st.info("检测到one-hot编码标签")
                else:
                    st.info("检测到类别索引标签")
                    
                # 检查标签分布
                unique_labels, counts = np.unique(y_data.flatten() if len(y_data.shape) > 1 else y_data, return_counts=True)
                st.info(f"标签分布: {dict(zip(unique_labels, counts))}")
            
            return x_data, y_data
            
        except Exception as e:
            st.error(f"准备数据集失败: {str(e)}")
            return None, None
    
    def evaluate_model_robustness(self, model_info: Dict, dataset_info: Dict, 
                        attack_config: Dict, evaluation_params: Dict) -> Dict:
        """评估模型鲁棒性"""
        try:
            # 准备数据
            x_data, y_data = self.prepare_dataset(
                dataset_info, 
                evaluation_params.get('sample_size', 1000)
            )
            
            if x_data is None or y_data is None:
                return {"error": "数据准备失败"}
            
            # 创建ART估计器
            input_shape = x_data.shape[1:]
            estimator = self.create_art_estimator(model_info, input_shape)
            
            if estimator is None:
                return {"error": "模型加载失败"}
            
            # 获取原始预测
            st.info("🔍 获取原始模型预测...")
            original_predictions = estimator.predict(x_data)
            
            # 处理标签格式：确保 y_data 和预测结果格式一致
            if len(y_data.shape) > 1 and y_data.shape[1] > 1:
                # 如果 y_data 是 one-hot 编码，转换为类别索引
                y_true = np.argmax(y_data, axis=1)
            else:
                # 如果 y_data 已经是类别索引，直接使用
                y_true = y_data.flatten() if len(y_data.shape) > 1 else y_data
            
            # 确保预测结果是类别索引
            y_pred_original = np.argmax(original_predictions, axis=1)
            
            # 找出原本被正确分类的样本
            correctly_classified_mask = (y_pred_original == y_true)
            correctly_classified_indices = np.where(correctly_classified_mask)[0]
            
            original_accuracy = accuracy_score(y_true, y_pred_original)
            
            # 如果没有正确分类的样本，无法进行攻击评估
            if len(correctly_classified_indices) == 0:
                return {"error": "模型在测试数据上准确率为0，无法进行攻击评估"}
            
            st.info(f"原始模型准确率: {original_accuracy:.3f}, 正确分类样本数: {len(correctly_classified_indices)}")
            
            # 创建攻击实例
            st.info("⚔️ 创建攻击实例...")
            attack_instance = self.attack_manager.create_attack_instance(
                attack_config['algorithm'],
                attack_config['params'],
                estimator
            )
            
            if attack_instance is None:
                return {"error": "攻击实例创建失败"}
            
            # 只对正确分类的样本进行攻击
            x_correct = x_data[correctly_classified_indices]
            y_correct = y_data[correctly_classified_indices] if y_data is not None else None
            y_true_correct = y_true[correctly_classified_indices]
            
            # 生成对抗样本
            st.info("🎯 生成对抗样本...")
            batch_size = attack_config['advanced_options'].get('batch_size', 32)
            
            adversarial_samples = []
            for i in range(0, len(x_correct), batch_size):
                batch_x = x_correct[i:i+batch_size]
                batch_y = y_correct[i:i+batch_size] if y_correct is not None else None
                
                # 生成对抗样本
                if attack_config.get('targeted', False) and batch_y is not None:
                    # 目标攻击
                    adv_batch = attack_instance.generate(x=batch_x, y=batch_y)
                else:
                    # 非目标攻击
                    adv_batch = attack_instance.generate(x=batch_x)
                
                adversarial_samples.append(adv_batch)
                
                # 更新进度
                progress = min((i + len(batch_x)) / len(x_correct), 1.0)
                st.progress(progress)
            
            adversarial_samples = np.concatenate(adversarial_samples, axis=0)
            
            # 评估对抗样本
            st.info("📊 评估对抗样本效果...")
            adversarial_predictions = estimator.predict(adversarial_samples)
            y_pred_adversarial = np.argmax(adversarial_predictions, axis=1)
            
            # 计算正确的攻击成功率
            # 攻击成功 = 原本正确分类的样本现在被误分类
            attack_successful_mask = (y_pred_adversarial != y_true_correct)
            attack_success_count = np.sum(attack_successful_mask)
            attack_success_rate = attack_success_count / len(y_true_correct)
            
            # 计算对抗样本在所有数据上的准确率
            # 为了完整性，我们需要重建完整的预测结果
            full_adversarial_predictions = np.copy(y_pred_original)
            full_adversarial_predictions[correctly_classified_indices] = y_pred_adversarial
            adversarial_accuracy = accuracy_score(y_true, full_adversarial_predictions)
            
            # 计算扰动统计
            perturbations = adversarial_samples - x_correct
            l0_norm = np.mean(np.sum(perturbations != 0, axis=tuple(range(1, len(perturbations.shape)))))
            l2_norm = np.mean(np.sqrt(np.sum(perturbations ** 2, axis=tuple(range(1, len(perturbations.shape))))))
            linf_norm = np.mean(np.max(np.abs(perturbations), axis=tuple(range(1, len(perturbations.shape)))))
            
            st.info(f"攻击结果: 成功攻击 {attack_success_count}/{len(y_true_correct)} 个样本, 攻击成功率: {attack_success_rate:.3f}")
            
            # 生成详细报告
            evaluation_result = {
                "evaluation_id": f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "model_info": model_info,
                "dataset_info": dataset_info,
                "attack_config": attack_config,
                "evaluation_params": evaluation_params,
                "results": {
                    "original_accuracy": float(original_accuracy),
                    "adversarial_accuracy": float(adversarial_accuracy),
                    "attack_success_rate": float(attack_success_rate),
                    "robustness_score": float(1.0 - attack_success_rate),
                    "perturbation_stats": {
                        "l0_norm": float(l0_norm),
                        "l2_norm": float(l2_norm),
                        "linf_norm": float(linf_norm)
                    },
                    "sample_count": len(x_data),
                    "correctly_classified_count": len(correctly_classified_indices),
                    "successful_attacks": int(attack_success_count),
                    "attack_details": {
                        "total_samples": len(x_data),
                        "correctly_classified": len(correctly_classified_indices),
                        "attacked_samples": len(y_true_correct),
                        "successful_attacks": int(attack_success_count)
                    }
                },
                "detailed_metrics": {
                    "original_classification_report": classification_report(
                        y_true, y_pred_original, output_dict=True
                    ),
                    "adversarial_classification_report": classification_report(
                        y_true, full_adversarial_predictions, output_dict=True
                    )
                }
            }
            
            # ... 保存结果的代码保持不变 ...
            
            return evaluation_result
            
        except Exception as e:
            st.error(f"安全评估失败: {str(e)}")
            return {"error": str(e)}
    
    def save_evaluation_result(self, result: Dict) -> bool:
        """保存评估结果"""
        try:
            result_file = os.path.join(
                self.results_dir, 
                f"{result['evaluation_id']}.json"
            )
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            st.error(f"保存评估结果失败: {str(e)}")
            return False
    
    def save_adversarial_samples(self, evaluation_id: str, original: np.ndarray, 
                               adversarial: np.ndarray, labels: np.ndarray) -> bool:
        """保存对抗样本"""
        try:
            samples_file = os.path.join(
                self.results_dir, 
                f"{evaluation_id}_samples.npz"
            )
            
            np.savez_compressed(
                samples_file,
                original=original,
                adversarial=adversarial,
                labels=labels
            )
            
            return True
        except Exception as e:
            st.error(f"保存对抗样本失败: {str(e)}")
            return False
    
    def load_evaluation_result(self, evaluation_id: str) -> Optional[Dict]:
        """加载评估结果"""
        try:
            result_file = os.path.join(self.results_dir, f"{evaluation_id}.json")
            
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            return None
        except Exception as e:
            st.error(f"加载评估结果失败: {str(e)}")
            return None
    
    def get_evaluation_history(self, user_id: str = None) -> List[Dict]:
        """获取评估历史"""
        results = []
        try:
            for filename in os.listdir(self.results_dir):
                if filename.endswith('.json'):
                    result_file = os.path.join(self.results_dir, filename)
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        
                        # 如果指定了用户ID，只返回该用户的结果
                        if user_id is None or result.get('model_info', {}).get('user_id') == user_id:
                            results.append(result)
        
        except Exception as e:
            st.error(f"获取评估历史失败: {str(e)}")
        
        return sorted(results, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    def generate_visualization(self, result: Dict) -> Dict:
        """生成可视化图表"""
        try:
            visualizations = {}
            
            # 1. 准确率对比图
            accuracy_fig = go.Figure(data=[
                go.Bar(
                    x=['原始模型', '对抗攻击后'],
                    y=[result['results']['original_accuracy'], 
                       result['results']['adversarial_accuracy']],
                    marker_color=['#2E86AB', '#A23B72'],
                    text=[f"{result['results']['original_accuracy']:.3f}", 
                          f"{result['results']['adversarial_accuracy']:.3f}"],
                    textposition='auto'
                )
            ])
            
            accuracy_fig.update_layout(
                title='模型准确率对比',
                yaxis_title='准确率',
                showlegend=False
            )
            
            visualizations['accuracy_comparison'] = accuracy_fig
            
            # 2. 鲁棒性指标雷达图
            metrics = {
                '原始准确率': result['results']['original_accuracy'],
                '鲁棒性得分': result['results']['robustness_score'],
                '抗攻击能力': 1.0 - result['results']['attack_success_rate'],
                '扰动敏感性': 1.0 - result['results']['perturbation_stats']['linf_norm']
            }
            
            radar_fig = go.Figure()
            
            radar_fig.add_trace(go.Scatterpolar(
                r=list(metrics.values()),
                theta=list(metrics.keys()),
                fill='toself',
                name='模型性能'
            ))
            
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title='模型鲁棒性雷达图'
            )
            
            visualizations['robustness_radar'] = radar_fig
            
            # 3. 扰动统计图
            perturbation_fig = go.Figure(data=[
                go.Bar(
                    x=['L0范数', 'L2范数', 'L∞范数'],
                    y=[result['results']['perturbation_stats']['l0_norm'],
                       result['results']['perturbation_stats']['l2_norm'],
                       result['results']['perturbation_stats']['linf_norm']],
                    marker_color=['#F18F01', '#C73E1D', '#592E83']
                )
            ])
            
            perturbation_fig.update_layout(
                title='扰动统计',
                yaxis_title='扰动大小'
            )
            
            visualizations['perturbation_stats'] = perturbation_fig
            
            return visualizations
            
        except Exception as e:
            st.error(f"生成可视化失败: {str(e)}")
            return {}
    
    def generate_report(self, result: Dict, report_format: str = 'html') -> str:
        """生成评估报告"""
        try:
            report_id = f"report_{result['evaluation_id']}"
            
            if report_format == 'html':
                report_content = self._generate_html_report(result)
                report_file = os.path.join(self.reports_dir, f"{report_id}.html")
            elif report_format == 'pdf':
                report_content = self._generate_pdf_report(result)
                report_file = os.path.join(self.reports_dir, f"{report_id}.pdf")
            else:
                report_content = self._generate_text_report(result)
                report_file = os.path.join(self.reports_dir, f"{report_id}.txt")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            return report_file
            
        except Exception as e:
            st.error(f"生成报告失败: {str(e)}")
            return None
    
    def _generate_html_report(self, result: Dict) -> str:
        """生成HTML报告"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI模型安全评估报告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin-bottom: 25px; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .success {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .danger {{ color: #dc3545; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI模型安全评估报告</h1>
                <p>评估ID: {result['evaluation_id']}</p>
                <p>评估时间: {result['timestamp']}</p>
            </div>
            
            <div class="section">
                <h2>评估概要</h2>
                <div class="metric">模型名称: {result['model_info']['name']}</div>
                <div class="metric">数据集: {result['dataset_info']['name']}</div>
                <div class="metric">攻击算法: {result['attack_config']['algorithm_name']}</div>
                <div class="metric">样本数量: {result['results']['sample_count']}</div>
            </div>
            
            <div class="section">
                <h2>评估结果</h2>
                <div class="metric">原始准确率: <span class="success">{result['results']['original_accuracy']:.3f}</span></div>
                <div class="metric">攻击后准确率: <span class="danger">{result['results']['adversarial_accuracy']:.3f}</span></div>
                <div class="metric">攻击成功率: <span class="warning">{result['results']['attack_success_rate']:.3f}</span></div>
                <div class="metric">鲁棒性得分: <span class="{'success' if result['results']['robustness_score'] > 0.7 else 'warning' if result['results']['robustness_score'] > 0.3 else 'danger'}">{result['results']['robustness_score']:.3f}</span></div>
            </div>
            
            <div class="section">
                <h2>扰动分析</h2>
                <div class="metric">L0范数: {result['results']['perturbation_stats']['l0_norm']:.6f}</div>
                <div class="metric">L2范数: {result['results']['perturbation_stats']['l2_norm']:.6f}</div>
                <div class="metric">L∞范数: {result['results']['perturbation_stats']['linf_norm']:.6f}</div>
            </div>
            
            <div class="section">
                <h2>安全建议</h2>
                <ul>
                    {'<li>模型具有良好的鲁棒性，建议继续保持当前的安全措施。</li>' if result['results']['robustness_score'] > 0.7 else ''}
                    {'<li>模型鲁棒性中等，建议考虑对抗训练或防御机制。</li>' if 0.3 <= result['results']['robustness_score'] <= 0.7 else ''}
                    {'<li>模型鲁棒性较差，强烈建议实施对抗训练和多层防御策略。</li>' if result['results']['robustness_score'] < 0.3 else ''}
                    <li>定期进行安全评估，监控模型在新攻击下的表现。</li>
                    <li>考虑部署输入验证和异常检测机制。</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_text_report(self, result: Dict) -> str:
        """生成文本报告"""
        report = f"""
===========================================
        AI模型安全评估报告
===========================================

评估ID: {result['evaluation_id']}
评估时间: {result['timestamp']}

-------------------------------------------
评估概要
-------------------------------------------
模型名称: {result['model_info']['name']}
模型类型: {result['model_info']['model_type']}
数据集: {result['dataset_info']['name']}
攻击算法: {result['attack_config']['algorithm_name']}
样本数量: {result['results']['sample_count']}

-------------------------------------------
评估结果
-------------------------------------------
原始准确率: {result['results']['original_accuracy']:.3f}
攻击后准确率: {result['results']['adversarial_accuracy']:.3f}
攻击成功率: {result['results']['attack_success_rate']:.3f}
鲁棒性得分: {result['results']['robustness_score']:.3f}
成功攻击数: {result['results']['successful_attacks']}

-------------------------------------------
扰动分析
-------------------------------------------
L0范数 (稀疏性): {result['results']['perturbation_stats']['l0_norm']:.6f}
L2范数 (欧几里得距离): {result['results']['perturbation_stats']['l2_norm']:.6f}
L∞范数 (最大扰动): {result['results']['perturbation_stats']['linf_norm']:.6f}

-------------------------------------------
安全评级
-------------------------------------------
{'🟢 优秀 - 模型具有很强的鲁棒性' if result['results']['robustness_score'] > 0.8 else ''}
{'🟡 良好 - 模型具有较好的鲁棒性' if 0.6 <= result['results']['robustness_score'] <= 0.8 else ''}
{'🟠 一般 - 模型鲁棒性中等，需要改进' if 0.3 <= result['results']['robustness_score'] < 0.6 else ''}
{'🔴 较差 - 模型鲁棒性不足，存在安全风险' if result['results']['robustness_score'] < 0.3 else ''}

-------------------------------------------
改进建议
-------------------------------------------
1. 实施对抗训练提高模型鲁棒性
2. 部署输入预处理和验证机制
3. 考虑集成多个防御策略
4. 定期进行安全评估和监控
5. 建立攻击检测和响应机制

===========================================
        """
        
        return report
    
    def get_storage_stats(self) -> Dict:
        """获取存储统计信息"""
        stats = {
            "total_evaluations": 0,
            "total_size": 0,
            "by_user": {},
            "by_model_type": {},
            "recent_evaluations": 0  # 最近30天的评估数量
        }
        
        try:
            from datetime import datetime, timedelta
            thirty_days_ago = datetime.now() - timedelta(days=30)
            
            for filename in os.listdir(self.results_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.results_dir, filename)
                    file_size = os.path.getsize(file_path)
                    
                    stats["total_evaluations"] += 1
                    stats["total_size"] += file_size
                    
                    # 读取评估结果获取更多统计信息
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            result = json.load(f)
                            
                            # 按用户统计
                            user_id = result.get('model_info', {}).get('uploader', 'Unknown')
                            if user_id not in stats["by_user"]:
                                stats["by_user"][user_id] = {'count': 0, 'size': 0}
                            stats["by_user"][user_id]['count'] += 1
                            stats["by_user"][user_id]['size'] += file_size
                            
                            # 按模型类型统计
                            model_type = result.get('model_info', {}).get('model_type', 'Unknown')
                            if model_type not in stats["by_model_type"]:
                                stats["by_model_type"][model_type] = {'count': 0, 'size': 0}
                            stats["by_model_type"][model_type]['count'] += 1
                            stats["by_model_type"][model_type]['size'] += file_size
                            
                            # 统计最近30天的评估
                            timestamp_str = result.get('timestamp', '')
                            if timestamp_str:
                                try:
                                    eval_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                                    if eval_time >= thirty_days_ago:
                                        stats["recent_evaluations"] += 1
                                except ValueError:
                                    pass  # 忽略时间格式错误
                                    
                    except (json.JSONDecodeError, KeyError):
                        continue  # 忽略损坏的文件
                        
        except Exception as e:
            st.error(f"获取存储统计失败: {str(e)}")
            
        return stats
    
    def get_completed_evaluations(self) -> List[Dict]:
        """获取所有已完成的评估（管理员用）"""
        try:
            evaluations = []
            for filename in os.listdir(self.results_dir):
                if filename.endswith('.json'):
                    result_file = os.path.join(self.results_dir, filename)
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        
                        # 转换为页面期望的格式
                        evaluation = {
                            'id': result.get('evaluation_id'),
                            'name': f"{result.get('model_info', {}).get('name', 'Unknown')} vs {result.get('attack_config', {}).get('algorithm_name', 'Unknown')}",
                            'type': 'security_evaluation',
                            'completed_at': result.get('timestamp'),
                            'results': result.get('results', {}),
                            'config': {
                                'model': result.get('model_info', {}),
                                'dataset': result.get('dataset_info', {}),
                                'attack_configs': [result.get('attack_config', {})]
                            }
                        }
                        evaluations.append(evaluation)
            
            return sorted(evaluations, key=lambda x: x.get('completed_at', ''), reverse=True)
            
        except Exception as e:
            st.error(f"获取已完成评估失败: {str(e)}")
            return []
    
    def get_user_completed_evaluations(self, user_id: str) -> List[Dict]:
        """获取指定用户的已完成评估"""
        try:
            evaluations = []
            for filename in os.listdir(self.results_dir):
                if filename.endswith('.json'):
                    result_file = os.path.join(self.results_dir, filename)
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        
                        # 检查是否属于指定用户
                        if result.get('model_info', {}).get('uploader') == user_id:
                            # 转换为页面期望的格式
                            evaluation = {
                                'id': result.get('evaluation_id'),
                                'name': f"{result.get('model_info', {}).get('name', 'Unknown')} vs {result.get('attack_config', {}).get('algorithm_name', 'Unknown')}",
                                'type': 'security_evaluation',
                                'completed_at': result.get('timestamp'),
                                'results': result.get('results', {}),
                                'config': {
                                    'model': result.get('model_info', {}),
                                    'dataset': result.get('dataset_info', {}),
                                    'attack_configs': [result.get('attack_config', {})]
                                }
                            }
                            evaluations.append(evaluation)
            
            return sorted(evaluations, key=lambda x: x.get('completed_at', ''), reverse=True)
            
        except Exception as e:
            st.error(f"获取用户已完成评估失败: {str(e)}")
            return []
    
    def start_evaluation(self, evaluation_config: Dict) -> str:
        """启动安全评估"""
        try:
            # 获取攻击配置列表，取第一个进行评估
            attack_configs = evaluation_config.get('attack_configs', [])
            if not attack_configs:
                st.error("没有找到攻击配置")
                return None
            
            # 目前只支持单个攻击配置的评估，取第一个
            raw_attack_config = attack_configs[0]
            
            # 提取实际的攻击配置：如果有 'config' 字段，使用它；否则直接使用原配置
            if 'config' in raw_attack_config:
                attack_config = raw_attack_config['config']
            else:
                attack_config = raw_attack_config
            
            # 准备评估参数
            evaluation_params = evaluation_config.get('parameters', {})
            
            # 直接调用现有的评估方法
            result = self.evaluate_model_robustness(
                evaluation_config['model'],
                evaluation_config['dataset'], 
                attack_config,
                evaluation_params
            )
            
            if 'error' in result:
                st.error(f"评估失败: {result['error']}")
                return None
            
            return result.get('evaluation_id')
            
        except Exception as e:
            st.error(f"启动评估失败: {str(e)}")
            return None
    
    def get_all_evaluations(self) -> List[Dict]:
        """获取所有评估（管理员用）"""
        try:
            evaluations = []
            for filename in os.listdir(self.results_dir):
                if filename.endswith('.json'):
                    result_file = os.path.join(self.results_dir, filename)
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        
                        # 转换为页面期望的格式
                        evaluation = {
                            'id': result.get('evaluation_id'),
                            'name': f"{result.get('model_info', {}).get('name', 'Unknown')} vs {result.get('attack_config', {}).get('algorithm_name', 'Unknown')}",
                            'type': 'security_evaluation',
                            'status': '已完成',  # 所有保存的评估都是已完成的
                            'created_at': result.get('timestamp'),
                            'completed_at': result.get('timestamp'),
                            'description': f"使用 {result.get('attack_config', {}).get('algorithm_name', 'Unknown')} 攻击算法对 {result.get('model_info', {}).get('name', 'Unknown')} 模型进行安全评估",
                            'results': result.get('results', {}),
                            'config': {
                                'model': result.get('model_info', {}),
                                'dataset': result.get('dataset_info', {}),
                                'attack_configs': [result.get('attack_config', {})]
                            }
                        }
                        evaluations.append(evaluation)
            
            return sorted(evaluations, key=lambda x: x.get('created_at', ''), reverse=True)
            
        except Exception as e:
            st.error(f"获取所有评估失败: {str(e)}")
            return []
    
    def get_user_evaluations(self, user_id: str) -> List[Dict]:
        """获取指定用户的评估"""
        try:
            evaluations = []
            for filename in os.listdir(self.results_dir):
                if filename.endswith('.json'):
                    result_file = os.path.join(self.results_dir, filename)
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        
                        # 检查是否属于指定用户
                        if result.get('model_info', {}).get('uploader') == user_id:
                            # 转换为页面期望的格式
                            evaluation = {
                                'id': result.get('evaluation_id'),
                                'name': f"{result.get('model_info', {}).get('name', 'Unknown')} vs {result.get('attack_config', {}).get('algorithm_name', 'Unknown')}",
                                'type': 'security_evaluation',
                                'status': '已完成',  # 所有保存的评估都是已完成的
                                'created_at': result.get('timestamp'),
                                'completed_at': result.get('timestamp'),
                                'description': f"使用 {result.get('attack_config', {}).get('algorithm_name', 'Unknown')} 攻击算法对 {result.get('model_info', {}).get('name', 'Unknown')} 模型进行安全评估",
                                'results': result.get('results', {}),
                                'config': {
                                    'model': result.get('model_info', {}),
                                    'dataset': result.get('dataset_info', {}),
                                    'attack_configs': [result.get('attack_config', {})]
                                }
                            }
                            evaluations.append(evaluation)
            
            return sorted(evaluations, key=lambda x: x.get('created_at', ''), reverse=True)
            
        except Exception as e:
            st.error(f"获取用户评估失败: {str(e)}")
            return []
    
    def delete_evaluation(self, evaluation_id: str, user_id: str) -> bool:
        """删除评估记录"""
        try:
            result_file = os.path.join(self.results_dir, f"{evaluation_id}.json")
            
            if os.path.exists(result_file):
                # 检查权限（非管理员只能删除自己的评估）
                with open(result_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    
                # 这里应该从session state或其他地方获取用户角色
                # 暂时假设可以删除
                model_uploader = result.get('model_info', {}).get('uploader')
                if model_uploader == user_id or user_id == 'admin':  # 简化的权限检查
                    os.remove(result_file)
                    
                    # 同时删除对抗样本文件（如果存在）
                    samples_file = os.path.join(self.results_dir, f"{evaluation_id}_samples.npz")
                    if os.path.exists(samples_file):
                        os.remove(samples_file)
                    
                    return True
                else:
                    st.error("权限不足：只能删除自己的评估记录")
                    return False
            else:
                st.error("评估记录不存在")
                return False
                
        except Exception as e:
            st.error(f"删除评估记录失败: {str(e)}")
            return False

# 在 prepare_dataset 方法中添加标签重映射
def prepare_dataset(self, dataset_info: Dict, sample_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
    # ... existing code ...
    
    # 标签重映射：将原始标签映射到0-999范围
    if y_data.min() != 0 or y_data.max() >= 1000:
        print(f"检测到标签范围异常: [{y_data.min()}, {y_data.max()}]")
        print("执行标签重映射...")
        
        # 创建标签映射
        unique_labels = np.unique(y_data)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        
        # 应用映射
        y_data_mapped = np.array([label_mapping[label] for label in y_data])
        
        print(f"重映射后标签范围: [{y_data_mapped.min()}, {y_data_mapped.max()}]")
        print(f"映射了 {len(unique_labels)} 个唯一类别")
        
        y_data = y_data_mapped
    
    # ... existing code ...

