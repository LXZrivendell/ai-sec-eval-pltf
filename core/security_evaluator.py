import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st

from .model_loader import ModelLoader
from .dataset_manager import DatasetManager
from .attack_manager import AttackManager
from .evaluation import (
    EvaluationConfig, DataProcessor, ARTEstimatorManager,
    AttackExecutor, MemoryManager, MetricsCalculator, ResultManager
)
from .evaluation.defense_evaluator import DefenseEvaluator  # 添加这行
from .visualization import ChartGenerator
from .reporting import ReportGenerator

class SecurityEvaluator:
    """安全评估引擎 - 重构后的主协调器"""
    
    def __init__(self):
        # 核心组件
        self.model_loader = ModelLoader()
        self.dataset_manager = DatasetManager()
        self.attack_manager = AttackManager()
        
        # 评估组件
        self.data_processor = DataProcessor()
        self.estimator_manager = ARTEstimatorManager()
        self.memory_manager = MemoryManager()
        self.attack_executor = AttackExecutor(self.attack_manager, self.memory_manager)
        self.metrics_calculator = MetricsCalculator()
        self.result_manager = ResultManager()
        self.defense_evaluator = DefenseEvaluator()  # 现在应该可以正常工作
        
        # 可视化和报告组件
        self.chart_generator = ChartGenerator()
        self.report_generator = ReportGenerator()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def evaluate_model_robustness(self, model_info: Dict, dataset_info: Dict, 
                                attack_config: Dict, evaluation_params: Dict) -> Dict:
        """评估模型鲁棒性 - 简化的主流程"""
        try:
            # 1. 配置验证
            eval_config = EvaluationConfig(
                sample_size=evaluation_params.get('sample_size', 1000),
                batch_size=attack_config.get('advanced_options', {}).get('batch_size', 32)
            )
            eval_config.validate()
            
            # 2. 数据准备
            st.info("🔄 准备数据集...")
            x_data, y_data = self.data_processor.prepare_dataset(
                dataset_info, eval_config.sample_size
            )
            
            if x_data is None or y_data is None:
                return {"error": "数据准备失败"}
            
            # 3. 创建估计器
            st.info("🤖 创建模型估计器...")
            input_shape = x_data.shape[1:]
            estimator = self.estimator_manager.create_estimator(model_info, input_shape)
            
            if estimator is None:
                return {"error": "模型加载失败"}
            
            # 4. 验证估计器
            if not self.estimator_manager.validate_estimator(estimator, x_data):
                return {"error": "模型验证失败"}
            
            # 5. 计算基线指标
            st.info("📊 计算基线指标...")
            baseline_metrics = self.metrics_calculator.calculate_baseline_metrics(
                estimator, x_data, y_data
            )
            
            if baseline_metrics is None:
                return {"error": "基线指标计算失败"}
            
            if len(baseline_metrics['correctly_classified_indices']) == 0:
                return {"error": "模型在测试数据上准确率为0，无法进行攻击评估"}
            
            # 6. 执行攻击
            st.info("⚔️ 执行对抗攻击...")
            correctly_classified_indices = baseline_metrics['correctly_classified_indices']
            x_correct = x_data[correctly_classified_indices]
            y_correct = y_data[correctly_classified_indices] if y_data is not None else None
            
            adversarial_samples, attack_stats = self.attack_executor.execute_attack(
                estimator, x_correct, y_correct, attack_config, eval_config.batch_size
            )
            
            if adversarial_samples is None:
                return {"error": "攻击执行失败"}
            
            # 7. 计算攻击指标
            st.info("📈 计算攻击效果...")
            attack_metrics = self.metrics_calculator.calculate_attack_metrics(
                estimator, x_data, y_data, adversarial_samples, baseline_metrics
            )
            
            if attack_metrics is None:
                return {"error": "攻击指标计算失败"}
            
            # 8. 生成评估结果
            evaluation_result = self.result_manager.create_evaluation_result(
                model_info, dataset_info, attack_config, evaluation_params,
                baseline_metrics, attack_metrics, attack_stats
            )
            
            if evaluation_result is None:
                return {"error": "评估结果生成失败"}
            
            # 9. 保存结果
            if self.result_manager.save_evaluation_result(evaluation_result):
                # 保存对抗样本（可选）
                if eval_config.save_adversarial_samples:
                    self.result_manager.save_adversarial_samples(
                        evaluation_result['evaluation_id'],
                        x_correct, adversarial_samples,
                        y_correct if y_correct is not None else baseline_metrics['y_true'][correctly_classified_indices]
                    )
                st.success("✅ 评估完成并保存成功")
            
            return evaluation_result
            
        except Exception as e:
            st.error(f"安全评估失败: {str(e)}")
            return {"error": str(e)}
    
    # 保留必要的接口方法，委托给相应的模块
    def get_evaluation_history(self, user_id: str = None) -> List[Dict]:
        """获取评估历史"""
        return self.result_manager.get_evaluation_history(user_id)
    
    def generate_visualization(self, result: Dict) -> Dict:
        """生成可视化图表"""
        return self.chart_generator.generate_charts(result)
    
    def generate_report(self, result: Dict, report_format: str = 'html') -> str:
        """生成评估报告"""
        return self.report_generator.generate_report(result, report_format)
    
    def get_storage_stats(self) -> Dict:
        """获取存储统计信息"""
        return self.result_manager.get_storage_stats()
    
    def load_evaluation_result(self, evaluation_id: str) -> Optional[Dict]:
        """加载评估结果"""
        return self.result_manager.load_evaluation_result(evaluation_id)
    
    def delete_evaluation(self, evaluation_id: str, user_id: str = None) -> bool:
        """删除评估记录"""
        return self.result_manager.delete_evaluation(evaluation_id, user_id)
    
    # 为了兼容性，保留一些原有方法的简化版本
    def get_completed_evaluations(self) -> List[Dict]:
        """获取所有已完成的评估（管理员用）"""
        try:
            evaluations = []
            history = self.get_evaluation_history()
            
            for result in history:
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
            
            return evaluations
            
        except Exception as e:
            st.error(f"获取已完成评估失败: {str(e)}")
            return []
    
    def get_user_completed_evaluations(self, user_id: str) -> List[Dict]:
        """获取指定用户的已完成评估"""
        try:
            evaluations = []
            history = self.get_evaluation_history(user_id)
            
            for result in history:
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
            
            return evaluations
            
        except Exception as e:
            st.error(f"获取用户已完成评估失败: {str(e)}")
            return []
    
    def start_evaluation(self, evaluation_config: Dict) -> str:
        """启动安全评估"""
        try:
            attack_configs = evaluation_config.get('attack_configs', [])
            if not attack_configs:
                st.error("没有找到攻击配置")
                return None
            
            # 取第一个攻击配置
            raw_attack_config = attack_configs[0]
            attack_config = raw_attack_config.get('config', raw_attack_config)
            
            evaluation_params = evaluation_config.get('parameters', {})
            
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
        """获取所有评估记录（管理员用）"""
        return self.result_manager.get_all_evaluations()
    
    def get_user_evaluations(self, user_id: str) -> List[Dict]:
        """获取指定用户的评估记录"""
        return self.result_manager.get_user_evaluations(user_id)
    
    def get_evaluation_stats(self) -> Dict:
        """获取评估统计信息"""
        try:
            stats = {
                'total_evaluations': 0,
                'completed_evaluations': 0,
                'running_evaluations': 0,
                'evaluation_types': {},
                'user_activity': {}
            }
            
            # 获取所有评估历史
            history = self.get_evaluation_history()
            stats['total_evaluations'] = len(history)
            stats['completed_evaluations'] = len([r for r in history if 'results' in r])
            stats['running_evaluations'] = len([r for r in history if 'results' not in r])
            
            # 统计评估类型
            for result in history:
                attack_name = result.get('attack_config', {}).get('algorithm_name', 'Unknown')
                stats['evaluation_types'][attack_name] = stats['evaluation_types'].get(attack_name, 0) + 1
            
            # 统计用户活动 - 修复数据结构
            for result in history:
                user_id = result.get('user_id', 'anonymous')
                if user_id not in stats['user_activity']:
                    stats['user_activity'][user_id] = {
                        'total': 0,
                        'completed': 0,
                        'running': 0,
                        'failed': 0
                    }
                
                stats['user_activity'][user_id]['total'] += 1
                
                # 根据结果状态分类
                if 'results' in result:
                    if result.get('results', {}).get('error'):
                        stats['user_activity'][user_id]['failed'] += 1
                    else:
                        stats['user_activity'][user_id]['completed'] += 1
                else:
                    stats['user_activity'][user_id]['running'] += 1
            
            return stats
            
        except Exception as e:
            st.error(f"获取评估统计失败: {str(e)}")
            return {
                'total_evaluations': 0,
                'completed_evaluations': 0,
                'running_evaluations': 0,
                'evaluation_types': {},
                'user_activity': {}
            }
    
    def evaluate_model_with_defense(self, model_info, dataset_info, 
                                  attack_config, defense_config, evaluation_params):
        """带防御的模型评估"""
        try:
            # 1. 执行基础攻击评估
            attack_result = self.evaluate_model_robustness(
                model_info, dataset_info, attack_config, evaluation_params
            )
            
            if 'error' in attack_result:
                return attack_result
            
            # 2. 执行防御评估
            defense_result = self.defense_evaluator.evaluate_defense(
                attack_result['model'], 
                attack_result['clean_data'],
                attack_result['labels'],
                attack_result['adversarial_samples'],
                defense_config
            )
            
            # 3. 合并结果
            combined_result = {
                **attack_result,
                'defense_metrics': defense_result,
                'evaluation_type': 'attack_and_defense'
            }
            
            return combined_result
            
        except Exception as e:
            return {"error": f"带防御的评估失败: {str(e)}"}