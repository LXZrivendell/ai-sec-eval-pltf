import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import streamlit as st

class ResultManager:
    """评估结果管理类"""
    
    def __init__(self, results_dir: str = "data/evaluation_results"):
        self.results_dir = results_dir
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保目录存在"""
        os.makedirs(self.results_dir, exist_ok=True)
    
    def create_evaluation_result(self, model_info: Dict, dataset_info: Dict,
                               attack_config: Dict, evaluation_params: Dict,
                               baseline_metrics: Dict, attack_metrics: Dict,
                               attack_stats: Dict) -> Dict:
        """创建评估结果"""
        try:
            evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            evaluation_result = {
                "evaluation_id": evaluation_id,
                "timestamp": datetime.now().isoformat(),
                "model_info": model_info,
                "dataset_info": dataset_info,
                "attack_config": attack_config,
                "evaluation_params": evaluation_params,
                "results": {
                    "original_accuracy": baseline_metrics['original_accuracy'],
                    "adversarial_accuracy": attack_metrics['adversarial_accuracy'],
                    "attack_success_rate": attack_metrics['attack_success_rate'],
                    "robustness_score": attack_metrics['robustness_score'],
                    "perturbation_stats": attack_metrics['perturbation_stats'],
                    "sample_count": baseline_metrics['total_samples'],
                    "correctly_classified_count": baseline_metrics['correctly_classified_count'],
                    "successful_attacks": attack_metrics['attack_success_count']
                },
                "attack_stats": attack_stats,
                "detailed_metrics": {
                    "original_classification_report": self._safe_classification_report(
                        baseline_metrics['y_true'], baseline_metrics['y_pred_original']
                    ),
                    "adversarial_classification_report": self._safe_classification_report(
                        baseline_metrics['y_true'], attack_metrics['full_adversarial_predictions']
                    )
                }
            }
            
            return evaluation_result
            
        except Exception as e:
            st.error(f"创建评估结果失败: {str(e)}")
            return None
    
    def _safe_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """安全生成分类报告"""
        try:
            from sklearn.metrics import classification_report
            return classification_report(y_true, y_pred, output_dict=True)
        except Exception:
            return {}
    
    def save_evaluation_result(self, result: Dict) -> bool:
        """保存评估结果"""
        try:
            result_file = os.path.join(
                self.results_dir, 
                f"{result['evaluation_id']}.json"
            )
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=self._json_serializer)
            
            st.success(f"评估结果已保存: {result['evaluation_id']}")
            return True
            
        except Exception as e:
            st.error(f"保存评估结果失败: {str(e)}")
            return False
    
    def _json_serializer(self, obj):
        """JSON序列化辅助函数"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
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
                        # 修复：使用 uploader 字段而不是 uploaded_by
                        if user_id is None or result.get('model_info', {}).get('uploader') == user_id:
                            results.append(result)
        
        except Exception as e:
            st.error(f"获取评估历史失败: {str(e)}")
        
        return sorted(results, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    def get_storage_stats(self) -> Dict:
        """获取存储统计信息"""
        stats = {
            "total_evaluations": 0,
            "total_size": 0,
            "by_user": {},
            "by_model_type": {},
            "recent_evaluations": 0
        }
        
        try:
            thirty_days_ago = datetime.now() - timedelta(days=30)
            
            for filename in os.listdir(self.results_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.results_dir, filename)
                    file_size = os.path.getsize(file_path)
                    
                    stats["total_evaluations"] += 1
                    stats["total_size"] += file_size
                    
                    # 读取详细信息
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
                            
                            # 统计最近评估
                            timestamp_str = result.get('timestamp', '')
                            if timestamp_str:
                                try:
                                    eval_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                    if eval_time >= thirty_days_ago:
                                        stats["recent_evaluations"] += 1
                                except ValueError:
                                    pass
                                    
                    except (json.JSONDecodeError, KeyError):
                        continue
                        
        except Exception as e:
            st.error(f"获取存储统计失败: {str(e)}")
            
        return stats
    
    def delete_evaluation(self, evaluation_id: str, user_id: str = None) -> bool:
        """删除评估记录"""
        try:
            result_file = os.path.join(self.results_dir, f"{evaluation_id}.json")
            
            if os.path.exists(result_file):
                # 检查权限
                if user_id:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        
                    model_uploader = result.get('model_info', {}).get('uploader')
                    if model_uploader != user_id and user_id != 'admin':
                        st.error("权限不足：只能删除自己的评估记录")
                        return False
                
                os.remove(result_file)
                
                # 删除对抗样本文件
                samples_file = os.path.join(self.results_dir, f"{evaluation_id}_samples.npz")
                if os.path.exists(samples_file):
                    os.remove(samples_file)
                
                return True
            else:
                st.error("评估记录不存在")
                return False
                
        except Exception as e:
            st.error(f"删除评估记录失败: {str(e)}")
            return False
    
    # 在 ResultManager 类的末尾添加以下方法
    
    def get_all_evaluations(self) -> List[Dict]:
        """获取所有评估记录（管理员用）"""
        results = []
        try:
            for filename in os.listdir(self.results_dir):
                if filename.endswith('.json'):
                    result_file = os.path.join(self.results_dir, filename)
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        
                        # 转换为统一格式
                        evaluation = {
                            'id': result.get('evaluation_id'),
                            'name': f"{result.get('model_info', {}).get('name', 'Unknown')} vs {result.get('attack_config', {}).get('algorithm_name', 'Unknown')}",
                            'type': 'security_evaluation',
                            'status': '已完成',
                            'created_at': result.get('timestamp'),
                            'completed_at': result.get('timestamp'),
                            'user_id': result.get('model_info', {}).get('uploader', 'unknown'),
                            'results': result.get('results', {}),
                            'config': {
                                'model': result.get('model_info', {}),
                                'dataset': result.get('dataset_info', {}),
                                'attack_configs': [result.get('attack_config', {})]
                            }
                        }
                        results.append(evaluation)
    
        except Exception as e:
            st.error(f"获取所有评估记录失败: {str(e)}")
        
        return sorted(results, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def get_user_evaluations(self, user_id: str) -> List[Dict]:
        """获取指定用户的评估记录"""
        all_evaluations = self.get_all_evaluations()
        return [eval for eval in all_evaluations if eval.get('user_id') == user_id]