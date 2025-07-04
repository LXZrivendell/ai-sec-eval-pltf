import numpy as np
from typing import Dict, Any, Tuple, List
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class MetricsCalculator:
    """指标计算专用类"""
    
    def __init__(self):
        self.supported_norms = ['l0', 'l2', 'linf']
    
    def calculate_baseline_metrics(self, estimator: Any, x_data: np.ndarray, 
                                 y_data: np.ndarray) -> Dict:
        """计算基线指标"""
        try:
            # 获取原始预测
            original_predictions = estimator.predict(x_data)
            
            # 处理标签格式
            if len(y_data.shape) > 1 and y_data.shape[1] > 1:
                y_true = np.argmax(y_data, axis=1)
            else:
                y_true = y_data.flatten() if len(y_data.shape) > 1 else y_data
            
            y_pred_original = np.argmax(original_predictions, axis=1)
            
            # 找出正确分类的样本
            correctly_classified_mask = (y_pred_original == y_true)
            correctly_classified_indices = np.where(correctly_classified_mask)[0]
            
            original_accuracy = accuracy_score(y_true, y_pred_original)
            
            baseline_metrics = {
                'original_accuracy': float(original_accuracy),
                'correctly_classified_indices': correctly_classified_indices,
                'correctly_classified_count': len(correctly_classified_indices),
                'total_samples': len(x_data),
                'y_true': y_true,
                'y_pred_original': y_pred_original,
                'original_predictions': original_predictions
            }
            
            st.info(f"基线指标: 准确率={original_accuracy:.3f}, 正确分类样本={len(correctly_classified_indices)}/{len(x_data)}")
            
            return baseline_metrics
            
        except Exception as e:
            st.error(f"计算基线指标失败: {str(e)}")
            return None
    
    def calculate_attack_metrics(self, estimator: Any, x_original: np.ndarray, 
                               y_data: np.ndarray, adversarial_samples: np.ndarray,
                               baseline_metrics: Dict) -> Dict:
        """计算攻击效果指标"""
        try:
            # 获取对抗样本预测
            adversarial_predictions = estimator.predict(adversarial_samples)
            y_pred_adversarial = np.argmax(adversarial_predictions, axis=1)
            
            # 获取基线数据
            correctly_classified_indices = baseline_metrics['correctly_classified_indices']
            y_true = baseline_metrics['y_true']
            y_pred_original = baseline_metrics['y_pred_original']
            
            # 计算攻击成功率（仅针对原本正确分类的样本）
            y_true_correct = y_true[correctly_classified_indices]
            attack_successful_mask = (y_pred_adversarial != y_true_correct)
            attack_success_count = np.sum(attack_successful_mask)
            attack_success_rate = attack_success_count / len(y_true_correct) if len(y_true_correct) > 0 else 0
            
            # 计算整体对抗准确率
            full_adversarial_predictions = np.copy(y_pred_original)
            full_adversarial_predictions[correctly_classified_indices] = y_pred_adversarial
            adversarial_accuracy = accuracy_score(y_true, full_adversarial_predictions)
            
            # 计算扰动统计
            x_correct = x_original[correctly_classified_indices]
            perturbation_stats = self.calculate_perturbation_statistics(
                x_correct, adversarial_samples
            )
            
            attack_metrics = {
                'adversarial_accuracy': float(adversarial_accuracy),
                'attack_success_rate': float(attack_success_rate),
                'attack_success_count': int(attack_success_count),
                'robustness_score': float(1.0 - attack_success_rate),
                'perturbation_stats': perturbation_stats,
                'y_pred_adversarial': y_pred_adversarial,
                'full_adversarial_predictions': full_adversarial_predictions
            }
            
            st.info(f"攻击指标: 成功率={attack_success_rate:.3f}, 鲁棒性得分={1.0-attack_success_rate:.3f}")
            
            return attack_metrics
            
        except Exception as e:
            st.error(f"计算攻击指标失败: {str(e)}")
            return None
    
    def calculate_perturbation_statistics(self, original: np.ndarray, 
                                        adversarial: np.ndarray) -> Dict:
        """计算扰动统计信息"""
        try:
            perturbations = adversarial - original
            
            # L0范数（稀疏性）
            l0_norm = np.mean(np.sum(perturbations != 0, axis=tuple(range(1, len(perturbations.shape)))))
            
            # L2范数（欧几里得距离）
            l2_norm = np.mean(np.sqrt(np.sum(perturbations ** 2, axis=tuple(range(1, len(perturbations.shape))))))
            
            # L∞范数（最大扰动）
            linf_norm = np.mean(np.max(np.abs(perturbations), axis=tuple(range(1, len(perturbations.shape)))))
            
            return {
                'l0_norm': float(l0_norm),
                'l2_norm': float(l2_norm),
                'linf_norm': float(linf_norm),
                'mean_perturbation': float(np.mean(np.abs(perturbations))),
                'max_perturbation': float(np.max(np.abs(perturbations))),
                'min_perturbation': float(np.min(np.abs(perturbations)))
            }
            
        except Exception as e:
            st.error(f"计算扰动统计失败: {str(e)}")
            return {}
    
    def generate_classification_report(self, y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> Dict:
        """生成分类报告"""
        try:
            return classification_report(y_true, y_pred, output_dict=True)
        except Exception as e:
            st.error(f"生成分类报告失败: {str(e)}")
            return {}
    
    def calculate_robustness_score(self, attack_success_rate: float, 
                                 perturbation_magnitude: float = None) -> float:
        """计算综合鲁棒性得分"""
        base_score = 1.0 - attack_success_rate
        
        # 如果有扰动大小信息，可以进一步调整得分
        if perturbation_magnitude is not None:
            # 扰动越小，攻击越强，模型鲁棒性越差
            perturbation_factor = max(0.1, min(1.0, perturbation_magnitude * 10))
            adjusted_score = base_score * perturbation_factor
            return float(adjusted_score)
        
        return float(base_score)
    
    def calculate_defense_metrics(self, estimator, x_clean, y_true, x_adversarial, 
                                x_purified, defense_config):
        """计算防御评估指标"""
        try:
            # 基线指标
            clean_acc = accuracy_score(y_true, estimator.predict(x_clean).argmax(axis=1))
            adversarial_acc = accuracy_score(y_true, estimator.predict(x_adversarial).argmax(axis=1))
            purified_acc = accuracy_score(y_true, estimator.predict(x_purified).argmax(axis=1))
            
            # 计算新指标
            defense_metrics_calc = DefenseMetrics()
            
            metrics = {
                'clean_accuracy': float(clean_acc),
                'adversarial_accuracy': float(adversarial_acc),
                'purified_accuracy': float(purified_acc),
                'adversarial_accuracy_gap': defense_metrics_calc.calculate_adversarial_accuracy_gap(
                    clean_acc, adversarial_acc
                ),
                'purification_recovery_rate': defense_metrics_calc.calculate_purification_recovery_rate(
                    clean_acc, purified_acc, adversarial_acc
                ),
                'clean_accuracy_preservation': defense_metrics_calc.calculate_clean_accuracy_preservation(
                    clean_acc, purified_acc
                )
            }
            
            return metrics
            
        except Exception as e:
            st.error(f"计算防御指标失败: {str(e)}")
            return None