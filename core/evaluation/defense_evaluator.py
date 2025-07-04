import streamlit as st
from ..defense.defense_manager import DefenseManager
from .metrics_calculator import MetricsCalculator

class DefenseEvaluator:
    def __init__(self):
        self.defense_manager = DefenseManager()
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate_defense(self, model, x_clean, y_true, x_adversarial, defense_config):
        """评估防御效果"""
        try:
            # 应用防御方法
            defense_method = defense_config['method']
            defense_params = defense_config['params']
            
            # 对对抗样本应用防御
            x_defended = self._apply_defense(x_adversarial, defense_method, defense_params)
            
            # 计算防御指标
            defense_metrics = self.metrics_calculator.calculate_defense_metrics(
                model, x_clean, y_true, x_adversarial, x_defended, defense_config
            )
            
            return defense_metrics
            
        except Exception as e:
            st.error(f"防御评估失败: {str(e)}")
            return None
    
    def _apply_defense(self, x_data, method, params):
        """应用防御方法"""
        defense_instance = self.defense_manager.create_defense_instance(
            method, params, None
        )
        return defense_instance.defend(x_data)