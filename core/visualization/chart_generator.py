import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any
import streamlit as st

class ChartGenerator:
    """图表生成器"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'info': '#17a2b8'
        }
    
    def generate_charts(self, result: Dict) -> Dict:
        """生成所有图表"""
        try:
            charts = {}
            
            # 1. 准确率对比图
            charts['accuracy_comparison'] = self._create_accuracy_comparison(result)
            
            # 2. 鲁棒性雷达图
            charts['robustness_radar'] = self._create_robustness_radar(result)
            
            # 3. 扰动统计图
            charts['perturbation_stats'] = self._create_perturbation_chart(result)
            
            # 4. 攻击效果分析图
            charts['attack_analysis'] = self._create_attack_analysis(result)
            
            return charts
            
        except Exception as e:
            st.error(f"生成图表失败: {str(e)}")
            return {}
    
    def _create_accuracy_comparison(self, result: Dict) -> go.Figure:
        """创建准确率对比图"""
        fig = go.Figure(data=[
            go.Bar(
                x=['原始模型', '对抗攻击后'],
                y=[result['results']['original_accuracy'], 
                   result['results']['adversarial_accuracy']],
                marker_color=[self.color_palette['primary'], self.color_palette['danger']],
                text=[f"{result['results']['original_accuracy']:.3f}", 
                      f"{result['results']['adversarial_accuracy']:.3f}"],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='模型准确率对比',
            yaxis_title='准确率',
            showlegend=False,
            height=400
        )
        
        return fig
    
    def _create_robustness_radar(self, result: Dict) -> go.Figure:
        """创建鲁棒性雷达图"""
        metrics = {
            '原始准确率': result['results']['original_accuracy'],
            '鲁棒性得分': result['results']['robustness_score'],
            '抗攻击能力': 1.0 - result['results']['attack_success_rate'],
            '扰动敏感性': max(0, 1.0 - result['results']['perturbation_stats']['linf_norm'] * 10)
        }
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=list(metrics.values()),
            theta=list(metrics.keys()),
            fill='toself',
            name='模型性能',
            line_color=self.color_palette['primary']
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='模型鲁棒性雷达图',
            height=500
        )
        
        return fig
    
    def _create_perturbation_chart(self, result: Dict) -> go.Figure:
        """创建扰动统计图"""
        perturbation_stats = result['results']['perturbation_stats']
        
        fig = go.Figure(data=[
            go.Bar(
                x=['L0范数', 'L2范数', 'L∞范数'],
                y=[perturbation_stats['l0_norm'],
                   perturbation_stats['l2_norm'],
                   perturbation_stats['linf_norm']],
                marker_color=[self.color_palette['info'], 
                             self.color_palette['warning'], 
                             self.color_palette['danger']],
                text=[f"{perturbation_stats['l0_norm']:.4f}",
                      f"{perturbation_stats['l2_norm']:.4f}",
                      f"{perturbation_stats['linf_norm']:.4f}"],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='扰动统计分析',
            yaxis_title='扰动大小',
            height=400
        )
        
        return fig
    
    def _create_attack_analysis(self, result: Dict) -> go.Figure:
        """创建攻击效果分析图"""
        # 创建子图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('攻击成功率', '样本分布'),
            specs=[[{'type': 'indicator'}, {'type': 'pie'}]]
        )
        
        # 攻击成功率指示器
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=result['results']['attack_success_rate'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "攻击成功率 (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.color_palette['danger']},
                    'steps': [
                        {'range': [0, 30], 'color': self.color_palette['success']},
                        {'range': [30, 70], 'color': self.color_palette['warning']},
                        {'range': [70, 100], 'color': self.color_palette['danger']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # 样本分布饼图
        successful_attacks = result['results']['successful_attacks']
        total_correct = result['results']['correctly_classified_count']
        failed_attacks = total_correct - successful_attacks
        
        fig.add_trace(
            go.Pie(
                labels=['攻击成功', '攻击失败'],
                values=[successful_attacks, failed_attacks],
                marker_colors=[self.color_palette['danger'], self.color_palette['success']]
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='攻击效果综合分析',
            height=400
        )
        
        return fig