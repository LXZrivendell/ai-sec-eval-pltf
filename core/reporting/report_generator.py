import os
from datetime import datetime
from typing import Dict, Any
import streamlit as st

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, reports_dir: str = "data/reports"):
        self.reports_dir = reports_dir
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保目录存在"""
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def generate_report(self, result: Dict, report_format: str = 'html') -> str:
        """生成评估报告"""
        try:
            report_id = f"report_{result['evaluation_id']}"
            
            if report_format == 'html':
                report_content = self._generate_html_report(result)
                report_file = os.path.join(self.reports_dir, f"{report_id}.html")
            elif report_format == 'text':
                report_content = self._generate_text_report(result)
                report_file = os.path.join(self.reports_dir, f"{report_id}.txt")
            else:
                raise ValueError(f"不支持的报告格式: {report_format}")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            return report_file
            
        except Exception as e:
            st.error(f"生成报告失败: {str(e)}")
            return None
    
    def _generate_html_report(self, result: Dict) -> str:
        """生成HTML报告"""
        robustness_score = result['results']['robustness_score']
        security_level = self._get_security_level(robustness_score)
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI模型安全评估报告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f8f9fa; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; border-bottom: 3px solid #007bff; padding-bottom: 20px; }}
                .header h1 {{ color: #007bff; margin-bottom: 10px; }}
                .section {{ margin-bottom: 25px; }}
                .section h2 {{ color: #495057; border-left: 4px solid #007bff; padding-left: 15px; }}
                .metric {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #007bff; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }}
                .success {{ color: #28a745; font-weight: bold; }}
                .warning {{ color: #ffc107; font-weight: bold; }}
                .danger {{ color: #dc3545; font-weight: bold; }}
                .security-badge {{ display: inline-block; padding: 8px 16px; border-radius: 20px; color: white; font-weight: bold; }}
                .security-excellent {{ background-color: #28a745; }}
                .security-good {{ background-color: #17a2b8; }}
                .security-fair {{ background-color: #ffc107; color: #212529; }}
                .security-poor {{ background-color: #dc3545; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ border: 1px solid #dee2e6; padding: 12px; text-align: left; }}
                th {{ background-color: #e9ecef; font-weight: 600; }}
                .recommendations {{ background: #e7f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
                .recommendations ul {{ margin: 10px 0; padding-left: 20px; }}
                .recommendations li {{ margin: 8px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🛡️ AI模型安全评估报告</h1>
                    <p><strong>评估ID:</strong> {result['evaluation_id']}</p>
                    <p><strong>评估时间:</strong> {result['timestamp']}</p>
                    <div class="security-badge security-{security_level['class']}">
                        {security_level['text']}
                    </div>
                </div>
                
                <div class="section">
                    <h2>📋 评估概要</h2>
                    <div class="metric-grid">
                        <div class="metric"><strong>模型名称:</strong> {result['model_info']['name']}</div>
                        <div class="metric"><strong>模型类型:</strong> {result['model_info']['model_type']}</div>
                        <div class="metric"><strong>数据集:</strong> {result['dataset_info']['name']}</div>
                        <div class="metric"><strong>攻击算法:</strong> {result['attack_config']['algorithm_name']}</div>
                        <div class="metric"><strong>样本数量:</strong> {result['results']['sample_count']}</div>
                        <div class="metric"><strong>正确分类样本:</strong> {result['results']['correctly_classified_count']}</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 核心指标</h2>
                    <div class="metric-grid">
                        <div class="metric">
                            <strong>原始准确率:</strong> 
                            <span class="success">{result['results']['original_accuracy']:.3f}</span>
                        </div>
                        <div class="metric">
                            <strong>攻击后准确率:</strong> 
                            <span class="danger">{result['results']['adversarial_accuracy']:.3f}</span>
                        </div>
                        <div class="metric">
                            <strong>攻击成功率:</strong> 
                            <span class="warning">{result['results']['attack_success_rate']:.3f}</span>
                        </div>
                        <div class="metric">
                            <strong>鲁棒性得分:</strong> 
                            <span class="{self._get_score_class(robustness_score)}">{robustness_score:.3f}</span>
                        </div>
                        <div class="metric">
                            <strong>成功攻击数:</strong> 
                            {result['results']['successful_attacks']}
                        </div>
                        <div class="metric">
                            <strong>准确率下降:</strong> 
                            <span class="danger">{(result['results']['original_accuracy'] - result['results']['adversarial_accuracy']):.3f}</span>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔍 扰动分析</h2>
                    <table>
                        <tr>
                            <th>扰动类型</th>
                            <th>数值</th>
                            <th>说明</th>
                        </tr>
                        <tr>
                            <td>L0范数 (稀疏性)</td>
                            <td>{result['results']['perturbation_stats']['l0_norm']:.6f}</td>
                            <td>被修改的像素数量</td>
                        </tr>
                        <tr>
                            <td>L2范数 (欧几里得距离)</td>
                            <td>{result['results']['perturbation_stats']['l2_norm']:.6f}</td>
                            <td>扰动的整体大小</td>
                        </tr>
                        <tr>
                            <td>L∞范数 (最大扰动)</td>
                            <td>{result['results']['perturbation_stats']['linf_norm']:.6f}</td>
                            <td>单个像素的最大变化</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>💡 安全建议</h2>
                    <div class="recommendations">
                        {self._generate_recommendations(result)}
                    </div>
                </div>
                
                <div class="section">
                    <h2>⚙️ 技术细节</h2>
                    <div class="metric-grid">
                        <div class="metric"><strong>攻击参数:</strong> {str(result['attack_config'].get('params', {}))}</div>
                        <div class="metric"><strong>批次大小:</strong> {result['attack_config'].get('advanced_options', {}).get('batch_size', 'N/A')}</div>
                        <div class="metric"><strong>评估参数:</strong> {str(result['evaluation_params'])}</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📈 性能统计</h2>
                    <div class="metric-grid">
                        <div class="metric"><strong>总批次数:</strong> {result.get('attack_stats', {}).get('total_batches', 'N/A')}</div>
                        <div class="metric"><strong>成功批次:</strong> {result.get('attack_stats', {}).get('successful_batches', 'N/A')}</div>
                        <div class="metric"><strong>失败批次:</strong> {result.get('attack_stats', {}).get('failed_batches', 'N/A')}</div>
                        <div class="metric"><strong>内存清理次数:</strong> {result.get('attack_stats', {}).get('memory_cleanups', 'N/A')}</div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_text_report(self, result: Dict) -> str:
        """生成文本报告"""
        robustness_score = result['results']['robustness_score']
        security_level = self._get_security_level(robustness_score)
        
        report = f"""
===========================================
        AI模型安全评估报告
===========================================

评估ID: {result['evaluation_id']}
评估时间: {result['timestamp']}
安全等级: {security_level['text']}

-------------------------------------------
评估概要
-------------------------------------------
模型名称: {result['model_info']['name']}
模型类型: {result['model_info']['model_type']}
数据集: {result['dataset_info']['name']}
攻击算法: {result['attack_config']['algorithm_name']}
样本数量: {result['results']['sample_count']}
正确分类样本: {result['results']['correctly_classified_count']}

-------------------------------------------
核心指标
-------------------------------------------
原始准确率: {result['results']['original_accuracy']:.3f}
攻击后准确率: {result['results']['adversarial_accuracy']:.3f}
攻击成功率: {result['results']['attack_success_rate']:.3f}
鲁棒性得分: {robustness_score:.3f}
成功攻击数: {result['results']['successful_attacks']}
准确率下降: {(result['results']['original_accuracy'] - result['results']['adversarial_accuracy']):.3f}

-------------------------------------------
扰动分析
-------------------------------------------
L0范数 (稀疏性): {result['results']['perturbation_stats']['l0_norm']:.6f}
L2范数 (欧几里得距离): {result['results']['perturbation_stats']['l2_norm']:.6f}
L∞范数 (最大扰动): {result['results']['perturbation_stats']['linf_norm']:.6f}

-------------------------------------------
安全评级
-------------------------------------------
{self._get_security_rating_text(robustness_score)}

-------------------------------------------
改进建议
-------------------------------------------
{self._generate_text_recommendations(result)}

-------------------------------------------
技术细节
-------------------------------------------
攻击参数: {str(result['attack_config'].get('params', {}))}
批次大小: {result['attack_config'].get('advanced_options', {}).get('batch_size', 'N/A')}
评估参数: {str(result['evaluation_params'])}

===========================================
        """
        
        return report
    
    def _get_security_level(self, robustness_score: float) -> Dict:
        """获取安全等级"""
        if robustness_score > 0.8:
            return {'class': 'excellent', 'text': '🟢 优秀 - 模型具有很强的鲁棒性'}
        elif robustness_score > 0.6:
            return {'class': 'good', 'text': '🔵 良好 - 模型具有较好的鲁棒性'}
        elif robustness_score > 0.3:
            return {'class': 'fair', 'text': '🟡 一般 - 模型鲁棒性中等，需要改进'}
        else:
            return {'class': 'poor', 'text': '🔴 较差 - 模型鲁棒性不足，存在安全风险'}
    
    def _get_score_class(self, score: float) -> str:
        """获取得分样式类"""
        if score > 0.7:
            return 'success'
        elif score > 0.3:
            return 'warning'
        else:
            return 'danger'
    
    def _generate_recommendations(self, result: Dict) -> str:
        """生成HTML格式的建议"""
        robustness_score = result['results']['robustness_score']
        attack_success_rate = result['results']['attack_success_rate']
        
        recommendations = []
        
        if robustness_score < 0.3:
            recommendations.extend([
                "🚨 <strong>紧急建议:</strong> 模型鲁棒性严重不足，建议立即实施对抗训练",
                "🛡️ 部署多层防御策略，包括输入预处理和异常检测",
                "⚠️ 在生产环境中谨慎使用，考虑增加人工审核环节"
            ])
        elif robustness_score < 0.6:
            recommendations.extend([
                "📈 建议实施对抗训练以提高模型鲁棒性",
                "🔍 考虑部署输入验证和预处理机制",
                "📊 定期进行安全评估和监控"
            ])
        else:
            recommendations.extend([
                "✅ 模型具有良好的鲁棒性，建议继续保持当前的安全措施",
                "🔄 定期进行安全评估，监控模型在新攻击下的表现"
            ])
        
        # 通用建议
        recommendations.extend([
            "🏗️ 建立完善的模型安全监控体系",
            "📚 关注最新的对抗攻击研究，及时更新防御策略",
            "🤝 考虑与安全专家合作，制定全面的AI安全策略"
        ])
        
        return "<ul>" + "".join([f"<li>{rec}</li>" for rec in recommendations]) + "</ul>"
    
    def _generate_text_recommendations(self, result: Dict) -> str:
        """生成文本格式的建议"""
        robustness_score = result['results']['robustness_score']
        
        recommendations = []
        
        if robustness_score < 0.3:
            recommendations.extend([
                "1. 紧急实施对抗训练提高模型鲁棒性",
                "2. 部署多层防御策略和异常检测机制",
                "3. 在生产环境中增加人工审核环节"
            ])
        elif robustness_score < 0.6:
            recommendations.extend([
                "1. 实施对抗训练提高模型鲁棒性",
                "2. 部署输入验证和预处理机制",
                "3. 定期进行安全评估和监控"
            ])
        else:
            recommendations.extend([
                "1. 继续保持当前的安全措施",
                "2. 定期进行安全评估和监控"
            ])
        
        recommendations.extend([
            "4. 建立完善的模型安全监控体系",
            "5. 关注最新的对抗攻击研究",
            "6. 与安全专家合作制定AI安全策略"
        ])
        
        return "\n".join(recommendations)
    
    def _get_security_rating_text(self, robustness_score: float) -> str:
        """获取安全评级文本"""
        if robustness_score > 0.8:
            return "🟢 优秀 - 模型具有很强的鲁棒性"
        elif robustness_score > 0.6:
            return "🟡 良好 - 模型具有较好的鲁棒性"
        elif robustness_score > 0.3:
            return "🟠 一般 - 模型鲁棒性中等，需要改进"
        else:
            return "🔴 较差 - 模型鲁棒性不足，存在安全风险"
    
    def get_report_list(self):
        """获取报告列表"""
        reports = []
        
        if not os.path.exists(self.reports_dir):
            return reports
        
        try:
            for filename in os.listdir(self.reports_dir):
                filepath = os.path.join(self.reports_dir, filename)
                
                if os.path.isfile(filepath):
                    # 获取文件信息
                    stat = os.stat(filepath)
                    created_at = datetime.fromtimestamp(stat.st_ctime).isoformat()
                    size = stat.st_size
                    
                    # 确定文件格式
                    if filename.endswith('.html'):
                        format_type = 'HTML'
                    elif filename.endswith('.txt'):
                        format_type = 'TEXT'
                    elif filename.endswith('.json'):
                        format_type = 'JSON'
                    elif filename.endswith('.pdf'):
                        format_type = 'PDF'
                    else:
                        format_type = 'OTHER'
                    
                    # 尝试从文件名提取评估ID
                    evaluation_name = None
                    if filename.startswith('report_'):
                        evaluation_name = filename.replace('report_', '').split('.')[0]
                    
                    reports.append({
                        'filename': filename,
                        'filepath': filepath,
                        'format': format_type,
                        'size': size,
                        'created_at': created_at,
                        'evaluation_name': evaluation_name
                    })
            
            return reports
            
        except Exception as e:
            st.error(f"获取报告列表失败: {str(e)}")
            return []
    
    def delete_report(self, filepath: str) -> bool:
        """删除报告文件"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
            return False
        except Exception as e:
            st.error(f"删除报告失败: {str(e)}")
            return False
    
    def get_storage_stats(self):
        """获取存储统计信息"""
        stats = {
            'total_reports': 0,
            'total_size': 0,
            'by_format': {}
        }
        
        if not os.path.exists(self.reports_dir):
            return stats
        
        try:
            for filename in os.listdir(self.reports_dir):
                filepath = os.path.join(self.reports_dir, filename)
                
                if os.path.isfile(filepath):
                    file_size = os.path.getsize(filepath)
                    stats['total_reports'] += 1
                    stats['total_size'] += file_size
                    
                    # 确定文件格式
                    if filename.endswith('.html'):
                        format_type = 'HTML'
                    elif filename.endswith('.txt'):
                        format_type = 'TEXT'
                    elif filename.endswith('.json'):
                        format_type = 'JSON'
                    elif filename.endswith('.pdf'):
                        format_type = 'PDF'
                    else:
                        format_type = 'OTHER'
                    
                    # 按格式统计
                    if format_type not in stats['by_format']:
                        stats['by_format'][format_type] = {
                            'count': 0,
                            'size': 0
                        }
                    
                    stats['by_format'][format_type]['count'] += 1
                    stats['by_format'][format_type]['size'] += file_size
            
            return stats
            
        except Exception as e:
            st.error(f"获取存储统计失败: {str(e)}")
            return stats
    
    def generate_summary_report(self, evaluations):
        """生成汇总报告"""
        try:
            import json
            
            # 生成汇总数据
            summary_data = {
                'report_type': 'summary',
                'generated_at': datetime.now().isoformat(),
                'total_evaluations': len(evaluations),
                'summary_stats': {
                    'completed': len([e for e in evaluations if e['status'] == '已完成']),
                    'running': len([e for e in evaluations if e['status'] == '运行中']),
                    'failed': len([e for e in evaluations if e['status'] == '失败'])
                },
                'evaluations': evaluations
            }
            
            # 计算平均指标（如果有已完成的评估）
            completed_evals = [e for e in evaluations if e['status'] == '已完成' and e.get('results')]
            if completed_evals:
                avg_accuracy = sum(e['results'].get('original_accuracy', 0) for e in completed_evals) / len(completed_evals)
                avg_attack_success = sum(e['results'].get('attack_success_rate', 0) for e in completed_evals) / len(completed_evals)
                avg_robustness = sum(e['results'].get('robustness_score', 0) for e in completed_evals) / len(completed_evals)
                
                summary_data['average_metrics'] = {
                    'original_accuracy': avg_accuracy,
                    'attack_success_rate': avg_attack_success,
                    'robustness_score': avg_robustness
                }
            
            # 保存汇总报告
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_filename = f"summary_report_{timestamp}.json"
            summary_filepath = os.path.join(self.reports_dir, summary_filename)
            
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            return summary_filepath
            
        except Exception as e:
            st.error(f"生成汇总报告失败: {str(e)}")
            return None
    
    def get_completed_evaluations(self):
        """获取已完成的评估列表（为了兼容性）"""
        # 这个方法可能被其他地方调用，返回空列表或从其他地方获取数据
        return []