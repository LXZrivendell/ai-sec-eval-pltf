import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Optional
import streamlit as st
from jinja2 import Template
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import base64
from io import BytesIO

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        self.reports_dir = "reports"
        self.templates_dir = "templates"
        self._ensure_directories()
        
        # 报告模板
        self.html_template = self._get_html_template()
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)
    
    def _get_html_template(self) -> str:
        """获取HTML报告模板"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #007bff;
            margin: 0;
            font-size: 2.5em;
        }
        .header .subtitle {
            color: #666;
            font-size: 1.2em;
            margin-top: 10px;
        }
        .section {
            margin-bottom: 30px;
        }
        .section h2 {
            color: #333;
            border-left: 4px solid #007bff;
            padding-left: 15px;
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .table-container {
            overflow-x: auto;
            margin: 20px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        .recommendations {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #28a745;
        }
        .recommendations h3 {
            color: #28a745;
            margin-top: 0;
        }
        .recommendations ul {
            list-style-type: none;
            padding: 0;
        }
        .recommendations li {
            margin: 10px 0;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            border-left: 3px solid #28a745;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }
        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }
        .status-high { background-color: #dc3545; color: white; }
        .status-medium { background-color: #ffc107; color: black; }
        .status-low { background-color: #28a745; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ report_title }}</h1>
            <div class="subtitle">{{ evaluation_info.name }} - {{ evaluation_info.type }}</div>
            <div class="subtitle">生成时间: {{ generated_at }}</div>
        </div>
        
        <div class="section">
            <h2>📋 评估概览</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{{ results.original_accuracy }}%</div>
                    <div class="metric-label">原始准确率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ results.attack_success_rate }}%</div>
                    <div class="metric-label">平均攻击成功率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ results.robustness_score }}</div>
                    <div class="metric-label">鲁棒性得分</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ results.security_level }}</div>
                    <div class="metric-label">安全等级</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 评估配置</h2>
            <table>
                <tr><th>配置项</th><th>值</th></tr>
                <tr><td>模型名称</td><td>{{ evaluation_info.model_name }}</td></tr>
                <tr><td>模型框架</td><td>{{ evaluation_info.model_framework }}</td></tr>
                <tr><td>数据集</td><td>{{ evaluation_info.dataset_name }}</td></tr>
                <tr><td>样本数量</td><td>{{ evaluation_info.sample_size }}</td></tr>
                <tr><td>攻击算法数量</td><td>{{ evaluation_info.attack_count }}</td></tr>
                <tr><td>评估时间</td><td>{{ evaluation_info.evaluation_time }}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>⚔️ 攻击结果详情</h2>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>攻击算法</th>
                            <th>成功率</th>
                            <th>平均扰动</th>
                            <th>平均查询次数</th>
                            <th>平均时间(秒)</th>
                            <th>风险等级</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for attack in attack_results %}
                        <tr>
                            <td>{{ attack.name }}</td>
                            <td>{{ attack.success_rate }}%</td>
                            <td>{{ attack.avg_perturbation }}</td>
                            <td>{{ attack.avg_queries }}</td>
                            <td>{{ attack.avg_time }}</td>
                            <td><span class="status-badge status-{{ attack.risk_level }}">{{ attack.risk_level_text }}</span></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 可视化分析</h2>
            <div class="chart-container">
                {{ charts_html }}
            </div>
        </div>
        
        <div class="section">
            <h2>💡 安全建议</h2>
            <div class="recommendations">
                <h3>改进建议</h3>
                <ul>
                    {% for recommendation in recommendations %}
                    <li>{{ recommendation }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>本报告由 AI模型安全评估平台 自动生成</p>
            <p>生成时间: {{ generated_at }}</p>
        </div>
    </div>
</body>
</html>
        """
    
    def generate_report(self, evaluation_data: Dict, format_type: str = "html") -> Optional[str]:
        """生成评估报告"""
        try:
            if format_type == "html":
                return self._generate_html_report(evaluation_data)
            elif format_type == "json":
                return self._generate_json_report(evaluation_data)
            elif format_type == "pdf":
                return self._generate_pdf_report(evaluation_data)
            else:
                raise ValueError(f"不支持的报告格式: {format_type}")
        except Exception as e:
            st.error(f"生成报告失败: {str(e)}")
            return None
    
    def _generate_html_report(self, evaluation_data: Dict) -> str:
        """生成HTML报告"""
        # 准备模板数据
        template_data = self._prepare_template_data(evaluation_data)
        
        # 生成图表HTML
        charts_html = self._generate_charts_html(evaluation_data)
        template_data['charts_html'] = charts_html
        
        # 渲染模板
        template = Template(self.html_template)
        html_content = template.render(**template_data)
        
        # 保存文件
        filename = f"{evaluation_data['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(self.reports_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filepath
    
    def _generate_json_report(self, evaluation_data: Dict) -> str:
        """生成JSON报告"""
        # 准备报告数据
        report_data = {
            "report_info": {
                "title": f"{evaluation_data['name']} - 安全评估报告",
                "generated_at": datetime.now().isoformat(),
                "format": "json",
                "version": "1.0"
            },
            "evaluation_info": evaluation_data,
            "results": evaluation_data.get('results', {}),
            "recommendations": self._generate_recommendations(evaluation_data),
            "metadata": {
                "platform": "AI模型安全评估平台",
                "generator": "ReportGenerator v1.0"
            }
        }
        
        # 保存文件
        filename = f"{evaluation_data['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.reports_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def _generate_pdf_report(self, evaluation_data: Dict) -> str:
        """生成PDF报告"""
        try:
            # 首先生成HTML报告
            html_filepath = self._generate_html_report(evaluation_data)
            
            # 转换为PDF（需要安装wkhtmltopdf）
            import pdfkit
            
            pdf_filename = html_filepath.replace('.html', '.pdf')
            
            options = {
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': "UTF-8",
                'no-outline': None,
                'enable-local-file-access': None
            }
            
            pdfkit.from_file(html_filepath, pdf_filename, options=options)
            
            return pdf_filename
        
        except ImportError:
            st.warning("PDF生成需要安装 pdfkit 和 wkhtmltopdf")
            # 返回HTML文件作为替代
            return self._generate_html_report(evaluation_data)
        except Exception as e:
            st.error(f"PDF生成失败: {str(e)}")
            return self._generate_html_report(evaluation_data)
    
    def _prepare_template_data(self, evaluation_data: Dict) -> Dict:
        """准备模板数据"""
        results = evaluation_data.get('results', {})
        
        # 处理攻击结果
        attack_results = []
        if results.get('attack_results'):
            for attack_name, attack_result in results['attack_results'].items():
                success_rate = attack_result.get('success_rate', 0) * 100
                
                # 确定风险等级
                if success_rate >= 70:
                    risk_level = "high"
                    risk_level_text = "高风险"
                elif success_rate >= 30:
                    risk_level = "medium"
                    risk_level_text = "中风险"
                else:
                    risk_level = "low"
                    risk_level_text = "低风险"
                
                attack_results.append({
                    "name": attack_name,
                    "success_rate": f"{success_rate:.1f}",
                    "avg_perturbation": f"{attack_result.get('avg_perturbation', 0):.4f}",
                    "avg_queries": attack_result.get('avg_queries', 'N/A'),
                    "avg_time": f"{attack_result.get('avg_time', 0):.2f}",
                    "risk_level": risk_level,
                    "risk_level_text": risk_level_text
                })
        
        template_data = {
            "report_title": f"{evaluation_data['name']} - 安全评估报告",
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "evaluation_info": {
                "name": evaluation_data['name'],
                "type": evaluation_data['type'],
                "model_name": evaluation_data.get('config', {}).get('model', {}).get('name', 'N/A'),
                "model_framework": evaluation_data.get('config', {}).get('model', {}).get('framework', 'N/A'),
                "dataset_name": evaluation_data.get('config', {}).get('dataset', {}).get('name', 'N/A'),
                "sample_size": evaluation_data.get('config', {}).get('parameters', {}).get('sample_size', 'N/A'),
                "attack_count": len(evaluation_data.get('config', {}).get('attack_configs', [])),
                "evaluation_time": evaluation_data.get('completed_at', 'N/A')[:19] if evaluation_data.get('completed_at') else 'N/A'
            },
            "results": {
                "original_accuracy": f"{results.get('original_accuracy', 0) * 100:.1f}",
                "attack_success_rate": f"{results.get('attack_success_rate', 0) * 100:.1f}",
                "robustness_score": f"{results.get('robustness_score', 0):.2f}",
                "security_level": results.get('security_level', 'N/A')
            },
            "attack_results": attack_results,
            "recommendations": self._generate_recommendations(evaluation_data)
        }
        
        return template_data
    
    def _generate_charts_html(self, evaluation_data: Dict) -> str:
        """生成图表HTML"""
        results = evaluation_data.get('results', {})
        charts_html = ""
        
        if results.get('attack_results'):
            # 攻击成功率对比图
            attack_names = list(results['attack_results'].keys())
            success_rates = [results['attack_results'][name].get('success_rate', 0) * 100 
                           for name in attack_names]
            
            fig1 = px.bar(
                x=attack_names,
                y=success_rates,
                title="攻击成功率对比",
                labels={'x': '攻击算法', 'y': '成功率 (%)'}
            )
            fig1.update_layout(showlegend=False)
            
            # 扰动大小对比图
            perturbations = [results['attack_results'][name].get('avg_perturbation', 0) 
                           for name in attack_names]
            
            fig2 = px.bar(
                x=attack_names,
                y=perturbations,
                title="平均扰动大小对比",
                labels={'x': '攻击算法', 'y': '平均扰动'}
            )
            fig2.update_layout(showlegend=False)
            
            # 转换为HTML
            chart1_html = plot(fig1, output_type='div', include_plotlyjs=True)
            chart2_html = plot(fig2, output_type='div', include_plotlyjs=False)
            
            charts_html = chart1_html + chart2_html
        
        return charts_html
    
    def _generate_recommendations(self, evaluation_data: Dict) -> List[str]:
        """生成安全建议"""
        results = evaluation_data.get('results', {})
        recommendations = []
        
        # 基于攻击成功率的建议
        attack_success_rate = results.get('attack_success_rate', 0)
        if attack_success_rate > 0.7:
            recommendations.append("模型对对抗攻击的鲁棒性较差，强烈建议进行对抗训练以提高安全性")
        elif attack_success_rate > 0.3:
            recommendations.append("模型存在一定的安全风险，建议使用数据增强和正则化技术提高鲁棒性")
        else:
            recommendations.append("模型具有较好的鲁棒性，建议继续保持并定期评估")
        
        # 基于鲁棒性得分的建议
        robustness_score = results.get('robustness_score', 0)
        if robustness_score < 0.5:
            recommendations.append("鲁棒性得分较低，建议重新设计模型架构或训练策略")
        elif robustness_score < 0.7:
            recommendations.append("鲁棒性有待提升，建议采用集成学习或防御蒸馏技术")
        
        # 基于具体攻击结果的建议
        if results.get('attack_results'):
            high_risk_attacks = []
            for attack_name, attack_result in results['attack_results'].items():
                if attack_result.get('success_rate', 0) > 0.8:
                    high_risk_attacks.append(attack_name)
            
            if high_risk_attacks:
                recommendations.append(f"对以下攻击算法特别脆弱：{', '.join(high_risk_attacks)}，建议针对性加强防护")
        
        # 通用建议
        recommendations.extend([
            "建立持续的安全监控机制，定期进行安全评估",
            "考虑部署攻击检测和防护系统",
            "对输入数据进行预处理和异常检测",
            "建立安全事件响应流程和应急预案"
        ])
        
        return recommendations
    
    def get_report_list(self, user_id: Optional[str] = None) -> List[Dict]:
        """获取报告列表"""
        reports = []
        
        try:
            for filename in os.listdir(self.reports_dir):
                if filename.endswith(('.html', '.json', '.pdf')):
                    filepath = os.path.join(self.reports_dir, filename)
                    file_stat = os.stat(filepath)
                    
                    report_info = {
                        "filename": filename,
                        "filepath": filepath,
                        "size": file_stat.st_size,
                        "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                        "format": filename.split('.')[-1].upper()
                    }
                    
                    # 从文件名提取信息
                    name_parts = filename.split('_')
                    if len(name_parts) >= 2:
                        report_info["evaluation_name"] = '_'.join(name_parts[:-2])
                    
                    reports.append(report_info)
        
        except Exception as e:
            st.error(f"获取报告列表失败: {str(e)}")
        
        return sorted(reports, key=lambda x: x['created_at'], reverse=True)
    
    def delete_report(self, filepath: str) -> bool:
        """删除报告"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
            return False
        except Exception as e:
            st.error(f"删除报告失败: {str(e)}")
            return False
    
    def get_storage_stats(self) -> Dict:
        """获取存储统计"""
        stats = {
            "total_reports": 0,
            "total_size": 0,
            "by_format": {}
        }
        
        try:
            for filename in os.listdir(self.reports_dir):
                if filename.endswith(('.html', '.json', '.pdf')):
                    filepath = os.path.join(self.reports_dir, filename)
                    file_size = os.path.getsize(filepath)
                    file_format = filename.split('.')[-1].upper()
                    
                    stats["total_reports"] += 1
                    stats["total_size"] += file_size
                    
                    if file_format not in stats["by_format"]:
                        stats["by_format"][file_format] = {"count": 0, "size": 0}
                    
                    stats["by_format"][file_format]["count"] += 1
                    stats["by_format"][file_format]["size"] += file_size
        
        except Exception as e:
            st.error(f"获取存储统计失败: {str(e)}")
        
        return stats
    
    def generate_summary_report(self, evaluations: List[Dict]) -> str:
        """生成汇总报告"""
        try:
            # 统计数据
            total_evaluations = len(evaluations)
            completed_evaluations = len([e for e in evaluations if e['status'] == '已完成'])
            
            # 按类型统计
            type_stats = {}
            for evaluation in evaluations:
                eval_type = evaluation['type']
                type_stats[eval_type] = type_stats.get(eval_type, 0) + 1
            
            # 安全性统计
            security_levels = []
            for evaluation in evaluations:
                if evaluation.get('results', {}).get('security_level'):
                    security_levels.append(evaluation['results']['security_level'])
            
            # 生成汇总报告
            summary_data = {
                "name": "平台汇总报告",
                "type": "汇总分析",
                "results": {
                    "total_evaluations": total_evaluations,
                    "completed_evaluations": completed_evaluations,
                    "completion_rate": completed_evaluations / total_evaluations if total_evaluations > 0 else 0,
                    "type_distribution": type_stats,
                    "security_distribution": {level: security_levels.count(level) for level in set(security_levels)}
                },
                "generated_at": datetime.now().isoformat()
            }
            
            return self._generate_json_report(summary_data)
        
        except Exception as e:
            st.error(f"生成汇总报告失败: {str(e)}")
            return None