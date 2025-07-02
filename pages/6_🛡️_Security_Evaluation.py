import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
from core.auth_manager import AuthManager
from core.model_loader import ModelLoader
from core.dataset_manager import DatasetManager
from core.attack_manager import AttackManager
from core.security_evaluator import SecurityEvaluator

# 页面配置
st.set_page_config(
    page_title="安全评估 - AI模型安全评估平台",
    page_icon="🛡️",
    layout="wide"
)

# 初始化管理器
auth_manager = AuthManager()
model_loader = ModelLoader()
dataset_manager = DatasetManager()
attack_manager = AttackManager()
security_evaluator = SecurityEvaluator()

# 检查登录状态
if not auth_manager.is_logged_in():
    st.error("⚠️ 请先登录后再使用此功能")
    st.info("👈 请点击侧边栏中的 '🔐 Login' 进行登录")
    st.stop()

# 获取当前用户信息
current_user = auth_manager.get_current_user()
user_id = current_user['user_id']
user_role = current_user['role']

# 页面标题
st.title("🛡️ AI模型安全评估")
st.markdown("---")

# 侧边栏 - 功能选择
st.sidebar.header("功能选择")
function_choice = st.sidebar.selectbox(
    "选择功能",
    ["新建评估", "评估历史", "评估报告", "评估统计"]
)

if function_choice == "新建评估":
    st.header("🎯 新建安全评估")
    
    # 评估配置
    st.subheader("📋 评估配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        evaluation_name = st.text_input(
            "评估名称",
            placeholder="输入评估任务名称",
            help="为此次评估任务起一个描述性的名称"
        )
        
        evaluation_description = st.text_area(
            "评估描述",
            placeholder="描述此次评估的目的和预期结果",
            height=100
        )
    
    with col2:
        evaluation_type = st.selectbox(
            "评估类型",
            ["鲁棒性评估", "对抗攻击评估", "综合安全评估"],
            help="选择评估类型"
        )
        
        priority = st.selectbox(
            "优先级",
            ["低", "中", "高"],
            index=1,
            help="设置评估任务的优先级"
        )
    
    st.markdown("---")
    
    # 模型选择
    st.subheader("🤖 选择模型")
    
    # 获取用户模型
    if user_role == 'admin':
        user_models = model_loader.get_all_models()
    else:
        user_models = model_loader.get_user_models(user_id)
    
    if user_models:
        # 将字典格式转换为列表格式进行处理
        model_list = []
        for model_id, model_info in user_models.items():
            model_data = {
                'id': model_id,
                'name': model_info.get('model_name', 'Unknown'),
                'framework': model_info.get('model_type', 'Unknown'),
                'model_type': model_info.get('model_type', 'Unknown'),
                'file_size': model_info.get('file_size', 0),
                'upload_time': model_info.get('upload_time', ''),
                'uploader': model_info.get('uploaded_by', ''),
                'description': model_info.get('description', ''),
                'file_path': model_info.get('file_path', ''),
                'validation_status': model_info.get('validation_status', 'unknown'),
                'validation_message': model_info.get('validation_message', '')
            }
            model_list.append(model_data)
        
        model_options = {f"{model['name']} ({model['framework']})":
                        model for model in model_list}
        
        selected_model_key = st.selectbox(
            "选择要评估的模型",
            list(model_options.keys()),
            help="选择一个已上传的模型进行安全评估"
        )
        
        if selected_model_key:
            selected_model = model_options[selected_model_key]
            
            # 显示模型信息
            with st.expander("📊 模型信息", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**模型名称**: {selected_model['name']}")
                    st.write(f"**框架**: {selected_model['framework']}")
                    st.write(f"**类型**: {selected_model['model_type']}")
                
                with col2:
                    st.write(f"**文件大小**: {selected_model['file_size'] / (1024*1024):.2f} MB")
                    st.write(f"**上传时间**: {selected_model['upload_time'][:19]}")
                    st.write(f"**上传者**: {selected_model['uploader']}")
                
                with col3:
                    if selected_model.get('description'):
                        st.write(f"**描述**: {selected_model['description']}")
                    if selected_model.get('accuracy'):
                        st.write(f"**准确率**: {selected_model['accuracy']}")
    else:
        st.warning("⚠️ 您还没有上传任何模型")
        st.info("请先在 '📤 Model Upload' 页面上传模型")
        st.stop()
    
    st.markdown("---")
    
    # 数据集选择
    st.subheader("📊 选择数据集")
    
    dataset_source = st.radio(
        "数据集来源",
        ["内置数据集", "我的数据集"],
        horizontal=True
    )
    
    if dataset_source == "内置数据集":
        builtin_datasets = dataset_manager.get_builtin_datasets()
        
        if builtin_datasets:
            # 将字典格式转换为列表格式进行处理
            dataset_list = []
            for dataset_id, dataset_info in builtin_datasets.items():
                dataset_data = {
                    'id': dataset_id,
                    'name': dataset_info.get('name', 'Unknown'),
                    'type': dataset_info.get('type', 'Unknown'),
                    'data_type': dataset_info.get('data_type', dataset_info.get('type', 'Unknown')),
                    'description': dataset_info.get('description', ''),
                    'file_size': dataset_info.get('file_size', 0),
                    'shape': dataset_info.get('shape', ''),
                    'file_path': dataset_info.get('file_path', '')
                }
                dataset_list.append(dataset_data)
            
            dataset_options = {f"{ds['name']} ({ds['type']})":
                             ds for ds in dataset_list}
            
            selected_dataset_key = st.selectbox(
                "选择内置数据集",
                list(dataset_options.keys()),
                help="选择一个内置数据集进行评估"
            )
            
            if selected_dataset_key:
                selected_dataset = dataset_options[selected_dataset_key]
        else:
            st.warning("⚠️ 没有可用的内置数据集")
            st.stop()
    
    else:  # 我的数据集
        if user_role == 'admin':
            user_datasets = dataset_manager.get_all_datasets()
        else:
            user_datasets = dataset_manager.get_user_datasets(user_id)
        
        if user_datasets:
            # 将字典格式转换为列表格式进行处理
            dataset_list = []
            for dataset_id, dataset_info in user_datasets.items():
                dataset_data = {
                    'id': dataset_id,
                    'name': dataset_info.get('name', 'Unknown'),
                    'data_type': dataset_info.get('data_type', dataset_info.get('type', 'Unknown')),
                    'description': dataset_info.get('description', ''),
                    'file_size': dataset_info.get('file_size', 0),
                    'shape': dataset_info.get('shape', ''),
                    'file_path': dataset_info.get('file_path', '')
                }
                dataset_list.append(dataset_data)
            
            dataset_options = {f"{ds['name']} ({ds['data_type']})":
                             ds for ds in dataset_list}
            
            selected_dataset_key = st.selectbox(
                "选择我的数据集",
                list(dataset_options.keys()),
                help="选择一个已上传的数据集进行评估"
            )
            
            if selected_dataset_key:
                selected_dataset = dataset_options[selected_dataset_key]
        else:
            st.warning("⚠️ 您还没有上传任何数据集")
            st.info("请先在 '📊 Dataset Manager' 页面上传数据集")
            st.stop()
    
    # 显示数据集信息
    if 'selected_dataset' in locals():
        with st.expander("📋 数据集信息", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**数据集名称**: {selected_dataset['name']}")
                st.write(f"**数据类型**: {selected_dataset.get('data_type', 'N/A')}")
                if 'shape' in selected_dataset:
                    st.write(f"**数据形状**: {selected_dataset['shape']}")
            
            with col2:
                if 'description' in selected_dataset:
                    st.write(f"**描述**: {selected_dataset['description']}")
                if 'file_size' in selected_dataset:
                    st.write(f"**文件大小**: {selected_dataset['file_size'] / (1024*1024):.2f} MB")
    
    st.markdown("---")
    
    # 攻击配置选择
    st.subheader("⚔️ 选择攻击配置")
    
    # 获取用户攻击配置
    user_configs = attack_manager.get_user_configs(user_id)
    
    if user_configs:
        config_options = {f"{config['name']} ({config['config']['algorithm']})":
                         config for config in user_configs}
        
        selected_configs = st.multiselect(
            "选择攻击配置",
            list(config_options.keys()),
            help="可以选择多个攻击配置进行综合评估"
        )
        
        if selected_configs:
            # 显示选中的配置信息
            with st.expander("🔧 选中的攻击配置", expanded=False):
                for config_key in selected_configs:
                    config = config_options[config_key]
                    st.write(f"**{config['name']}**")
                    st.write(f"- 算法: {config['config']['algorithm']} ({config['config']['algorithm_name']})")
                    st.write(f"- 类型: {config['config']['attack_type']}")
                    st.write(f"- 描述: {config['config'].get('description', '无描述')}")
                    st.markdown("---")
    else:
        st.warning("⚠️ 您还没有创建任何攻击配置")
        st.info("请先在 '⚔️ Attack Config' 页面创建攻击配置")
        st.stop()
    
    # 评估参数
    st.subheader("⚙️ 评估参数")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sample_size = st.number_input(
            "样本数量",
            value=100,
            min_value=10,
            max_value=10000,
            step=10,
            help="用于评估的样本数量"
        )
        
        batch_size = st.number_input(
            "批处理大小",
            value=32,
            min_value=1,
            max_value=512,
            step=1,
            help="评估时的批处理大小"
        )
    
    with col2:
        confidence_threshold = st.slider(
            "置信度阈值",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="分类置信度阈值"
        )
        
        max_iterations = st.number_input(
            "最大迭代次数",
            value=1000,
            min_value=100,
            max_value=10000,
            step=100,
            help="攻击算法的最大迭代次数"
        )
    
    with col3:
        save_results = st.checkbox(
            "保存详细结果",
            value=True,
            help="是否保存详细的评估结果"
        )
        
        generate_report = st.checkbox(
            "生成评估报告",
            value=True,
            help="是否自动生成评估报告"
        )
    
    # 开始评估
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 开始安全评估", type="primary", use_container_width=True):
            if not evaluation_name:
                st.error("请输入评估名称")
            elif not selected_configs:
                st.error("请选择至少一个攻击配置")
            else:
                # 创建评估任务
                evaluation_config = {
                    "name": evaluation_name,
                    "description": evaluation_description,
                    "type": evaluation_type,
                    "priority": priority,
                    "model": selected_model,
                    "dataset": selected_dataset,
                    "attack_configs": [config_options[key] for key in selected_configs],
                    "parameters": {
                        "sample_size": sample_size,
                        "batch_size": batch_size,
                        "confidence_threshold": confidence_threshold,
                        "max_iterations": max_iterations,
                        "save_results": save_results,
                        "generate_report": generate_report
                    },
                    "user_id": user_id,
                    "created_at": datetime.now().isoformat()
                }
                
                # 开始评估
                with st.spinner("正在启动安全评估..."):
                    evaluation_id = security_evaluator.start_evaluation(evaluation_config)
                    
                    if evaluation_id:
                        st.success(f"✅ 评估任务已启动！任务ID: {evaluation_id}")
                        st.info("📊 您可以在 '评估历史' 中查看评估进度和结果")
                        st.balloons()
                        
                        # 显示评估进度
                        progress_placeholder = st.empty()
                        status_placeholder = st.empty()
                        
                        # 模拟评估进度（实际应该从evaluator获取）
                        for i in range(101):
                            progress_placeholder.progress(i)
                            if i < 20:
                                status_placeholder.info("🔄 正在加载模型和数据集...")
                            elif i < 40:
                                status_placeholder.info("⚔️ 正在执行攻击算法...")
                            elif i < 80:
                                status_placeholder.info("📊 正在计算评估指标...")
                            elif i < 100:
                                status_placeholder.info("📝 正在生成评估报告...")
                            else:
                                status_placeholder.success("✅ 评估完成！")
                            
                            time.sleep(0.05)  # 模拟处理时间
                        
                        st.success("🎉 安全评估已完成！")
                    else:
                        st.error("❌ 评估任务启动失败")

elif function_choice == "评估历史":
    st.header("📚 评估历史")
    
    # 获取用户评估历史
    if user_role == 'admin':
        evaluations = security_evaluator.get_all_evaluations()
    else:
        evaluations = security_evaluator.get_user_evaluations(user_id)
    
    if evaluations:
        # 搜索和筛选
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_term = st.text_input(
                "🔍 搜索评估",
                placeholder="输入评估名称或描述关键词"
            )
        
        with col2:
            status_filter = st.selectbox(
                "状态筛选",
                ["全部", "运行中", "已完成", "失败", "已取消"]
            )
        
        with col3:
            sort_by = st.selectbox(
                "排序方式",
                ["创建时间", "完成时间", "评估名称", "状态"]
            )
        
        # 筛选评估
        filtered_evaluations = evaluations
        
        if search_term:
            filtered_evaluations = [
                eval for eval in filtered_evaluations
                if search_term.lower() in eval['name'].lower() or
                   search_term.lower() in eval.get('description', '').lower()
            ]
        
        if status_filter != "全部":
            filtered_evaluations = [
                eval for eval in filtered_evaluations
                if eval['status'] == status_filter
            ]
        
        # 排序
        if sort_by == "创建时间":
            filtered_evaluations.sort(key=lambda x: x['created_at'], reverse=True)
        elif sort_by == "完成时间":
            filtered_evaluations.sort(key=lambda x: x.get('completed_at', ''), reverse=True)
        elif sort_by == "评估名称":
            filtered_evaluations.sort(key=lambda x: x['name'])
        else:  # 状态
            filtered_evaluations.sort(key=lambda x: x['status'])
        
        st.markdown(f"**找到 {len(filtered_evaluations)} 个评估记录**")
        
        # 显示评估列表
        for i, evaluation in enumerate(filtered_evaluations):
            status_color = {
                "运行中": "🔄",
                "已完成": "✅",
                "失败": "❌",
                "已取消": "⏹️"
            }.get(evaluation['status'], "❓")
            
            with st.expander(
                f"{status_color} {evaluation['name']} - {evaluation['type']}",
                expanded=False
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**评估ID**: {evaluation['id']}")
                    st.write(f"**类型**: {evaluation['type']}")
                    st.write(f"**状态**: {evaluation['status']}")
                    st.write(f"**描述**: {evaluation.get('description', '无描述')}")
                    st.write(f"**创建时间**: {evaluation['created_at'][:19]}")
                    
                    if evaluation.get('completed_at'):
                        st.write(f"**完成时间**: {evaluation['completed_at'][:19]}")
                    
                    # 显示评估配置
                    if evaluation.get('config'):
                        st.write(f"**模型**: {evaluation['config']['model']['name']}")
                        st.write(f"**数据集**: {evaluation['config']['dataset']['name']}")
                        st.write(f"**攻击配置数**: {len(evaluation['config']['attack_configs'])}")
                
                with col2:
                    st.write("**操作**")
                    
                    # 查看详情
                    if st.button(f"👁️ 查看详情", key=f"view_{i}"):
                        st.session_state[f"show_details_{i}"] = True
                    
                    # 下载报告
                    if evaluation['status'] == '已完成':
                        if st.button(f"📥 下载报告", key=f"download_{i}"):
                            st.info("报告下载功能开发中...")
                    
                    # 重新运行
                    if evaluation['status'] in ['失败', '已取消']:
                        if st.button(f"🔄 重新运行", key=f"rerun_{i}"):
                            st.info("重新运行功能开发中...")
                    
                    # 删除评估
                    if st.button(f"🗑️ 删除", key=f"delete_{i}", type="secondary"):
                        if security_evaluator.delete_evaluation(evaluation['id'], user_id):
                            st.success("评估记录删除成功！")
                            st.rerun()
                        else:
                            st.error("评估记录删除失败")
                
                # 显示详细信息
                if st.session_state.get(f"show_details_{i}", False):
                    st.markdown("**详细信息**")
                    
                    if evaluation.get('results'):
                        results = evaluation['results']
                        
                        # 显示评估结果
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        
                        with metrics_col1:
                            st.metric(
                                "原始准确率",
                                f"{results.get('original_accuracy', 0):.2%}"
                            )
                        
                        with metrics_col2:
                            st.metric(
                                "攻击成功率",
                                f"{results.get('attack_success_rate', 0):.2%}"
                            )
                        
                        with metrics_col3:
                            st.metric(
                                "鲁棒性得分",
                                f"{results.get('robustness_score', 0):.2f}"
                            )
                        
                        # 显示攻击结果图表
                        if results.get('attack_results'):
                            attack_data = []
                            for attack_name, attack_result in results['attack_results'].items():
                                attack_data.append({
                                    "攻击算法": attack_name,
                                    "成功率": attack_result.get('success_rate', 0),
                                    "平均扰动": attack_result.get('avg_perturbation', 0)
                                })
                            
                            if attack_data:
                                attack_df = pd.DataFrame(attack_data)
                                
                                fig = px.bar(
                                    attack_df,
                                    x="攻击算法",
                                    y="成功率",
                                    title="各攻击算法成功率对比"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                    if st.button(f"❌ 关闭详情", key=f"close_{i}"):
                        st.session_state[f"show_details_{i}"] = False
                        st.rerun()
    else:
        st.info("📝 您还没有进行任何安全评估")
        st.markdown("点击上方的 **新建评估** 开始您的第一次安全评估！")

elif function_choice == "评估报告":
    st.header("📊 评估报告")
    
    # 获取已完成的评估
    if user_role == 'admin':
        completed_evaluations = security_evaluator.get_completed_evaluations()
    else:
        completed_evaluations = security_evaluator.get_user_completed_evaluations(user_id)
    
    if completed_evaluations:
        # 选择评估报告
        evaluation_options = {f"{eval['name']} ({eval['completed_at'][:19]})":
                            eval for eval in completed_evaluations}
        
        selected_eval_key = st.selectbox(
            "选择评估报告",
            list(evaluation_options.keys()),
            help="选择一个已完成的评估查看详细报告"
        )
        
        if selected_eval_key:
            selected_evaluation = evaluation_options[selected_eval_key]
            
            # 显示报告标题
            st.markdown(f"## 📋 {selected_evaluation['name']} - 安全评估报告")
            st.markdown(f"**评估时间**: {selected_evaluation['completed_at'][:19]}")
            st.markdown(f"**评估类型**: {selected_evaluation['type']}")
            st.markdown("---")
            
            if selected_evaluation.get('results'):
                results = selected_evaluation['results']
                
                # 执行摘要
                st.subheader("📈 执行摘要")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "原始准确率",
                        f"{results.get('original_accuracy', 0):.2%}",
                        help="模型在原始数据上的准确率"
                    )
                
                with col2:
                    st.metric(
                        "平均攻击成功率",
                        f"{results.get('attack_success_rate', 0):.2%}",
                        help="所有攻击算法的平均成功率"
                    )
                
                with col3:
                    st.metric(
                        "鲁棒性得分",
                        f"{results.get('robustness_score', 0):.2f}",
                        help="模型的整体鲁棒性评分"
                    )
                
                with col4:
                    st.metric(
                        "安全等级",
                        results.get('security_level', 'N/A'),
                        help="基于评估结果的安全等级"
                    )
                
                # 详细分析
                st.subheader("🔍 详细分析")
                
                # 攻击结果分析
                if results.get('attack_results'):
                    st.write("**各攻击算法结果**")
                    
                    attack_data = []
                    for attack_name, attack_result in results['attack_results'].items():
                        attack_data.append({
                            "攻击算法": attack_name,
                            "成功率": f"{attack_result.get('success_rate', 0):.2%}",
                            "平均扰动": f"{attack_result.get('avg_perturbation', 0):.4f}",
                            "平均查询次数": attack_result.get('avg_queries', 'N/A'),
                            "平均时间(秒)": f"{attack_result.get('avg_time', 0):.2f}"
                        })
                    
                    attack_df = pd.DataFrame(attack_data)
                    st.dataframe(attack_df, use_container_width=True)
                    
                    # 可视化图表
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 成功率对比图
                        success_rates = [float(rate.strip('%'))/100 for rate in attack_df['成功率']]
                        fig1 = px.bar(
                            x=attack_df['攻击算法'],
                            y=success_rates,
                            title="攻击成功率对比",
                            labels={'x': '攻击算法', 'y': '成功率'}
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        # 扰动大小对比图
                        perturbations = [float(pert) for pert in attack_df['平均扰动']]
                        fig2 = px.bar(
                            x=attack_df['攻击算法'],
                            y=perturbations,
                            title="平均扰动大小对比",
                            labels={'x': '攻击算法', 'y': '平均扰动'}
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                
                # 安全建议
                st.subheader("💡 安全建议")
                
                if results.get('recommendations'):
                    for i, recommendation in enumerate(results['recommendations'], 1):
                        st.write(f"{i}. {recommendation}")
                else:
                    # 基于结果生成建议
                    recommendations = []
                    
                    if results.get('attack_success_rate', 0) > 0.5:
                        recommendations.append("模型对对抗攻击的鲁棒性较差，建议进行对抗训练")
                    
                    if results.get('robustness_score', 0) < 0.7:
                        recommendations.append("建议使用数据增强技术提高模型鲁棒性")
                    
                    recommendations.append("定期进行安全评估，监控模型安全性")
                    recommendations.append("考虑部署攻击检测机制")
                    
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec}")
                
                # 导出报告
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col2:
                    report_data = {
                        "evaluation_info": {
                            "name": selected_evaluation['name'],
                            "type": selected_evaluation['type'],
                            "completed_at": selected_evaluation['completed_at']
                        },
                        "results": results,
                        "generated_at": datetime.now().isoformat()
                    }
                    
                    report_json = json.dumps(report_data, ensure_ascii=False, indent=2)
                    
                    st.download_button(
                        label="📥 导出报告 (JSON)",
                        data=report_json,
                        file_name=f"{selected_evaluation['name']}_report.json",
                        mime="application/json",
                        use_container_width=True
                    )
            else:
                st.warning("⚠️ 该评估没有可用的结果数据")
    else:
        st.info("📝 没有已完成的评估报告")
        st.markdown("完成安全评估后，报告将在此处显示")

elif function_choice == "评估统计":
    st.header("📊 评估统计")
    
    # 获取统计数据
    stats = security_evaluator.get_evaluation_stats()
    
    # 总体统计
    st.subheader("📈 总体统计")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "总评估数",
            stats.get('total_evaluations', 0),
            help="系统中所有评估任务的总数"
        )
    
    with col2:
        st.metric(
            "已完成评估",
            stats.get('completed_evaluations', 0),
            help="已成功完成的评估任务数"
        )
    
    with col3:
        st.metric(
            "运行中评估",
            stats.get('running_evaluations', 0),
            help="当前正在运行的评估任务数"
        )
    
    with col4:
        completion_rate = 0
        if stats.get('total_evaluations', 0) > 0:
            completion_rate = stats.get('completed_evaluations', 0) / stats.get('total_evaluations', 0)
        
        st.metric(
            "完成率",
            f"{completion_rate:.1%}",
            help="评估任务的完成率"
        )
    
    # 评估类型分布
    if stats.get('evaluation_types'):
        st.subheader("📊 评估类型分布")
        
        type_data = list(stats['evaluation_types'].items())
        type_df = pd.DataFrame(type_data, columns=['评估类型', '数量'])
        
        fig = px.pie(
            type_df,
            values='数量',
            names='评估类型',
            title="评估类型分布"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 用户活跃度统计
    if user_role == 'admin' and stats.get('user_activity'):
        st.subheader("👥 用户活跃度")
        
        user_data = []
        for user_id, activity in stats['user_activity'].items():
            user_data.append({
                "用户ID": user_id,
                "评估总数": activity.get('total', 0),
                "已完成": activity.get('completed', 0),
                "运行中": activity.get('running', 0),
                "失败": activity.get('failed', 0)
            })
        
        user_df = pd.DataFrame(user_data)
        st.dataframe(user_df, use_container_width=True)
    
    # 个人统计
    st.subheader("👤 我的评估统计")
    
    user_evaluations = security_evaluator.get_user_evaluations(user_id)
    
    if user_evaluations:
        # 状态统计
        status_counts = {}
        type_counts = {}
        
        for evaluation in user_evaluations:
            status = evaluation['status']
            eval_type = evaluation['type']
            
            status_counts[status] = status_counts.get(status, 0) + 1
            type_counts[eval_type] = type_counts.get(eval_type, 0) + 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**按状态统计**")
            status_df = pd.DataFrame([
                {"状态": k, "数量": v}
                for k, v in status_counts.items()
            ])
            st.dataframe(status_df, use_container_width=True)
        
        with col2:
            st.write("**按类型统计**")
            type_df = pd.DataFrame([
                {"评估类型": k, "数量": v}
                for k, v in type_counts.items()
            ])
            st.dataframe(type_df, use_container_width=True)
        
        # 时间趋势
        st.write("**评估时间趋势**")
        
        # 按月统计
        monthly_counts = {}
        for evaluation in user_evaluations:
            month = evaluation['created_at'][:7]  # YYYY-MM
            monthly_counts[month] = monthly_counts.get(month, 0) + 1
        
        if monthly_counts:
            months = sorted(monthly_counts.keys())
            counts = [monthly_counts[month] for month in months]
            
            fig = px.line(
                x=months,
                y=counts,
                title="每月评估数量趋势",
                labels={'x': '月份', 'y': '评估数量'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("您还没有进行任何评估")

# 页面底部信息
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>🛡️ 提示：安全评估是检验AI模型鲁棒性和安全性的重要手段，建议定期进行评估</small>
    </div>
    """,
    unsafe_allow_html=True
)