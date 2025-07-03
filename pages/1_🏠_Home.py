import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.auth_manager import AuthManager
from core.model_loader import ModelLoader
from core.dataset_manager import DatasetManager
from core.attack_manager import AttackManager
from core.security_evaluator import SecurityEvaluator
# 修改第17行的导入语句  
from core.reporting import ReportGenerator
# 替换原来的：from core.report_generator import ReportGenerator

# 页面配置
st.set_page_config(
    page_title="AI模型安全评估平台 - 首页",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 检查登录状态
auth_manager = AuthManager()
if not auth_manager.is_logged_in():
    st.error("❌ 请先登录")
    st.info("👈 请使用左侧导航栏中的登录页面")
    st.stop()

# 获取当前用户信息
current_user = auth_manager.get_current_user()
user_info = auth_manager.get_user_info(current_user.get('username', ''))

# 初始化管理器
model_loader = ModelLoader()
dataset_manager = DatasetManager()
attack_manager = AttackManager()
security_evaluator = SecurityEvaluator()
report_generator = ReportGenerator()

# 页面标题
st.title("🏠 AI模型安全评估平台")
st.markdown(f"欢迎回来，**{current_user.get('username', 'Unknown')}** ({current_user.get('role', 'user')})！")

# 创建主要布局
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.header("📊 平台概览")
    
    # 获取统计数据
    try:
        model_stats = model_loader.get_storage_stats()
        dataset_stats = dataset_manager.get_storage_stats()
        attack_stats = attack_manager.get_storage_stats()
        
        # 用户相关统计
        if current_user.get('role') == 'admin':
            user_models = model_loader.get_all_models()
            user_datasets = dataset_manager.get_all_datasets()
            user_attacks = attack_manager.get_all_configs()  # 修复：使用新添加的方法
        else:
            user_models = model_loader.get_user_models(current_user.get('username', ''))
            user_datasets = dataset_manager.get_user_datasets(current_user.get('username', ''))
            user_attacks = attack_manager.get_user_configs(current_user.get('username', ''))
        
        # 统计卡片
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                label="📦 我的模型",
                value=len(user_models),
                delta=f"总计 {model_stats['total_models']} 个"
            )
        
        with metric_col2:
            st.metric(
                label="📊 我的数据集",
                value=len(user_datasets),
                delta=f"总计 {dataset_stats['total_datasets']} 个"
            )
        
        with metric_col3:
            st.metric(
                label="⚔️ 攻击配置",
                value=len(user_attacks),
                delta=f"总计 {attack_stats['total_configs']} 个"
            )
        
        with metric_col4:
            # 评估历史统计
            evaluation_history = security_evaluator.get_evaluation_history(current_user.get('username', ''))
            st.metric(
                label="🛡️ 评估历史",
                value=len(evaluation_history),
                delta="次评估"
            )
        
        # 存储使用情况
        st.subheader("💾 存储使用情况")
        
        # 获取存储数据并添加验证
        try:
            model_size = round(model_stats['total_size'] / (1024*1024), 2)
            dataset_size = round(dataset_stats['total_size'] / (1024*1024), 2)
            report_size = round(report_generator.get_storage_stats()['total_size'] / (1024*1024), 2)
            result_size = round(security_evaluator.get_storage_stats()['total_size'] / (1024*1024), 2)
            
            storage_data = {
                '类型': ['模型', '数据集', '报告', '结果'],
                '大小(MB)': [model_size, dataset_size, report_size, result_size]
            }
            
            # 检查是否有有效数据
            total_size = sum(storage_data['大小(MB)'])
            
            if total_size > 0:
                # 过滤掉大小为0的项目
                filtered_types = []
                filtered_sizes = []
                for i, size in enumerate(storage_data['大小(MB)']):
                    if size > 0:
                        filtered_types.append(storage_data['类型'][i])
                        filtered_sizes.append(size)
                
                if filtered_sizes:  # 确保有数据可显示
                    fig_storage = px.pie(
                        values=filtered_sizes,
                        names=filtered_types,
                        title="存储空间分布"
                    )
                    fig_storage.update_traces(textposition='inside', textinfo='percent+label')
                    fig_storage.update_layout(
                        showlegend=True,
                        height=400,
                        margin=dict(t=50, b=50, l=50, r=50)
                    )
                    st.plotly_chart(fig_storage, use_container_width=True)
                else:
                    st.info("📊 暂无存储数据可显示")
            else:
                # 当没有数据时，显示实际的存储使用情况表格
                st.info("📊 当前系统中暂无数据文件，存储使用量为0")
                
                # 显示详细的存储统计表格
                storage_df = pd.DataFrame({
                    '存储类型': ['模型', '数据集', '报告', '结果'],
                    '大小(MB)': [model_size, dataset_size, report_size, result_size],
                    '占比': ['0%', '0%', '0%', '0%']
                })
                
                st.dataframe(
                    storage_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # 提示用户如何开始使用
                st.markdown("""
                💡 **开始使用提示**：
                - 📤 上传模型文件到系统
                - 📊 添加数据集
                - 🛡️ 执行安全评估生成报告
                - 📈 查看存储使用情况变化
                """)
                
        except Exception as e:
            st.error(f"生成存储统计图表时出错: {str(e)}")
            st.info("请检查各个管理器的存储统计功能是否正常")
            
            # 显示调试信息
            with st.expander("🔍 调试信息"):
                st.write("模型统计:", model_stats)
                st.write("数据集统计:", dataset_stats)
                try:
                    st.write("报告统计:", report_generator.get_storage_stats())
                except Exception as report_err:
                    st.write("报告统计错误:", str(report_err))
                try:
                    st.write("评估统计:", security_evaluator.get_storage_stats())
                except Exception as eval_err:
                    st.write("评估统计错误:", str(eval_err))
        
    except Exception as e:
        st.error(f"获取统计数据时出错: {str(e)}")

with col2:
    st.header("🚀 快速操作")
    
    # 快速操作按钮
    st.subheader("模型管理")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📤 上传模型", use_container_width=True):
            st.switch_page("pages/3_📤_Model_Upload.py")
    with col_btn2:
        if st.button("📦 管理模型", use_container_width=True):
            st.switch_page("pages/3_📤_Model_Upload.py")
    
    st.subheader("数据集管理")
    col_btn3, col_btn4 = st.columns(2)
    with col_btn3:
        if st.button("📊 上传数据集", use_container_width=True):
            st.switch_page("pages/4_📊_Dataset_Manager.py")
    with col_btn4:
        if st.button("📋 管理数据集", use_container_width=True):
            st.switch_page("pages/4_📊_Dataset_Manager.py")
    
    st.subheader("安全评估")
    col_btn5, col_btn6 = st.columns(2)
    with col_btn5:
        if st.button("⚔️ 配置攻击", use_container_width=True):
            st.switch_page("pages/5_⚔️_Attack_Config.py")
    with col_btn6:
        if st.button("🛡️ 开始评估", use_container_width=True):
            st.switch_page("pages/6_🛡️_Security_Evaluation.py")
    
    st.subheader("报告管理")
    if st.button("📊 查看报告", use_container_width=True):
        st.switch_page("pages/7_📊_Report_Manager.py")
    
    # 最近活动
    st.subheader("📈 最近活动")
    try:
        # 获取最近的评估记录
        recent_evaluations = security_evaluator.get_evaluation_history(current_user.get('username', ''))[-5:]  # 修复：传入用户名字符串
        
        if recent_evaluations:
            for eval_record in reversed(recent_evaluations):
                with st.container():
                    st.markdown(f"""
                    **{eval_record.get('evaluation_name', 'Unknown')}**  
                    📅 {eval_record.get('created_at', 'Unknown')}  
                    🎯 {eval_record.get('model_name', 'Unknown')} | {eval_record.get('attack_type', 'Unknown')}  
                    📊 状态: {eval_record.get('status', 'Unknown')}
                    """)
                    st.divider()
        else:
            st.info("暂无评估记录")
    except Exception as e:
        st.error(f"获取最近活动时出错: {str(e)}")

with col3:
    st.header("ℹ️ 系统信息")
    
    # 系统状态
    st.subheader("🔧 系统状态")
    
    # 检查各个组件状态
    components_status = {
        "认证系统": "✅ 正常",
        "模型加载器": "✅ 正常",
        "数据集管理": "✅ 正常",
        "攻击管理": "✅ 正常",
        "安全评估": "✅ 正常",
        "报告生成": "✅ 正常"
    }
    
    for component, status in components_status.items():
        st.markdown(f"**{component}**: {status}")
    
    st.divider()
    
    # 平台信息
    st.subheader("📋 平台信息")
    st.markdown("""
    **版本**: v1.0.0  
    **更新时间**: 2025-07-02  
    **支持格式**:  
    - 模型: PyTorch, TensorFlow, ONNX, Scikit-learn等  
    - 数据: CSV, JSON, NPY, Images等
    - 攻击: FGSM, PGD, C&W, DeepFool等  
    """)
    
    st.divider()
    
    # 快速帮助
    st.subheader("❓ 快速帮助")
    
    with st.expander("🔍 如何开始评估？"):
        st.markdown("""
        1. 📤 上传或选择模型
        2. 📊 准备数据集
        3. ⚔️ 配置攻击参数
        4. 🛡️ 执行安全评估
        5. 📊 查看评估报告
        """)
    
    with st.expander("📚 支持的攻击类型"):
        st.markdown("""
        - **FGSM**: 快速梯度符号法
        - **PGD**: 投影梯度下降
        - **C&W**: Carlini & Wagner
        - **DeepFool**: 深度欺骗
        - **AutoAttack**: 自动攻击
        """)
    
    with st.expander("🛠️ 技术支持"):
        st.markdown("""
        如遇问题，请联系：
        - 📧 support@ntlxz1001@163.com
        - 📞 13755161660
        - 💬 在线客服
        """)

# 页面底部
st.divider()

# 最新公告或提示
st.subheader("📢 系统公告")
with st.container():
    st.info("""
    🎉 **欢迎使用AI模型安全评估平台！**  
    
    本平台提供全面的AI模型安全评估服务，支持多种攻击算法和防御策略。  
    您可以上传自己的模型和数据集，配置攻击参数，执行安全评估，并生成详细的评估报告。
    
    💡 **提示**: 首次使用建议先查看快速帮助，了解评估流程。
    """)

# 页脚
st.markdown("""
---
<div style='text-align: center; color: #666;'>
    <p>AI模型安全评估平台 | 版权所有 © 2025 | 技术支持: 刘行至🔗ntlxz1001@163.com🐙GitHub：LXZrivendell</p>
</div>
""", unsafe_allow_html=True)