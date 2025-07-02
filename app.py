import streamlit as st
import os
from pathlib import Path

# 设置页面配置
st.set_page_config(
    page_title="AI模型安全评估平台",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 创建必要的目录
def create_directories():
    directories = [
        "data/models",
        "data/datasets", 
        "data/results",
        "data/reports",
        "static/css",
        "static/images",
        "static/js"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# 初始化会话状态
def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'uploaded_model' not in st.session_state:
        st.session_state.uploaded_model = None
    if 'selected_dataset' not in st.session_state:
        st.session_state.selected_dataset = None
    if 'attack_config' not in st.session_state:
        st.session_state.attack_config = {}
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None

# 在main()函数开始处添加会话验证
def validate_session():
    """验证用户会话"""
    if 'session_id' in st.session_state and 'logged_in' in st.session_state:
        from core.auth_manager import AuthManager
        auth_manager = AuthManager()
        
        valid, username = auth_manager.validate_session(st.session_state.session_id)
        if not valid:
            # 会话无效，清除状态
            for key in ['logged_in', 'username', 'session_id', 'user_info']:
                if key in st.session_state:
                    del st.session_state[key]
            st.warning("⚠️ 会话已过期，请重新登录")
            return False
        
        # 更新用户名（防止不一致）
        st.session_state.username = username
        return True
    
    return False

def main():
    # 创建目录和初始化
    create_directories()
    init_session_state()
    
    # 验证会话
    validate_session()
    
    # 自定义CSS样式
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 1rem;
    }
    .sidebar-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 主标题
    st.markdown('<h1 class="main-header">🛡️ AI模型安全评估平台</h1>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.markdown('<h2 class="sidebar-header">📋 导航菜单</h2>', unsafe_allow_html=True)
        
        # 显示登录状态
        if st.session_state.logged_in:
            st.success(f"👤 欢迎, {st.session_state.username}!")
            if st.button("🚪 退出登录"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.rerun()
        else:
            st.warning("⚠️ 请先登录")
        
        st.markdown("---")
        
        # 显示当前状态
        st.markdown("### 📊 当前状态")
        
        # 模型状态
        if st.session_state.uploaded_model:
            st.markdown('<p class="status-success">✅ 模型已上传</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-warning">⏳ 未上传模型</p>', unsafe_allow_html=True)
        
        # 数据集状态
        if st.session_state.selected_dataset:
            st.markdown('<p class="status-success">✅ 数据集已选择</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-warning">⏳ 未选择数据集</p>', unsafe_allow_html=True)
        
        # 攻击配置状态
        if st.session_state.attack_config:
            st.markdown('<p class="status-success">✅ 攻击已配置</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-warning">⏳ 未配置攻击</p>', unsafe_allow_html=True)
        
        # 评估结果状态
        if st.session_state.evaluation_results:
            st.markdown('<p class="status-success">✅ 评估已完成</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-warning">⏳ 未进行评估</p>', unsafe_allow_html=True)
    
    # 主内容区域
    if not st.session_state.logged_in:
        st.info("👈 请使用左侧导航栏中的登录页面进行登录")
        st.markdown("""
        ## 🌟 平台功能介绍
        
        本平台提供以下核心功能：
        
        - 🔐 **用户认证**: 安全的用户登录和会话管理
        - 📤 **模型上传**: 支持PyTorch、TensorFlow、Keras等主流框架
        - 📊 **数据集管理**: 内置数据集和用户自定义数据集支持
        - ⚔️ **攻击配置**: 多种对抗攻击算法和参数配置
        - 📈 **安全评估**: 全面的模型安全性评估和可视化
        - 📋 **报告生成**: 专业的安全评估报告导出
        
        ### 🚀 开始使用
        
        1. 点击左侧导航中的"登录"页面进行用户认证
        2. 上传您的AI模型文件
        3. 选择或上传测试数据集
        4. 配置对抗攻击参数
        5. 执行安全性评估
        6. 查看结果并生成报告
        """)
    else:
        st.success(f"🎉 欢迎使用AI模型安全评估平台，{st.session_state.username}！")
        
        # 显示工作流程
        st.markdown("""
        ## 📋 评估流程
        
        请按照以下步骤完成模型安全评估：
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.uploaded_model:
                st.success("✅ 1. 模型上传")
            else:
                st.error("❌ 1. 模型上传")
            st.markdown("[📤 上传模型](Model_Upload)")
        
        with col2:
            if st.session_state.selected_dataset:
                st.success("✅ 2. 数据集选择")
            else:
                st.error("❌ 2. 数据集选择")
            st.markdown("[📊 管理数据集](Dataset_Manager)")
        
        with col3:
            if st.session_state.attack_config:
                st.success("✅ 3. 攻击配置")
            else:
                st.error("❌ 3. 攻击配置")
            st.markdown("[⚔️ 配置攻击](Attack_Config)")
        
        with col4:
            if st.session_state.evaluation_results:
                st.success("✅ 4. 评估完成")
            else:
                st.error("❌ 4. 评估完成")
            st.markdown("[📈 查看结果](Evaluation_Results)")
        
        # 快速操作面板
        st.markdown("---")
        st.markdown("## ⚡ 快速操作")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 开始新评估", use_container_width=True):
                # 重置状态
                st.session_state.uploaded_model = None
                st.session_state.selected_dataset = None
                st.session_state.attack_config = {}
                st.session_state.evaluation_results = None
                st.success("✅ 已重置，可以开始新的评估流程")
        
        with col2:
            if st.button("📊 查看历史结果", use_container_width=True):
                st.info("📂 历史结果功能开发中...")
        
        with col3:
            if st.button("📋 生成报告", use_container_width=True):
                if st.session_state.evaluation_results:
                    st.success("📄 报告生成功能请访问报告生成页面")
                else:
                    st.warning("⚠️ 请先完成模型评估")

if __name__ == "__main__":
    main()