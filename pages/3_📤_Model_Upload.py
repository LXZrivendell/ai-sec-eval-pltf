import streamlit as st
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from core.model_loader import ModelLoader
from core.auth_manager import AuthManager

# 页面配置
st.set_page_config(
    page_title="模型上传 - AI模型安全评估平台",
    page_icon="📤",
    layout="wide"
)

# 检查登录状态
if not st.session_state.get('logged_in', False):
    st.error("❌ 请先登录后再访问此页面")
    st.info("👈 请使用左侧导航栏中的登录页面进行登录")
    if st.button("🔐 前往登录页面"):
        st.switch_page("pages/2_🔐_Login.py")
    st.stop()

# 初始化管理器
model_loader = ModelLoader()
auth_manager = AuthManager()

# 获取用户信息
username = st.session_state.username
user_info = auth_manager.get_user_info(username)
is_admin = user_info.get('role') == 'admin'

# 自定义CSS
st.markdown("""
<style>
.model-card {
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 8px;
    border: 1px solid #dee2e6;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.stats-card {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #2196f3;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("# 📤 模型上传管理")
    
    # 侧边栏统计信息
    with st.sidebar:
        st.markdown("### 📊 存储统计")
        stats = model_loader.get_storage_stats()
        
        st.metric("总模型数", stats['total_models'])
        st.metric("总存储大小", f"{stats['total_size'] / (1024*1024):.2f} MB")
        
        if stats['type_stats']:
            st.markdown("**按类型统计:**")
            for model_type, type_stat in stats['type_stats'].items():
                st.write(f"• {model_type}: {type_stat['count']}个")
    
    # 主要功能选项卡
    tab1, tab2, tab3 = st.tabs(["📤 上传模型", "📋 我的模型", "🔍 模型详情"])
    
    with tab1:
        upload_model_interface()
    
    with tab2:
        my_models_interface()
    
    with tab3:
        model_details_interface()

def upload_model_interface():
    """模型上传界面"""
    st.markdown("## 📤 上传新模型")
    
    # 支持的格式说明
    with st.expander("📋 支持的模型格式", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **深度学习框架:**
            - PyTorch: `.pth`, `.pt`
            - TensorFlow: `.pb`, `.h5`
            - Keras: `.h5`, `.keras`
            - ONNX: `.onnx`
            """)
        
        with col2:
            st.markdown("""
            **机器学习框架:**
            - Scikit-learn: `.pkl`, `.pickle`, `.joblib`
            - 其他: 支持pickle序列化的模型
            """)
    
    # 上传表单
    with st.form("upload_model_form"):
        # 文件上传 - 移除了upload-container包装
        uploaded_file = st.file_uploader(
            "选择模型文件",
            type=['pth', 'pt', 'h5', 'keras', 'pb', 'onnx', 'pkl', 'pickle', 'joblib'],
            help="支持PyTorch、TensorFlow、Keras、ONNX、Scikit-learn等格式"
        )
        
        # 模型信息
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.text_input(
                "模型名称 *",
                placeholder="请输入模型名称",
                help="用于识别模型的唯一名称"
            )
            
            # 自动检测模型类型
            model_type = "自动检测"
            if uploaded_file:
                file_ext = Path(uploaded_file.name).suffix.lower()
                if file_ext in ['.pth', '.pt']:
                    model_type = "PyTorch"
                elif file_ext in ['.h5', '.keras']:
                    model_type = "Keras/TensorFlow"
                elif file_ext == '.pb':
                    model_type = "TensorFlow"
                elif file_ext == '.onnx':
                    model_type = "ONNX"
                elif file_ext in ['.pkl', '.pickle', '.joblib']:
                    model_type = "Scikit-learn"
            
            st.text_input("模型类型", value=model_type, disabled=True)
        
        with col2:
            description = st.text_area(
                "模型描述",
                placeholder="请描述模型的用途、架构、训练数据等信息",
                height=100
            )
            
            # 显示文件信息
            if uploaded_file:
                st.info(f"📁 文件名: {uploaded_file.name}")
                st.info(f"📏 文件大小: {uploaded_file.size / (1024*1024):.2f} MB")
        
        # 上传按钮
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.form_submit_button("🚀 上传模型", use_container_width=True):
                if not uploaded_file:
                    st.error("❌ 请选择要上传的模型文件")
                elif not model_name:
                    st.error("❌ 请输入模型名称")
                elif model_type == "自动检测":
                    st.error("❌ 无法识别模型类型，请检查文件格式")
                else:
                    # 执行上传
                    with st.spinner("正在上传和验证模型..."):
                        success, message, model_id = model_loader.save_uploaded_model(
                            uploaded_file, model_name, description, model_type, username
                        )
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.balloons()
                        
                        # 更新会话状态
                        st.session_state.uploaded_model = model_id
                        
                        # 显示下一步提示
                        st.info("🎯 模型上传成功！您现在可以选择数据集进行安全评估。")
                        
                        if st.button("📊 前往数据集管理"):
                            st.switch_page("pages/4_📊_Dataset_Manager.py")
                    else:
                        st.error(f"❌ {message}")

def my_models_interface():
    """我的模型界面"""
    st.markdown("## 📋 我的模型")
    
    # 获取用户模型
    if is_admin:
        user_models = model_loader.get_all_models()
        st.info("👑 管理员模式：显示所有用户的模型")
    else:
        user_models = model_loader.get_user_models(username)
    
    if not user_models:
        st.info("📭 您还没有上传任何模型")
        return
    
    # 搜索和过滤
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 搜索模型", placeholder="输入模型名称或描述")
    
    with col2:
        model_types = list(set(info.get('model_type', 'Unknown') for info in user_models.values()))
        selected_type = st.selectbox("筛选类型", ['全部'] + model_types)
    
    with col3:
        sort_by = st.selectbox("排序方式", ['上传时间', '模型名称', '文件大小'])
    
    # 过滤和排序模型
    filtered_models = {}
    for model_id, info in user_models.items():
        # 搜索过滤
        if search_term:
            if (search_term.lower() not in info.get('model_name', '').lower() and 
                search_term.lower() not in info.get('description', '').lower()):
                continue
        
        # 类型过滤
        if selected_type != '全部' and info.get('model_type') != selected_type:
            continue
        
        filtered_models[model_id] = info
    
    # 排序
    if sort_by == '上传时间':
        sorted_models = sorted(filtered_models.items(), 
                             key=lambda x: x[1].get('upload_time', ''), reverse=True)
    elif sort_by == '模型名称':
        sorted_models = sorted(filtered_models.items(), 
                             key=lambda x: x[1].get('model_name', ''))
    else:  # 文件大小
        sorted_models = sorted(filtered_models.items(), 
                             key=lambda x: x[1].get('file_size', 0), reverse=True)
    
    # 显示模型列表
    st.markdown(f"**找到 {len(filtered_models)} 个模型**")
    
    for model_id, info in sorted_models:
        with st.container():
            # 移除了 model-card 样式包装
            
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.markdown(f"**🤖 {info.get('model_name', 'Unknown')}**")
                st.caption(f"📝 {info.get('description', '无描述')[:100]}..." if len(info.get('description', '')) > 100 else info.get('description', '无描述'))
                st.caption(f"👤 上传者: {info.get('uploaded_by', 'Unknown')}")
            
            with col2:
                st.metric("模型类型", info.get('model_type', 'Unknown'))
                st.caption(f"📅 {info.get('upload_time', '')[:19]}")
            
            with col3:
                file_size_mb = info.get('file_size', 0) / (1024 * 1024)
                st.metric("文件大小", f"{file_size_mb:.2f} MB")
                
                # 验证状态
                status = info.get('validation_status', 'unknown')
                if status == 'valid':
                    st.success("✅ 已验证")
                else:
                    st.error("❌ 验证失败")
            
            with col4:
                # 操作按钮
                if st.button("📋 详情", key=f"detail_{model_id}"):
                    st.session_state.selected_model_id = model_id
                    st.rerun()
                
                if st.button("🎯 使用", key=f"use_{model_id}"):
                    st.session_state.uploaded_model = model_id
                    st.success(f"✅ 已选择模型: {info.get('model_name')}")
                
                # 删除按钮（仅限模型所有者或管理员）
                if is_admin or info.get('uploaded_by') == username:
                    if st.button("🗑️ 删除", key=f"delete_{model_id}"):
                        success, message = model_loader.delete_model(model_id, username, is_admin)
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
            
            # 移除了 model-card 结束标签
            st.markdown("---")

def model_details_interface():
    """模型详情界面"""
    st.markdown("## 🔍 模型详情")
    
    # 检查是否选择了模型
    selected_model_id = st.session_state.get('selected_model_id')
    
    if not selected_model_id:
        st.info("📋 请在'我的模型'选项卡中选择一个模型查看详情")
        return
    
    # 获取模型信息
    model_info = model_loader.get_model_info(selected_model_id)
    
    if not model_info:
        st.error("❌ 模型不存在或已被删除")
        st.session_state.selected_model_id = None
        return
    
    # 显示模型详情
    st.markdown(f"### 🤖 {model_info.get('model_name', 'Unknown')}")
    
    # 基本信息
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 基本信息")
        st.info(f"**模型ID:** {selected_model_id}")
        st.info(f"**模型类型:** {model_info.get('model_type', 'Unknown')}")
        st.info(f"**文件名:** {model_info.get('file_name', 'Unknown')}")
        st.info(f"**文件大小:** {model_info.get('file_size', 0) / (1024*1024):.2f} MB")
        st.info(f"**文件哈希:** {model_info.get('file_hash', 'Unknown')[:16]}...")
    
    with col2:
        st.markdown("#### 👤 上传信息")
        st.info(f"**上传者:** {model_info.get('uploaded_by', 'Unknown')}")
        st.info(f"**上传时间:** {model_info.get('upload_time', '')[:19]}")
        
        if model_info.get('last_modified'):
            st.info(f"**最后修改:** {model_info.get('last_modified', '')[:19]}")
            st.info(f"**修改者:** {model_info.get('modified_by', 'Unknown')}")
        
        # 验证状态
        status = model_info.get('validation_status', 'unknown')
        if status == 'valid':
            st.success(f"✅ 验证状态: {model_info.get('validation_message', '已验证')}")
        else:
            st.error(f"❌ 验证状态: {model_info.get('validation_message', '验证失败')}")
    
    # 模型描述
    st.markdown("#### 📝 模型描述")
    description = model_info.get('description', '无描述')
    st.text_area("描述内容", value=description, height=100, disabled=True)
    
    # 编辑模型信息（仅限所有者或管理员）
    if is_admin or model_info.get('uploaded_by') == username:
        st.markdown("---")
        st.markdown("#### ✏️ 编辑模型信息")
        
        with st.form("edit_model_form"):
            new_name = st.text_input("模型名称", value=model_info.get('model_name', ''))
            new_description = st.text_area("模型描述", value=model_info.get('description', ''), height=100)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("💾 保存修改", use_container_width=True):
                    updates = {
                        'model_name': new_name,
                        'description': new_description
                    }
                    
                    success, message = model_loader.update_model_info(
                        selected_model_id, updates, username, is_admin
                    )
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
            
            with col2:
                if st.form_submit_button("🗑️ 删除模型", use_container_width=True):
                    success, message = model_loader.delete_model(selected_model_id, username, is_admin)
                    if success:
                        st.success(f"✅ {message}")
                        st.session_state.selected_model_id = None
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
    
    # 操作按钮
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 选择此模型进行评估", use_container_width=True):
            st.session_state.uploaded_model = selected_model_id
            st.success(f"✅ 已选择模型: {model_info.get('model_name')}")
    
    with col2:
        if st.button("📊 前往数据集管理", use_container_width=True):
            st.switch_page("pages/4_📊_Dataset_Manager.py")
    
    with col3:
        if st.button("🔙 返回模型列表", use_container_width=True):
            st.session_state.selected_model_id = None
            st.rerun()

if __name__ == "__main__":
    main()