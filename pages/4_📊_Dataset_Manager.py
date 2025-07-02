import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from PIL import Image

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from core.dataset_manager import DatasetManager
from core.auth_manager import AuthManager

# 页面配置
st.set_page_config(
    page_title="数据集管理 - AI模型安全评估平台",
    page_icon="📊",
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
dataset_manager = DatasetManager()
auth_manager = AuthManager()

# 获取用户信息
username = st.session_state.username
user_info = auth_manager.get_user_info(username)
is_admin = user_info.get('role') == 'admin'

# 自定义CSS
st.markdown("""
<style>
.dataset-card {
    background-color: transparent;
    padding: 1rem 0;
    border: none;
    margin: 0.5rem 0;
    box-shadow: none;
}
.builtin-card {
    background-color: transparent;
    border-left: 3px solid #28a745;
    padding-left: 1rem;
    border: none;
    box-shadow: none;
}
.user-card {
    background-color: transparent;
    border-left: 3px solid #ffc107;
    padding-left: 1rem;
    border: none;
    box-shadow: none;
}
.upload-container {
    background-color: transparent;
    padding: 1.5rem;
    border-radius: 10px;
    border: 2px dashed rgba(0, 123, 255, 0.3);
    text-align: center;
    margin: 1rem 0;
}
.stats-card {
    background-color: transparent;
    padding: 0.8rem;
    border-left: 3px solid #2196f3;
    margin: 0.3rem 0;
    border: none;
    box-shadow: none;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("# 📊 数据集管理")
    
    # 侧边栏统计信息
    with st.sidebar:
        st.markdown("### 📈 数据集统计")
        stats = dataset_manager.get_storage_stats()
        
        st.metric("内置数据集", stats['total_builtin_datasets'])
        st.metric("用户数据集", stats['total_user_datasets'])
        st.metric("总存储大小", f"{stats['total_size'] / (1024*1024):.2f} MB")
        
        if stats['type_stats']:
            st.markdown("**按类型统计:**")
            for dataset_type, type_stat in stats['type_stats'].items():
                st.write(f"• {dataset_type}: {type_stat['count']}个")
    
    # 主要功能选项卡
    tab1, tab2, tab3, tab4 = st.tabs(["📤 上传数据集", "🏛️ 内置数据集", "📋 我的数据集", "🔍 数据集详情"])
    
    with tab1:
        upload_dataset_interface()
    
    with tab2:
        builtin_datasets_interface()
    
    with tab3:
        my_datasets_interface()
    
    with tab4:
        dataset_details_interface()

def upload_dataset_interface():
    """数据集上传界面"""
    st.markdown("## 📤 上传新数据集")
    
    # 支持的格式说明
    with st.expander("📋 支持的数据集格式", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **表格数据:**
            - CSV: `.csv`
            - JSON: `.json`
            - Parquet: `.parquet`
            
            **数组数据:**
            - NumPy: `.npy`, `.npz`
            - Pickle: `.pkl`, `.pickle`
            """)
        
        with col2:
            st.markdown("""
            **图像数据:**
            - JPEG: `.jpg`, `.jpeg`
            - PNG: `.png`
            - BMP: `.bmp`
            
            **文本数据:**
            - 文本文件: `.txt`
            """)
    
    # 上传表单
    with st.form("upload_dataset_form"):
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "选择数据集文件",
            type=['csv', 'json', 'parquet', 'npy', 'npz', 'pkl', 'pickle', 'jpg', 'jpeg', 'png', 'bmp', 'txt'],
            help="支持多种数据格式，系统会自动检测数据类型"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 数据集信息
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_name = st.text_input(
                "数据集名称 *",
                placeholder="请输入数据集名称",
                help="用于识别数据集的唯一名称"
            )
            
            # 自动检测数据集类型
            dataset_type = "自动检测"
            if uploaded_file:
                file_ext = Path(uploaded_file.name).suffix.lower()
                if file_ext in ['.csv', '.json', '.parquet']:
                    dataset_type = "表格数据"
                elif file_ext in ['.npy', '.npz', '.pkl', '.pickle']:
                    dataset_type = "数组数据"
                elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    dataset_type = "图像数据"
                elif file_ext == '.txt':
                    dataset_type = "文本数据"
            
            st.text_input("数据集类型", value=dataset_type, disabled=True)
        
        with col2:
            description = st.text_area(
                "数据集描述",
                placeholder="请描述数据集的内容、来源、用途等信息",
                height=100
            )
            
            # 显示文件信息
            if uploaded_file:
                st.info(f"📁 文件名: {uploaded_file.name}")
                st.info(f"📏 文件大小: {uploaded_file.size / (1024*1024):.2f} MB")
        
        # 上传按钮
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.form_submit_button("🚀 上传数据集", use_container_width=True):
                if not uploaded_file:
                    st.error("❌ 请选择要上传的数据集文件")
                elif not dataset_name:
                    st.error("❌ 请输入数据集名称")
                elif dataset_type == "自动检测":
                    st.error("❌ 无法识别数据集类型，请检查文件格式")
                else:
                    # 执行上传
                    with st.spinner("正在上传和验证数据集..."):
                        # 转换类型名称
                        type_mapping = {
                            "表格数据": "tabular",
                            "数组数据": "array",
                            "图像数据": "image",
                            "文本数据": "text"
                        }
                        actual_type = type_mapping.get(dataset_type, "auto")
                        
                        success, message, dataset_id = dataset_manager.save_uploaded_dataset(
                            uploaded_file, dataset_name, description, actual_type, username
                        )
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.balloons()
                        
                        # 更新会话状态
                        st.session_state.selected_dataset = dataset_id
                        
                        # 显示下一步提示
                        st.info("🎯 数据集上传成功！您现在可以配置攻击参数进行安全评估。")
                        
                        if st.button("⚔️ 前往攻击配置"):
                            st.switch_page("pages/5_⚔️_Attack_Config.py")
                    else:
                        st.error(f"❌ {message}")

def builtin_datasets_interface():
    """内置数据集界面"""
    st.markdown("## 🏛️ 内置数据集")
    
    builtin_datasets = dataset_manager.get_builtin_datasets()
    
    if not builtin_datasets:
        st.info("📭 暂无可用的内置数据集")
        return
    
    st.markdown(f"**共有 {len(builtin_datasets)} 个内置数据集可用**")
    
    for dataset_name, info in builtin_datasets.items():
        with st.container():
            st.markdown('<div class="dataset-card builtin-card">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.markdown(f"**🏛️ {info.get('name', dataset_name)}**")
                st.caption(f"📝 {info.get('description', '无描述')}")
                st.caption(f"🏷️ 类型: {info.get('type', 'Unknown')}")
            
            with col2:
                st.metric("类别数", info.get('classes', 'N/A'))
                st.caption(f"📊 样本数: {info.get('samples', 'N/A')}")
            
            with col3:
                input_shape = info.get('input_shape', 'N/A')
                if isinstance(input_shape, (list, tuple)):
                    shape_str = 'x'.join(map(str, input_shape))
                else:
                    shape_str = str(input_shape)
                st.metric("输入形状", shape_str)
                
                # 显示类别名称（如果有）
                if 'class_names' in info and len(info['class_names']) <= 5:
                    st.caption(f"🏷️ 类别: {', '.join(info['class_names'])}")
                elif 'class_names' in info:
                    st.caption(f"🏷️ 类别: {', '.join(info['class_names'][:3])}...")
            
            with col4:
                dataset_id = f"builtin_{dataset_name}"
                
                if st.button("📋 详情", key=f"detail_{dataset_id}"):
                    st.session_state.selected_dataset_id = dataset_id
                    st.rerun()
                
                if st.button("🎯 使用", key=f"use_{dataset_id}"):
                    st.session_state.selected_dataset = dataset_id
                    st.success(f"✅ 已选择数据集: {info.get('name')}")
                
                if st.button("👁️ 预览", key=f"preview_{dataset_id}"):
                    with st.spinner("正在加载数据集预览..."):
                        success, preview_data, message = dataset_manager.preview_dataset(dataset_id, max_samples=5)
                    
                    if success:
                        st.success(f"✅ {message}")
                        # 显示预览信息
                        with st.expander(f"📊 {info.get('name')} 预览", expanded=True):
                            st.json(preview_data)
                    else:
                        st.error(f"❌ {message}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

def my_datasets_interface():
    """我的数据集界面"""
    st.markdown("## 📋 我的数据集")
    
    # 获取用户数据集
    if is_admin:
        user_datasets = dataset_manager.load_datasets_info()
        st.info("👑 管理员模式：显示所有用户的数据集")
    else:
        user_datasets = dataset_manager.get_user_datasets(username)
    
    if not user_datasets:
        st.info("📭 您还没有上传任何数据集")
        return
    
    # 搜索和过滤
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 搜索数据集", placeholder="输入数据集名称或描述")
    
    with col2:
        dataset_types = list(set(info.get('dataset_type', 'Unknown') for info in user_datasets.values()))
        selected_type = st.selectbox("筛选类型", ['全部'] + dataset_types)
    
    with col3:
        sort_by = st.selectbox("排序方式", ['上传时间', '数据集名称', '文件大小'])
    
    # 过滤和排序数据集
    filtered_datasets = {}
    for dataset_id, info in user_datasets.items():
        # 搜索过滤
        if search_term:
            if (search_term.lower() not in info.get('dataset_name', '').lower() and 
                search_term.lower() not in info.get('description', '').lower()):
                continue
        
        # 类型过滤
        if selected_type != '全部' and info.get('dataset_type') != selected_type:
            continue
        
        filtered_datasets[dataset_id] = info
    
    # 排序
    if sort_by == '上传时间':
        sorted_datasets = sorted(filtered_datasets.items(), 
                               key=lambda x: x[1].get('upload_time', ''), reverse=True)
    elif sort_by == '数据集名称':
        sorted_datasets = sorted(filtered_datasets.items(), 
                               key=lambda x: x[1].get('dataset_name', ''))
    else:  # 文件大小
        sorted_datasets = sorted(filtered_datasets.items(), 
                               key=lambda x: x[1].get('file_size', 0), reverse=True)
    
    # 显示数据集列表
    st.markdown(f"**找到 {len(filtered_datasets)} 个数据集**")
    
    for dataset_id, info in sorted_datasets:
        with st.container():
            st.markdown('<div class="dataset-card user-card">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.markdown(f"**📊 {info.get('dataset_name', 'Unknown')}**")
                st.caption(f"📝 {info.get('description', '无描述')[:100]}..." if len(info.get('description', '')) > 100 else info.get('description', '无描述'))
                st.caption(f"👤 上传者: {info.get('uploaded_by', 'Unknown')}")
            
            with col2:
                st.metric("数据类型", info.get('dataset_type', 'Unknown'))
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
                if st.button("📋 详情", key=f"detail_{dataset_id}"):
                    st.session_state.selected_dataset_id = dataset_id
                    st.rerun()
                
                if st.button("🎯 使用", key=f"use_{dataset_id}"):
                    st.session_state.selected_dataset = dataset_id
                    st.success(f"✅ 已选择数据集: {info.get('dataset_name')}")
                
                # 删除按钮（仅限数据集所有者或管理员）
                if is_admin or info.get('uploaded_by') == username:
                    if st.button("🗑️ 删除", key=f"delete_{dataset_id}"):
                        success, message = dataset_manager.delete_dataset(dataset_id, username, is_admin)
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

def dataset_details_interface():
    """数据集详情界面"""
    st.markdown("## 🔍 数据集详情")
    
    # 检查是否选择了数据集
    selected_dataset_id = st.session_state.get('selected_dataset_id')
    
    if not selected_dataset_id:
        st.info("📋 请在其他选项卡中选择一个数据集查看详情")
        return
    
    # 获取数据集信息
    dataset_info = dataset_manager.get_dataset_info(selected_dataset_id)
    
    if not dataset_info:
        st.error("❌ 数据集不存在或已被删除")
        st.session_state.selected_dataset_id = None
        return
    
    # 显示数据集详情
    is_builtin = dataset_info.get('is_builtin', False)
    dataset_name = dataset_info.get('dataset_name') or dataset_info.get('name', 'Unknown')
    
    st.markdown(f"### {'🏛️' if is_builtin else '📊'} {dataset_name}")
    
    # 基本信息
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 基本信息")
        st.info(f"**数据集ID:** {selected_dataset_id}")
        st.info(f"**数据类型:** {dataset_info.get('dataset_type') or dataset_info.get('type', 'Unknown')}")
        
        if not is_builtin:
            st.info(f"**文件名:** {dataset_info.get('file_name', 'Unknown')}")
            st.info(f"**文件大小:** {dataset_info.get('file_size', 0) / (1024*1024):.2f} MB")
            st.info(f"**文件哈希:** {dataset_info.get('file_hash', 'Unknown')[:16]}...")
        else:
            st.info(f"**类别数:** {dataset_info.get('classes', 'N/A')}")
            st.info(f"**样本数:** {dataset_info.get('samples', 'N/A')}")
            input_shape = dataset_info.get('input_shape', 'N/A')
            if isinstance(input_shape, (list, tuple)):
                shape_str = 'x'.join(map(str, input_shape))
            else:
                shape_str = str(input_shape)
            st.info(f"**输入形状:** {shape_str}")
    
    with col2:
        st.markdown("#### 👤 数据集信息")
        if is_builtin:
            st.info(f"**类型:** 内置数据集")
            st.info(f"**维护者:** 系统")
            
            # 显示类别名称
            if 'class_names' in dataset_info:
                class_names = dataset_info['class_names']
                if len(class_names) <= 10:
                    st.info(f"**类别:** {', '.join(class_names)}")
                else:
                    st.info(f"**类别:** {', '.join(class_names[:5])}... (共{len(class_names)}个)")
        else:
            st.info(f"**上传者:** {dataset_info.get('uploaded_by', 'Unknown')}")
            st.info(f"**上传时间:** {dataset_info.get('upload_time', '')[:19]}")
            
            if dataset_info.get('last_modified'):
                st.info(f"**最后修改:** {dataset_info.get('last_modified', '')[:19]}")
                st.info(f"**修改者:** {dataset_info.get('modified_by', 'Unknown')}")
            
            # 验证状态
            status = dataset_info.get('validation_status', 'unknown')
            if status == 'valid':
                st.success(f"✅ 验证状态: {dataset_info.get('validation_message', '已验证')}")
            else:
                st.error(f"❌ 验证状态: {dataset_info.get('validation_message', '验证失败')}")
    
    # 数据集描述
    st.markdown("#### 📝 数据集描述")
    description = dataset_info.get('description', '无描述')
    st.text_area("描述内容", value=description, height=100, disabled=True)
    
    # 元数据信息
    if 'metadata' in dataset_info and dataset_info['metadata']:
        st.markdown("#### 📊 元数据信息")
        metadata = dataset_info['metadata']
        
        if isinstance(metadata, dict):
            # 以表格形式显示元数据
            metadata_df = pd.DataFrame(list(metadata.items()), columns=['属性', '值'])
            st.dataframe(metadata_df, use_container_width=True)
        else:
            st.json(metadata)
    
    # 数据预览
    st.markdown("#### 👁️ 数据预览")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🔄 加载预览", use_container_width=True):
            with st.spinner("正在加载数据预览..."):
                success, preview_data, message = dataset_manager.preview_dataset(selected_dataset_id, max_samples=10)
            
            if success:
                st.session_state.preview_data = preview_data
                st.session_state.preview_message = message
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
    
    with col2:
        # 显示预览数据
        if 'preview_data' in st.session_state:
            preview_data = st.session_state.preview_data
            
            if preview_data.get('type') == 'builtin':
                st.markdown("**内置数据集信息:**")
                st.json({
                    '训练样本': preview_data.get('train_samples'),
                    '测试样本': preview_data.get('test_samples'),
                    '数据集信息': preview_data.get('info')
                })
            
            elif preview_data.get('type') == 'user':
                dataset_type = preview_data.get('dataset_type')
                
                if dataset_type == 'tabular' and 'preview' in preview_data:
                    st.markdown("**表格数据预览:**")
                    st.dataframe(preview_data['preview'], use_container_width=True)
                
                elif dataset_type == 'array':
                    st.markdown("**数组数据信息:**")
                    st.json({
                        '形状': preview_data.get('shape'),
                        '数据类型': preview_data.get('dtype')
                    })
                
                elif dataset_type == 'image':
                    st.markdown("**图像数据:**")
                    try:
                        st.image(preview_data['data'], caption="数据集图像预览", width=300)
                    except:
                        st.info("无法显示图像预览")
                
                else:
                    st.markdown("**数据集信息:**")
                    st.json(preview_data.get('info', {}))
    
    # 编辑数据集信息（仅限用户数据集的所有者或管理员）
    if not is_builtin and (is_admin or dataset_info.get('uploaded_by') == username):
        st.markdown("---")
        st.markdown("#### ✏️ 编辑数据集信息")
        
        with st.form("edit_dataset_form"):
            new_name = st.text_input("数据集名称", value=dataset_info.get('dataset_name', ''))
            new_description = st.text_area("数据集描述", value=dataset_info.get('description', ''), height=100)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("💾 保存修改", use_container_width=True):
                    updates = {
                        'dataset_name': new_name,
                        'description': new_description
                    }
                    
                    success, message = dataset_manager.update_dataset_info(
                        selected_dataset_id, updates, username, is_admin
                    )
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
            
            with col2:
                if st.form_submit_button("🗑️ 删除数据集", use_container_width=True):
                    success, message = dataset_manager.delete_dataset(selected_dataset_id, username, is_admin)
                    if success:
                        st.success(f"✅ {message}")
                        st.session_state.selected_dataset_id = None
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
    
    # 操作按钮
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 选择此数据集进行评估", use_container_width=True):
            st.session_state.selected_dataset = selected_dataset_id
            st.success(f"✅ 已选择数据集: {dataset_name}")
    
    with col2:
        if st.button("⚔️ 前往攻击配置", use_container_width=True):
            st.switch_page("pages/5_⚔️_Attack_Config.py")
    
    with col3:
        if st.button("🔙 返回数据集列表", use_container_width=True):
            st.session_state.selected_dataset_id = None
            st.rerun()

if __name__ == "__main__":
    main()