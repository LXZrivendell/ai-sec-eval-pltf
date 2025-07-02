import streamlit as st
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from core.model_downloader import ModelDownloader
from core.auth_manager import AuthManager

# 页面配置
st.set_page_config(
    page_title="模型下载 - AI模型安全评估平台",
    page_icon="📥",
    layout="wide"
)

# 初始化
auth_manager = AuthManager()
model_downloader = ModelDownloader()

# 检查登录状态
if not st.session_state.get('logged_in', False):
    st.error("❌ 请先登录")
    st.stop()

def main():
    st.markdown("# 📥 官方模型下载")
    
    # 侧边栏信息
    with st.sidebar:
        st.markdown("### 📊 下载统计")
        downloaded_models = model_downloader.list_downloaded_models()
        st.metric("已下载模型", len(downloaded_models))
        
        total_size = sum(info.get('total_size', 0) for info in downloaded_models.values())
        st.metric("总存储大小", f"{total_size / (1024*1024*1024):.2f} GB")
    
    # 主要选项卡
    tab1, tab2, tab3 = st.tabs(["🔽 下载模型", "📋 已下载模型", "🌟 热门模型"])
    
    with tab1:
        download_interface()
    
    with tab2:
        downloaded_models_interface()
    
    with tab3:
        popular_models_interface()

def download_interface():
    """模型下载界面"""
    st.markdown("## 🔽 下载新模型")
    
    # 选择下载源
    source = st.selectbox(
        "选择模型源",
        options=["huggingface", "tensorflow_hub", "pytorch_hub", "custom_url"],
        format_func=lambda x: {
            "huggingface": "🤗 Hugging Face Hub",
            "tensorflow_hub": "🔥 TensorFlow Hub", 
            "pytorch_hub": "⚡ PyTorch Hub",
            "custom_url": "🔗 自定义 URL"
        }[x]
    )
    
    if source == "huggingface":
        huggingface_download_form()
    elif source == "tensorflow_hub":
        tensorflow_hub_download_form()
    elif source == "pytorch_hub":
        pytorch_hub_download_form()
    elif source == "custom_url":
        custom_url_download_form()

def huggingface_download_form():
    """Hugging Face 下载表单"""
    st.markdown("### 🤗 Hugging Face Hub")
    
    with st.form("hf_download_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.text_input(
                "模型名称", 
                placeholder="例如: bert-base-uncased",
                help="输入 Hugging Face Hub 上的模型名称"
            )
            
        with col2:
            model_alias = st.text_input(
                "本地别名", 
                placeholder="例如: bert_base",
                help="为模型设置一个本地别名"
            )
        
        include_tokenizer = st.checkbox("包含 Tokenizer", value=True)
        
        if st.form_submit_button("🚀 开始下载", use_container_width=True):
            if not model_name:
                st.error("❌ 请输入模型名称")
                return
            
            with st.spinner(f"正在下载 {model_name}..."):
                success, message, model_id = model_downloader.download_huggingface_model(
                    model_name, model_alias, include_tokenizer
                )
                
                if success:
                    st.success(f"✅ {message}")
                    st.balloons()
                else:
                    st.error(f"❌ {message}")

def tensorflow_hub_download_form():
    """TensorFlow Hub 下载表单"""
    st.markdown("### 🔥 TensorFlow Hub")
    
    with st.form("tfhub_download_form"):
        model_url = st.text_input(
            "模型 URL", 
            placeholder="例如: https://tfhub.dev/google/universal-sentence-encoder/4",
            help="输入 TensorFlow Hub 模型的完整 URL"
        )
        
        model_alias = st.text_input(
            "本地别名", 
            placeholder="例如: universal_sentence_encoder",
            help="为模型设置一个本地别名"
        )
        
        if st.form_submit_button("🚀 开始下载", use_container_width=True):
            if not model_url or not model_alias:
                st.error("❌ 请填写完整信息")
                return
            
            with st.spinner(f"正在下载模型..."):
                success, message, model_id = model_downloader.download_tensorflow_hub_model(
                    model_url, model_alias
                )
                
                if success:
                    st.success(f"✅ {message}")
                    st.balloons()
                else:
                    st.error(f"❌ {message}")

def pytorch_hub_download_form():
    """PyTorch Hub 下载表单"""
    st.markdown("### ⚡ PyTorch Hub")
    
    with st.form("pytorch_download_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            repo = st.text_input(
                "仓库", 
                placeholder="例如: pytorch/vision:v0.10.0",
                help="输入 PyTorch Hub 仓库名称"
            )
            
        with col2:
            model_name = st.text_input(
                "模型名称", 
                placeholder="例如: resnet18",
                help="输入模型名称"
            )
        
        model_alias = st.text_input(
            "本地别名", 
            placeholder="例如: resnet18_pretrained"
        )
        
        pretrained = st.checkbox("使用预训练权重", value=True)
        
        if st.form_submit_button("🚀 开始下载", use_container_width=True):
            if not repo or not model_name:
                st.error("❌ 请填写完整信息")
                return
            
            # 检查torch库是否可用
            try:
                import torch
                st.info(f"✅ PyTorch 版本: {torch.__version__}")
            except ImportError:
                st.error("❌ PyTorch 库未安装，无法下载 PyTorch Hub 模型")
                return
            
            kwargs = {'pretrained': pretrained} if pretrained else {}
            
            # 显示下载参数
            st.info(f"📋 下载参数: repo={repo}, model={model_name}, kwargs={kwargs}")
            
            with st.spinner(f"正在下载 {repo}:{model_name}..."):
                try:
                    success, message, model_id = model_downloader.download_pytorch_hub_model(
                        repo, model_name, model_alias, **kwargs
                    )
                    
                    if success:
                        st.success(f"✅ {message}")
                        if model_id:
                            st.info(f"📋 模型ID: {model_id}")
                        st.balloons()
                    else:
                        st.error(f"❌ {message}")
                        
                except Exception as e:
                    st.error(f"❌ 下载过程中发生异常: {str(e)}")
                    st.exception(e)  # 显示详细的异常信息

def custom_url_download_form():
    """自定义 URL 下载表单"""
    st.markdown("### 🔗 自定义 URL")
    
    with st.form("custom_download_form"):
        url = st.text_input(
            "下载 URL", 
            placeholder="例如: https://example.com/model.zip",
            help="输入模型文件的下载链接"
        )
        
        model_alias = st.text_input(
            "本地别名", 
            placeholder="例如: custom_model"
        )
        
        extract = st.checkbox("自动解压", value=True, help="自动解压 zip/tar 文件")
        
        if st.form_submit_button("🚀 开始下载", use_container_width=True):
            if not url or not model_alias:
                st.error("❌ 请填写完整信息")
                return
            
            with st.spinner(f"正在下载模型..."):
                success, message, model_id = model_downloader.download_from_url(
                    url, model_alias, extract
                )
                
                if success:
                    st.success(f"✅ {message}")
                    st.balloons()
                else:
                    st.error(f"❌ {message}")

def downloaded_models_interface():
    """已下载模型界面"""
    st.markdown("## 📋 已下载的模型")
    
    downloaded_models = model_downloader.list_downloaded_models()
    
    if not downloaded_models:
        st.info("📭 还没有下载任何模型")
        return
    
    # 转换为 DataFrame
    models_data = []
    for model_id, info in downloaded_models.items():
        models_data.append({
            "ID": model_id,
            "名称": info.get('model_name', 'N/A'),
            "来源": info.get('source', 'N/A'),
            "类型": info.get('model_type', 'N/A'),
            "大小": f"{info.get('total_size', 0) / (1024*1024):.2f} MB",
            "下载时间": info.get('download_time', 'N/A')[:19] if info.get('download_time') else 'N/A'
        })
    
    df = pd.DataFrame(models_data)
    
    # 显示表格
    st.dataframe(df, use_container_width=True)
    
    # 删除模型功能
    st.markdown("### 🗑️ 删除模型")
    
    selected_model = st.selectbox(
        "选择要删除的模型",
        options=list(downloaded_models.keys()),
        format_func=lambda x: f"{downloaded_models[x].get('model_name', x)} ({x})"
    )
    
    if st.button("🗑️ 删除选中模型", type="secondary"):
        if st.session_state.get('confirm_delete', False):
            success, message = model_downloader.delete_downloaded_model(selected_model)
            if success:
                st.success(f"✅ {message}")
                st.rerun()
            else:
                st.error(f"❌ {message}")
            st.session_state.confirm_delete = False
        else:
            st.warning("⚠️ 确认删除？再次点击确认")
            st.session_state.confirm_delete = True

def popular_models_interface():
    """热门模型界面"""
    st.markdown("## 🌟 热门模型推荐")
    
    popular_models = model_downloader.get_popular_models()
    
    for source, models in popular_models.items():
        st.markdown(f"### {source.title()}")
        
        for model in models:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if source == "pytorch_hub":
                    # 修复：正确处理包含多个冒号的模型字符串
                    parts = model.split(":")
                    if len(parts) >= 3:
                        repo = f"{parts[0]}:{parts[1]}"  # repo:version
                        model_name = parts[2]  # model_name
                    else:
                        repo, model_name = model.split(":", 1)  # 向后兼容
                    st.markdown(f"**{model_name}** - `{repo}`")
                else:
                    st.markdown(f"**{model}**")
            
            with col2:
                if st.button(f"下载", key=f"download_{source}_{model}"):
                    # 设置下载参数并触发下载
                    if source == "huggingface":
                        with st.spinner(f"正在下载 {model}..."):
                            success, message, _ = model_downloader.download_huggingface_model(model)
                            if success:
                                st.success(f"✅ {message}")
                            else:
                                st.error(f"❌ {message}")
                    # 其他源的快速下载可以类似实现
        
        st.markdown("---")

if __name__ == "__main__":
    main()