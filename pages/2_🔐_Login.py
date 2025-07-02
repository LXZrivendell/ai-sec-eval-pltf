import streamlit as st
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from core.auth_manager import AuthManager

# 页面配置
st.set_page_config(
    page_title="用户登录 - AI模型安全评估平台",
    page_icon="🔐",
    layout="wide"
)

# 初始化认证管理器
auth_manager = AuthManager()

# 自定义CSS
st.markdown("""
<style>
.login-container {
    max-width: 400px;
    margin: 0 auto;
    padding: 2rem;
    border-radius: 10px;
    /* 移除背景色和阴影 */
    /* background-color: #f8f9fa; */
    /* box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); */
}
.login-header {
    text-align: center;
    color: #1f77b4;
    margin-bottom: 2rem;
}
.demo-info {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #2196f3;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="login-header">🔐 用户认证</h1>', unsafe_allow_html=True)
    
    # 检查是否已登录
    if st.session_state.get('logged_in', False):
        st.success(f"✅ 您已登录为: {st.session_state.username}")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🏠 返回首页", use_container_width=True):
                st.switch_page("app.py")
        
        with col2:
            if st.button("🔄 修改密码", use_container_width=True):
                st.session_state.show_change_password = True
                st.rerun()
        
        with col3:
            if st.button("🚪 退出登录", use_container_width=True):
                # 清除会话
                if 'session_id' in st.session_state:
                    auth_manager.logout_user(st.session_state.session_id)
                
                # 重置状态
                for key in ['logged_in', 'username', 'session_id', 'user_info']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.success("✅ 已成功退出登录")
                st.rerun()
        
        # 显示用户信息
        st.markdown("---")
        st.markdown("### 👤 用户信息")
        
        user_info = auth_manager.get_user_info(st.session_state.username)
        if user_info:
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**用户名:** {st.session_state.username}")
                st.info(f"**角色:** {user_info.get('role', 'user')}")
            with col2:
                st.info(f"**邮箱:** {user_info.get('email', 'N/A')}")
                if user_info.get('last_login'):
                    st.info(f"**上次登录:** {user_info['last_login'][:19]}")
        
        # 修改密码界面
        if st.session_state.get('show_change_password', False):
            st.markdown("---")
            st.markdown("### 🔄 修改密码")
            
            with st.form("change_password_form"):
                old_password = st.text_input("原密码", type="password")
                new_password = st.text_input("新密码", type="password")
                confirm_password = st.text_input("确认新密码", type="password")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("✅ 确认修改", use_container_width=True):
                        if new_password != confirm_password:
                            st.error("❌ 两次输入的新密码不一致")
                        else:
                            success, message = auth_manager.change_password(
                                st.session_state.username, old_password, new_password
                            )
                            if success:
                                st.success(f"✅ {message}")
                                st.session_state.show_change_password = False
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                
                with col2:
                    if st.form_submit_button("❌ 取消", use_container_width=True):
                        st.session_state.show_change_password = False
                        st.rerun()
        
        return
    
    # 登录/注册选项卡
    tab1, tab2 = st.tabs(["🔑 登录", "📝 注册"])
    
    with tab1:
        login_form()
    
    with tab2:
        register_form()
    
    # 演示账号信息
    st.markdown("---")
    st.markdown('<div class="demo-info">', unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 演示账号
    
    **管理员账号:**
    - 用户名: `admin`
    - 密码: `admin123`
    
    **普通用户账号:**
    - 用户名: `demo`
    - 密码: `demo123`
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def login_form():
    """登录表单"""
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    with st.form("login_form"):
        st.markdown("#### 🔑 用户登录")
        
        username = st.text_input("用户名", placeholder="请输入用户名")
        password = st.text_input("密码", type="password", placeholder="请输入密码")
        
        remember_me = st.checkbox("记住我")
        
        if st.form_submit_button("🚀 登录", use_container_width=True):
            if not username or not password:
                st.error("❌ 请填写完整的登录信息")
                return
            
            # 验证用户
            success, message = auth_manager.authenticate_user(username, password)
            
            if success:
                # 创建会话
                session_id = auth_manager.create_session(username)
                
                # 设置会话状态
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.session_id = session_id
                st.session_state.user_info = auth_manager.get_user_info(username)
                
                st.success(f"✅ {message}")
                st.balloons()
                
                # 延迟跳转到首页
                st.info("🔄 正在跳转到首页...")
                st.rerun()
            else:
                st.error(f"❌ {message}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def register_form():
    """注册表单"""
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    with st.form("register_form"):
        st.markdown("#### 📝 用户注册")
        
        username = st.text_input("用户名", placeholder="请输入用户名 (至少3个字符)")
        email = st.text_input("邮箱", placeholder="请输入邮箱地址")
        password = st.text_input("密码", type="password", placeholder="请输入密码 (至少6个字符)")
        confirm_password = st.text_input("确认密码", type="password", placeholder="请再次输入密码")
        
        agree_terms = st.checkbox("我同意用户协议和隐私政策")
        
        if st.form_submit_button("📝 注册", use_container_width=True):
            if not all([username, email, password, confirm_password]):
                st.error("❌ 请填写完整的注册信息")
                return
            
            if password != confirm_password:
                st.error("❌ 两次输入的密码不一致")
                return
            
            if not agree_terms:
                st.error("❌ 请同意用户协议和隐私政策")
                return
            
            # 注册用户
            success, message = auth_manager.register_user(username, password, email)
            
            if success:
                st.success(f"✅ {message}")
                st.info("🔄 请切换到登录选项卡进行登录")
                st.balloons()
            else:
                st.error(f"❌ {message}")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()