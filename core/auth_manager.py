import streamlit as st
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import uuid

class AuthManager:
    """用户认证管理器"""
    
    def __init__(self):
        self.users_file = Path("data/users.json")
        self.sessions_file = Path("data/sessions.json")
        self.ensure_data_files()
    
    def ensure_data_files(self):
        """确保数据文件存在"""
        # 创建data目录
        Path("data").mkdir(exist_ok=True)
        
        # 初始化用户文件
        if not self.users_file.exists():
            default_users = {
                "admin": {
                    "password_hash": self.hash_password("admin123"),
                    "email": "admin@example.com",
                    "role": "admin",
                    "created_at": datetime.now().isoformat(),
                    "last_login": None
                },
                "demo": {
                    "password_hash": self.hash_password("demo123"),
                    "email": "demo@example.com",
                    "role": "user",
                    "created_at": datetime.now().isoformat(),
                    "last_login": None
                }
            }
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(default_users, f, ensure_ascii=False, indent=2)
        
        # 初始化会话文件
        if not self.sessions_file.exists():
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
    
    def hash_password(self, password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def load_users(self) -> dict:
        """加载用户数据"""
        try:
            with open(self.users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def save_users(self, users: dict):
        """保存用户数据"""
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    
    def load_sessions(self) -> dict:
        """加载会话数据"""
        try:
            with open(self.sessions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def save_sessions(self, sessions: dict):
        """保存会话数据"""
        with open(self.sessions_file, 'w', encoding='utf-8') as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
    
    def register_user(self, username: str, password: str, email: str) -> tuple[bool, str]:
        """用户注册"""
        users = self.load_users()
        
        # 检查用户名是否已存在
        if username in users:
            return False, "用户名已存在"
        
        # 检查邮箱是否已存在
        for user_data in users.values():
            if user_data.get('email') == email:
                return False, "邮箱已被注册"
        
        # 验证输入
        if len(username) < 3:
            return False, "用户名至少需要3个字符"
        
        if len(password) < 6:
            return False, "密码至少需要6个字符"
        
        if '@' not in email:
            return False, "请输入有效的邮箱地址"
        
        # 创建新用户
        users[username] = {
            "password_hash": self.hash_password(password),
            "email": email,
            "role": "user",
            "created_at": datetime.now().isoformat(),
            "last_login": None
        }
        
        self.save_users(users)
        return True, "注册成功"
    
    def authenticate_user(self, username: str, password: str) -> tuple[bool, str]:
        """用户认证"""
        users = self.load_users()
        
        if username not in users:
            return False, "用户名不存在"
        
        user_data = users[username]
        password_hash = self.hash_password(password)
        
        if user_data['password_hash'] != password_hash:
            return False, "密码错误"
        
        # 更新最后登录时间
        users[username]['last_login'] = datetime.now().isoformat()
        self.save_users(users)
        
        return True, "登录成功"
    
    def create_session(self, username: str) -> str:
        """创建用户会话"""
        session_id = str(uuid.uuid4())
        sessions = self.load_sessions()
        
        sessions[session_id] = {
            "username": username,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        self.save_sessions(sessions)
        return session_id
    
    def validate_session(self, session_id: str) -> tuple[bool, str]:
        """验证会话"""
        sessions = self.load_sessions()
        
        if session_id not in sessions:
            return False, None
        
        session_data = sessions[session_id]
        expires_at = datetime.fromisoformat(session_data['expires_at'])
        
        if datetime.now() > expires_at:
            # 会话已过期，删除
            del sessions[session_id]
            self.save_sessions(sessions)
            return False, None
        
        return True, session_data['username']
    
    def logout_user(self, session_id: str):
        """用户登出"""
        sessions = self.load_sessions()
        if session_id in sessions:
            del sessions[session_id]
            self.save_sessions(sessions)
    
    def get_user_info(self, username: str) -> dict:
        """获取用户信息"""
        users = self.load_users()
        if username in users:
            user_data = users[username].copy()
            # 不返回密码哈希
            user_data.pop('password_hash', None)
            return user_data
        return {}
    
    def change_password(self, username: str, old_password: str, new_password: str) -> tuple[bool, str]:
        """修改密码"""
        users = self.load_users()
        
        if username not in users:
            return False, "用户不存在"
        
        # 验证旧密码
        if users[username]['password_hash'] != self.hash_password(old_password):
            return False, "原密码错误"
        
        # 验证新密码
        if len(new_password) < 6:
            return False, "新密码至少需要6个字符"
        
        # 更新密码
        users[username]['password_hash'] = self.hash_password(new_password)
        self.save_users(users)
        
        return True, "密码修改成功"
    
    def is_logged_in(self) -> bool:
        """检查用户是否已登录"""
        return st.session_state.get('logged_in', False)
    
    def get_current_user(self) -> dict:
        """获取当前登录用户信息"""
        if not self.is_logged_in():
            return {}
        
        username = st.session_state.get('username', '')
        user_info = self.get_user_info(username)
        
        return {
            'user_id': username,
            'username': username,
            'role': user_info.get('role', 'user'),
            'email': user_info.get('email', ''),
            'session_id': st.session_state.get('session_id', '')
        }