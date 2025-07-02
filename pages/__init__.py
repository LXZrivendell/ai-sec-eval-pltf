# Pages模块初始化文件
"""
AI模型安全评估平台 - 页面模块

本模块包含Streamlit应用的各个页面：
- Home: 主页
- Login: 登录页面
- Model Upload: 模型上传
- Dataset Manager: 数据集管理
- Attack Config: 攻击配置
- Security Evaluation: 安全评估
- Report Manager: 报告管理
"""

# 页面配置
PAGE_CONFIG = {
    'page_title': 'AI模型安全评估平台',
    'page_icon': '🛡️',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# 页面路由映射
PAGE_ROUTES = {
    'home': '1_🏠_Home.py',
    'login': '2_🔐_Login.py', 
    'model_upload': '3_📤_Model_Upload.py',
    'dataset_manager': '4_📊_Dataset_Manager.py',
    'attack_config': '5_⚔️_Attack_Config.py',
    'security_evaluation': '6_🛡️_Security_Evaluation.py',
    'report_manager': '7_📊_Report_Manager.py'
}

# 页面元数据
PAGE_METADATA = {
    'home': {
        'title': '主页',
        'description': '平台概览和快速导航',
        'icon': '🏠'
    },
    'login': {
        'title': '用户登录',
        'description': '用户认证和会话管理',
        'icon': '🔐'
    },
    'model_upload': {
        'title': '模型上传',
        'description': '上传和管理AI模型',
        'icon': '📤'
    },
    'dataset_manager': {
        'title': '数据集管理',
        'description': '管理评估数据集',
        'icon': '📊'
    },
    'attack_config': {
        'title': '攻击配置',
        'description': '配置对抗攻击参数',
        'icon': '⚔️'
    },
    'security_evaluation': {
        'title': '安全评估',
        'description': '执行模型安全性评估',
        'icon': '🛡️'
    },
    'report_manager': {
        'title': '报告管理',
        'description': '查看和管理评估报告',
        'icon': '📊'
    }
}

# 导航菜单配置
NAVIGATION_MENU = [
    {'name': '主页', 'page': 'home', 'icon': '🏠'},
    {'name': '登录', 'page': 'login', 'icon': '🔐'},
    {'name': '模型上传', 'page': 'model_upload', 'icon': '📤'},
    {'name': '数据集管理', 'page': 'dataset_manager', 'icon': '📊'},
    {'name': '攻击配置', 'page': 'attack_config', 'icon': '⚔️'},
    {'name': '安全评估', 'page': 'security_evaluation', 'icon': '🛡️'},
    {'name': '报告管理', 'page': 'report_manager', 'icon': '📊'}
]