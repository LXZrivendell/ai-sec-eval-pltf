import streamlit as st
import pandas as pd
from datetime import datetime
import json
from core.auth_manager import AuthManager
from core.attack_manager import AttackManager
from core.model_loader import ModelLoader

# 页面配置
st.set_page_config(
    page_title="攻击配置 - AI模型安全评估平台",
    page_icon="⚔️",
    layout="wide"
)

# 初始化管理器
auth_manager = AuthManager()
attack_manager = AttackManager()
model_loader = ModelLoader()

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
st.title("⚔️ 攻击配置管理")
st.markdown("---")

# 侧边栏 - 功能选择
st.sidebar.header("功能选择")
function_choice = st.sidebar.selectbox(
    "选择功能",
    ["创建攻击配置", "我的配置", "攻击算法库", "配置统计"]
)

if function_choice == "创建攻击配置":
    st.header("🎯 创建攻击配置")
    
    # 配置基本信息
    col1, col2 = st.columns(2)
    
    with col1:
        config_name = st.text_input(
            "配置名称",
            placeholder="输入配置名称",
            help="为您的攻击配置起一个描述性的名称"
        )
        
        attack_type = st.selectbox(
            "攻击类型",
            ["evasion", "poisoning", "extraction", "inference"],
            help="选择攻击类型：\n- evasion: 逃避攻击\n- poisoning: 投毒攻击\n- extraction: 提取攻击\n- inference: 推理攻击"
        )
    
    with col2:
        config_description = st.text_area(
            "配置描述",
            placeholder="描述此配置的用途和特点",
            height=100
        )
    
    # 获取对应类型的攻击算法
    available_attacks = attack_manager.get_attack_by_type(attack_type)
    
    if available_attacks:
        st.subheader("🔧 攻击算法配置")
        
        # 选择攻击算法
        algorithm_choice = st.selectbox(
            "选择攻击算法",
            list(available_attacks.keys()),
            format_func=lambda x: f"{x} - {available_attacks[x]['name']}"
        )
        
        if algorithm_choice:
            algorithm_info = available_attacks[algorithm_choice]
            
            # 显示算法信息
            with st.expander("📖 算法说明", expanded=True):
                st.write(f"**算法名称**: {algorithm_info['name']}")
                st.write(f"**算法类型**: {algorithm_info['type']}")
                st.write(f"**算法描述**: {algorithm_info['description']}")
            
            # 参数配置
            st.subheader("⚙️ 参数配置")
            attack_params = {}
            
            # 创建参数输入界面
            param_cols = st.columns(2)
            col_idx = 0
            
            for param_name, param_config in algorithm_info['params'].items():
                with param_cols[col_idx % 2]:
                    if param_config['type'] == 'float':
                        attack_params[param_name] = st.number_input(
                            f"{param_name}",
                            value=param_config['default'],
                            min_value=param_config.get('min', 0.0),
                            max_value=param_config.get('max', 1.0),
                            step=0.01,
                            help=param_config['description']
                        )
                    elif param_config['type'] == 'int':
                        attack_params[param_name] = st.number_input(
                            f"{param_name}",
                            value=param_config['default'],
                            min_value=param_config.get('min', 1),
                            max_value=param_config.get('max', 1000),
                            step=1,
                            help=param_config['description']
                        )
                    elif param_config['type'] == 'bool':
                        attack_params[param_name] = st.checkbox(
                            f"{param_name}",
                            value=param_config['default'],
                            help=param_config['description']
                        )
                    elif param_config['type'] == 'select':
                        attack_params[param_name] = st.selectbox(
                            f"{param_name}",
                            options=param_config['options'],
                            index=param_config['options'].index(param_config['default']),
                            help=param_config['description']
                        )
                
                col_idx += 1
            
            # 高级选项
            with st.expander("🔬 高级选项"):
                batch_size = st.number_input(
                    "批处理大小",
                    value=32,
                    min_value=1,
                    max_value=512,
                    help="攻击时的批处理大小"
                )
                
                verbose = st.checkbox(
                    "详细输出",
                    value=True,
                    help="是否显示详细的攻击过程信息"
                )
                
                save_adversarial = st.checkbox(
                    "保存对抗样本",
                    value=True,
                    help="是否保存生成的对抗样本"
                )
            
            # 保存配置
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                if st.button("💾 保存配置", type="primary", use_container_width=True):
                    if not config_name:
                        st.error("请输入配置名称")
                    else:
                        # 验证参数
                        is_valid, message = attack_manager.validate_attack_params(
                            algorithm_choice, attack_params
                        )
                        
                        if is_valid:
                            # 构建完整配置
                            full_config = {
                                "algorithm": algorithm_choice,
                                "algorithm_name": algorithm_info['name'],
                                "attack_type": attack_type,
                                "description": config_description,
                                "params": attack_params,
                                "advanced_options": {
                                    "batch_size": batch_size,
                                    "verbose": verbose,
                                    "save_adversarial": save_adversarial
                                }
                            }
                            
                            # 保存配置
                            if attack_manager.save_attack_config(
                                config_name, full_config, user_id
                            ):
                                st.success(f"✅ 配置 '{config_name}' 保存成功！")
                                st.balloons()
                            else:
                                st.error("❌ 配置保存失败")
                        else:
                            st.error(f"❌ 参数验证失败: {message}")

elif function_choice == "我的配置":
    st.header("📋 我的攻击配置")
    
    # 获取用户配置
    user_configs = attack_manager.get_user_configs(user_id)
    
    if user_configs:
        # 搜索和筛选
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_term = st.text_input(
                "🔍 搜索配置",
                placeholder="输入配置名称或描述关键词"
            )
        
        with col2:
            attack_type_filter = st.selectbox(
                "攻击类型",
                ["全部", "evasion", "poisoning", "extraction", "inference"]
            )
        
        with col3:
            sort_by = st.selectbox(
                "排序方式",
                ["更新时间", "创建时间", "配置名称"]
            )
        
        # 筛选配置
        filtered_configs = user_configs
        
        if search_term:
            filtered_configs = [
                config for config in filtered_configs
                if search_term.lower() in config['name'].lower() or
                   search_term.lower() in config['config'].get('description', '').lower()
            ]
        
        if attack_type_filter != "全部":
            filtered_configs = [
                config for config in filtered_configs
                if config['config']['attack_type'] == attack_type_filter
            ]
        
        # 排序
        if sort_by == "创建时间":
            filtered_configs.sort(key=lambda x: x['created_at'], reverse=True)
        elif sort_by == "配置名称":
            filtered_configs.sort(key=lambda x: x['name'])
        else:  # 更新时间
            filtered_configs.sort(key=lambda x: x['updated_at'], reverse=True)
        
        st.markdown(f"**找到 {len(filtered_configs)} 个配置**")
        
        # 显示配置列表
        for i, config in enumerate(filtered_configs):
            with st.expander(
                f"⚔️ {config['name']} - {config['config']['algorithm_name']}",
                expanded=False
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**算法**: {config['config']['algorithm']} ({config['config']['algorithm_name']})")
                    st.write(f"**类型**: {config['config']['attack_type']}")
                    st.write(f"**描述**: {config['config'].get('description', '无描述')}")
                    st.write(f"**创建时间**: {config['created_at'][:19]}")
                    st.write(f"**更新时间**: {config['updated_at'][:19]}")
                    
                    # 显示参数
                    st.write("**参数配置**:")
                    params_df = pd.DataFrame([
                        {"参数名": k, "参数值": v}
                        for k, v in config['config']['params'].items()
                    ])
                    st.dataframe(params_df, use_container_width=True)
                
                with col2:
                    st.write("**操作**")
                    
                    # 编辑按钮
                    if st.button(f"✏️ 编辑", key=f"edit_{i}"):
                        st.info("编辑功能开发中...")
                    
                    # 复制按钮
                    if st.button(f"📋 复制", key=f"copy_{i}"):
                        st.info("复制功能开发中...")
                    
                    # 删除按钮
                    if st.button(f"🗑️ 删除", key=f"delete_{i}", type="secondary"):
                        if attack_manager.delete_attack_config(config['name'], user_id):
                            st.success("配置删除成功！")
                            st.rerun()
                        else:
                            st.error("配置删除失败")
                    
                    # 导出按钮
                    config_json = json.dumps(config, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="📥 导出",
                        data=config_json,
                        file_name=f"{config['name']}_config.json",
                        mime="application/json",
                        key=f"export_{i}"
                    )
    else:
        st.info("📝 您还没有创建任何攻击配置")
        st.markdown("点击上方的 **创建攻击配置** 开始创建您的第一个配置！")

elif function_choice == "攻击算法库":
    st.header("📚 攻击算法库")
    
    # 获取所有攻击算法
    all_attacks = attack_manager.get_attack_algorithms()
    
    # 按类型分组显示
    attack_types = list(set(attack['type'] for attack in all_attacks.values()))
    
    for attack_type in attack_types:
        st.subheader(f"🎯 {attack_type.title()} 攻击")
        
        type_attacks = attack_manager.get_attack_by_type(attack_type)
        
        # 创建卡片式布局
        cols = st.columns(2)
        col_idx = 0
        
        for algorithm_key, algorithm_info in type_attacks.items():
            with cols[col_idx % 2]:
                with st.container():
                    st.markdown(f"**{algorithm_key} - {algorithm_info['name']}**")
                    st.write(algorithm_info['description'])
                    
                    # 参数信息
                    with st.expander("查看参数"):
                        params_data = []
                        for param_name, param_config in algorithm_info['params'].items():
                            params_data.append({
                                "参数名": param_name,
                                "类型": param_config['type'],
                                "默认值": param_config['default'],
                                "描述": param_config['description']
                            })
                        
                        if params_data:
                            params_df = pd.DataFrame(params_data)
                            st.dataframe(params_df, use_container_width=True)
                    
                    st.markdown("---")
            
            col_idx += 1

elif function_choice == "配置统计":
    st.header("📊 配置统计")
    
    # 获取存储统计
    stats = attack_manager.get_storage_stats()
    
    # 总体统计
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "总配置数",
            stats['total_configs'],
            help="系统中所有用户的攻击配置总数"
        )
    
    with col2:
        total_size_mb = stats['total_size'] / (1024 * 1024)
        st.metric(
            "总存储大小",
            f"{total_size_mb:.2f} MB",
            help="所有攻击配置文件占用的存储空间"
        )
    
    with col3:
        user_count = len(stats['by_user'])
        st.metric(
            "活跃用户数",
            user_count,
            help="创建了攻击配置的用户数量"
        )
    
    # 用户配置统计
    if user_role == 'admin' and stats['by_user']:
        st.subheader("👥 用户配置分布")
        
        user_stats_data = []
        for uid, user_stat in stats['by_user'].items():
            user_stats_data.append({
                "用户ID": uid,
                "配置数量": user_stat['count'],
                "存储大小(KB)": f"{user_stat['size'] / 1024:.2f}"
            })
        
        user_stats_df = pd.DataFrame(user_stats_data)
        st.dataframe(user_stats_df, use_container_width=True)
    
    # 个人统计
    st.subheader("👤 我的配置统计")
    user_configs = attack_manager.get_user_configs(user_id)
    
    if user_configs:
        # 按算法类型统计
        type_counts = {}
        algorithm_counts = {}
        
        for config in user_configs:
            attack_type = config['config']['attack_type']
            algorithm = config['config']['algorithm']
            
            type_counts[attack_type] = type_counts.get(attack_type, 0) + 1
            algorithm_counts[algorithm] = algorithm_counts.get(algorithm, 0) + 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**按攻击类型统计**")
            type_df = pd.DataFrame([
                {"攻击类型": k, "配置数量": v}
                for k, v in type_counts.items()
            ])
            st.dataframe(type_df, use_container_width=True)
        
        with col2:
            st.write("**按算法统计**")
            algorithm_df = pd.DataFrame([
                {"算法": k, "配置数量": v}
                for k, v in algorithm_counts.items()
            ])
            st.dataframe(algorithm_df, use_container_width=True)
    else:
        st.info("您还没有创建任何攻击配置")

# 页面底部信息
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>💡 提示：攻击配置用于定义对抗攻击的算法和参数，是安全评估的重要组成部分</small>
    </div>
    """,
    unsafe_allow_html=True
)