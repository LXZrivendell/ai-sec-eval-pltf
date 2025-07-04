import streamlit as st
from core.defense.defense_manager import DefenseManager

def main():
    st.title("🛡️ 防御策略配置")
    
    defense_manager = DefenseManager()
    
    # 防御方法选择
    defense_method = st.selectbox(
        "选择防御方法",
        list(defense_manager.defense_methods.keys())
    )
    
    # 参数配置
    method_info = defense_manager.defense_methods[defense_method]
    st.write(f"**方法类型**: {method_info['type']}")
    
    # 动态参数配置界面
    params = {}
    for param_name, default_value in method_info['params'].items():
        if isinstance(default_value, float):
            params[param_name] = st.slider(
                param_name, 0.0, 1.0, default_value
            )
        elif isinstance(default_value, int):
            params[param_name] = st.number_input(
                param_name, value=default_value
            )
    
    # 保存配置
    if st.button("保存防御配置"):
        defense_config = {
            'method': defense_method,
            'params': params
        }
        # 保存逻辑
        st.success("防御配置已保存")

if __name__ == "__main__":
    main()