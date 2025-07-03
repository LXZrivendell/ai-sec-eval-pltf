import streamlit as st
import pandas as pd
from datetime import datetime
import os
from core.auth_manager import AuthManager
from core.security_evaluator import SecurityEvaluator
# 修改第7行的导入语句
from core.reporting import ReportGenerator
# 替换原来的：from core.report_generator import ReportGenerator

# 页面配置
st.set_page_config(
    page_title="报告管理 - AI模型安全评估平台",
    page_icon="📊",
    layout="wide"
)

# 初始化管理器
auth_manager = AuthManager()
security_evaluator = SecurityEvaluator()
report_generator = ReportGenerator()

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
st.title("📊 报告管理")
st.markdown("---")

# 侧边栏 - 功能选择
st.sidebar.header("功能选择")
function_choice = st.sidebar.selectbox(
    "选择功能",
    ["生成报告", "报告列表", "汇总报告", "报告统计"]
)

if function_choice == "生成报告":
    st.header("📝 生成评估报告")
    
    # 获取已完成的评估
    if user_role == 'admin':
        completed_evaluations = security_evaluator.get_completed_evaluations()
    else:
        completed_evaluations = security_evaluator.get_user_completed_evaluations(user_id)
    
    if completed_evaluations:
        # 选择评估
        evaluation_options = {f"{eval['name']} ({eval['completed_at'][:19]})":
                            eval for eval in completed_evaluations}
        
        selected_eval_key = st.selectbox(
            "选择要生成报告的评估",
            list(evaluation_options.keys()),
            help="选择一个已完成的评估生成详细报告"
        )
        
        if selected_eval_key:
            selected_evaluation = evaluation_options[selected_eval_key]
            
            # 显示评估信息
            with st.expander("📋 评估信息", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**评估名称**: {selected_evaluation['name']}")
                    st.write(f"**评估类型**: {selected_evaluation['type']}")
                    st.write(f"**完成时间**: {selected_evaluation['completed_at'][:19]}")
                
                with col2:
                    if selected_evaluation.get('results'):
                        results = selected_evaluation['results']
                        st.write(f"**原始准确率**: {results.get('original_accuracy', 0):.2%}")
                        st.write(f"**攻击成功率**: {results.get('attack_success_rate', 0):.2%}")
                        st.write(f"**鲁棒性得分**: {results.get('robustness_score', 0):.2f}")
                
                with col3:
                    if selected_evaluation.get('config'):
                        config = selected_evaluation['config']
                        st.write(f"**模型**: {config.get('model', {}).get('name', 'N/A')}")
                        st.write(f"**数据集**: {config.get('dataset', {}).get('name', 'N/A')}")
                        st.write(f"**攻击算法**: {len(config.get('attack_configs', []))} 个")
            
            # 报告配置
            st.subheader("⚙️ 报告配置")
            
            col1, col2 = st.columns(2)
            
            with col1:
                report_format = st.selectbox(
                    "报告格式",
                    ["HTML", "JSON", "PDF"],
                    help="选择生成的报告格式"
                )
                
                include_charts = st.checkbox(
                    "包含图表",
                    value=True,
                    help="是否在报告中包含可视化图表"
                )
            
            with col2:
                include_recommendations = st.checkbox(
                    "包含安全建议",
                    value=True,
                    help="是否在报告中包含安全改进建议"
                )
                
                detailed_analysis = st.checkbox(
                    "详细分析",
                    value=True,
                    help="是否包含详细的攻击分析"
                )
            
            # 生成报告
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                if st.button("📊 生成报告", type="primary", use_container_width=True):
                    with st.spinner("正在生成报告..."):
                        # 准备报告数据 - 重构数据结构以匹配报告生成器期望
                        report_data = {
                            'evaluation_id': selected_evaluation.get('id'),
                            'timestamp': selected_evaluation.get('created_at'),
                            'model_info': selected_evaluation.get('config', {}).get('model', {}),
                            'dataset_info': selected_evaluation.get('config', {}).get('dataset', {}),
                            'attack_config': selected_evaluation.get('config', {}).get('attack_configs', [{}])[0] if selected_evaluation.get('config', {}).get('attack_configs') else {},
                            'results': selected_evaluation.get('results', {}),
                            'evaluation_params': {},  # 如果需要可以从其他地方获取
                            'attack_stats': {}  # 如果需要可以从其他地方获取
                        }
                        
                        # 添加报告配置
                        report_data['config'] = {
                            'include_charts': include_charts,
                            'include_recommendations': include_recommendations,
                            'detailed_analysis': detailed_analysis
                        }
                        
                        # 生成报告
                        report_path = report_generator.generate_report(
                            report_data, 
                            report_format=report_format.lower()
                        )
                        
                        if report_path:
                            st.success(f"✅ 报告生成成功！")
                            st.info(f"📁 报告保存路径: {report_path}")
                            
                            # 提供下载链接
                            if os.path.exists(report_path):
                                with open(report_path, 'rb') as f:
                                    file_data = f.read()
                                
                                st.download_button(
                                    label=f"📥 下载 {report_format} 报告",
                                    data=file_data,
                                    file_name=os.path.basename(report_path),
                                    mime={
                                        'html': 'text/html',
                                        'json': 'application/json',
                                        'pdf': 'application/pdf'
                                    }.get(report_format.lower(), 'application/octet-stream')
                                )
                            
                            st.balloons()
                        else:
                            st.error("❌ 报告生成失败")
    else:
        st.info("📝 没有已完成的评估可以生成报告")
        st.markdown("请先完成安全评估，然后再生成报告")

elif function_choice == "报告列表":
    st.header("📋 报告列表")
    
    # 获取报告列表
    reports = report_generator.get_report_list()
    
    if reports:
        # 搜索和筛选
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_term = st.text_input(
                "🔍 搜索报告",
                placeholder="输入报告名称关键词"
            )
        
        with col2:
            format_filter = st.selectbox(
                "格式筛选",
                ["全部", "HTML", "JSON", "PDF"]
            )
        
        with col3:
            sort_by = st.selectbox(
                "排序方式",
                ["创建时间", "文件名", "文件大小"]
            )
        
        # 筛选报告
        filtered_reports = reports
        
        if search_term:
            filtered_reports = [
                report for report in filtered_reports
                if search_term.lower() in report['filename'].lower()
            ]
        
        if format_filter != "全部":
            filtered_reports = [
                report for report in filtered_reports
                if report['format'] == format_filter
            ]
        
        # 排序
        if sort_by == "创建时间":
            filtered_reports.sort(key=lambda x: x['created_at'], reverse=True)
        elif sort_by == "文件名":
            filtered_reports.sort(key=lambda x: x['filename'])
        else:  # 文件大小
            filtered_reports.sort(key=lambda x: x['size'], reverse=True)
        
        st.markdown(f"**找到 {len(filtered_reports)} 个报告**")
        
        # 显示报告列表
        for i, report in enumerate(filtered_reports):
            with st.expander(
                f"📊 {report['filename']} ({report['format']})",
                expanded=False
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**文件名**: {report['filename']}")
                    st.write(f"**格式**: {report['format']}")
                    st.write(f"**大小**: {report['size'] / 1024:.2f} KB")
                    st.write(f"**创建时间**: {report['created_at'][:19]}")
                    
                    if report.get('evaluation_name'):
                        st.write(f"**评估名称**: {report['evaluation_name']}")
                
                with col2:
                    st.write("**操作**")
                    
                    # 下载按钮
                    if os.path.exists(report['filepath']):
                        with open(report['filepath'], 'rb') as f:
                            file_data = f.read()
                        
                        st.download_button(
                            label="📥 下载",
                            data=file_data,
                            file_name=report['filename'],
                            mime={
                                'HTML': 'text/html',
                                'JSON': 'application/json',
                                'PDF': 'application/pdf'
                            }.get(report['format'], 'application/octet-stream'),
                            key=f"download_{i}"
                        )
                    
                    # 预览按钮（仅HTML和JSON）
                    if report['format'] in ['HTML', 'JSON']:
                        if st.button(f"👁️ 预览", key=f"preview_{i}"):
                            st.session_state[f"show_preview_{i}"] = True
                    
                    # 删除按钮
                    if st.button(f"🗑️ 删除", key=f"delete_{i}", type="secondary"):
                        if report_generator.delete_report(report['filepath']):
                            st.success("报告删除成功！")
                            st.rerun()
                        else:
                            st.error("报告删除失败")
                
                # 显示预览
                if st.session_state.get(f"show_preview_{i}", False):
                    st.markdown("**预览**")
                    
                    try:
                        if report['format'] == 'HTML':
                            with open(report['filepath'], 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            st.components.v1.html(html_content, height=600, scrolling=True)
                        
                        elif report['format'] == 'JSON':
                            with open(report['filepath'], 'r', encoding='utf-8') as f:
                                json_content = f.read()
                            st.code(json_content, language='json')
                    
                    except Exception as e:
                        st.error(f"预览失败: {str(e)}")
                    
                    if st.button(f"❌ 关闭预览", key=f"close_preview_{i}"):
                        st.session_state[f"show_preview_{i}"] = False
                        st.rerun()
    else:
        st.info("📝 还没有生成任何报告")
        st.markdown("在 **生成报告** 页面创建您的第一个报告！")

elif function_choice == "汇总报告":
    st.header("📈 汇总报告")
    
    # 获取所有评估数据
    if user_role == 'admin':
        all_evaluations = security_evaluator.get_all_evaluations()
    else:
        all_evaluations = security_evaluator.get_user_evaluations(user_id)
    
    if all_evaluations:
        # 汇总统计
        st.subheader("📊 总体统计")
        
        total_evaluations = len(all_evaluations)
        completed_evaluations = len([e for e in all_evaluations if e['status'] == '已完成'])
        running_evaluations = len([e for e in all_evaluations if e['status'] == '运行中'])
        failed_evaluations = len([e for e in all_evaluations if e['status'] == '失败'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总评估数", total_evaluations)
        
        with col2:
            st.metric("已完成", completed_evaluations)
        
        with col3:
            st.metric("运行中", running_evaluations)
        
        with col4:
            st.metric("失败", failed_evaluations)
        
        # 评估类型分布
        st.subheader("📊 评估类型分布")
        
        type_counts = {}
        for evaluation in all_evaluations:
            eval_type = evaluation['type']
            type_counts[eval_type] = type_counts.get(eval_type, 0) + 1
        
        if type_counts:
            type_df = pd.DataFrame([
                {"评估类型": k, "数量": v, "占比": f"{v/total_evaluations:.1%}"}
                for k, v in type_counts.items()
            ])
            st.dataframe(type_df, use_container_width=True)
        
        # 安全性分析
        completed_evals = [e for e in all_evaluations if e['status'] == '已完成' and e.get('results')]
        
        if completed_evals:
            st.subheader("🛡️ 安全性分析")
            
            # 计算平均指标
            avg_accuracy = sum(e['results'].get('original_accuracy', 0) for e in completed_evals) / len(completed_evals)
            avg_attack_success = sum(e['results'].get('attack_success_rate', 0) for e in completed_evals) / len(completed_evals)
            avg_robustness = sum(e['results'].get('robustness_score', 0) for e in completed_evals) / len(completed_evals)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("平均原始准确率", f"{avg_accuracy:.2%}")
            
            with col2:
                st.metric("平均攻击成功率", f"{avg_attack_success:.2%}")
            
            with col3:
                st.metric("平均鲁棒性得分", f"{avg_robustness:.2f}")
            
            # 安全等级分布
            security_levels = []
            for evaluation in completed_evals:
                if evaluation['results'].get('security_level'):
                    security_levels.append(evaluation['results']['security_level'])
            
            if security_levels:
                st.write("**安全等级分布**")
                level_counts = {level: security_levels.count(level) for level in set(security_levels)}
                level_df = pd.DataFrame([
                    {"安全等级": k, "数量": v, "占比": f"{v/len(security_levels):.1%}"}
                    for k, v in level_counts.items()
                ])
                st.dataframe(level_df, use_container_width=True)
        
        # 时间趋势分析
        st.subheader("📈 时间趋势")
        
        # 按月统计
        monthly_counts = {}
        for evaluation in all_evaluations:
            month = evaluation['created_at'][:7]  # YYYY-MM
            monthly_counts[month] = monthly_counts.get(month, 0) + 1
        
        if monthly_counts:
            months = sorted(monthly_counts.keys())
            counts = [monthly_counts[month] for month in months]
            
            trend_df = pd.DataFrame({
                "月份": months,
                "评估数量": counts
            })
            
            st.line_chart(trend_df.set_index("月份"))
        
        # 生成汇总报告
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("📊 生成汇总报告", type="primary", use_container_width=True):
                with st.spinner("正在生成汇总报告..."):
                    summary_report_path = report_generator.generate_summary_report(all_evaluations)
                    
                    if summary_report_path:
                        st.success("✅ 汇总报告生成成功！")
                        
                        # 提供下载
                        with open(summary_report_path, 'rb') as f:
                            report_data = f.read()
                        
                        st.download_button(
                            label="📥 下载汇总报告",
                            data=report_data,
                            file_name=os.path.basename(summary_report_path),
                            mime="application/json"
                        )
                    else:
                        st.error("❌ 汇总报告生成失败")
    else:
        st.info("📝 没有评估数据可以生成汇总报告")

elif function_choice == "报告统计":
    st.header("📊 报告统计")
    
    # 获取存储统计
    stats = report_generator.get_storage_stats()
    
    # 总体统计
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "报告总数",
            stats['total_reports'],
            help="系统中所有报告文件的总数"
        )
    
    with col2:
        total_size_mb = stats['total_size'] / (1024 * 1024)
        st.metric(
            "总存储大小",
            f"{total_size_mb:.2f} MB",
            help="所有报告文件占用的存储空间"
        )
    
    with col3:
        format_count = len(stats['by_format'])
        st.metric(
            "支持格式数",
            format_count,
            help="系统支持的报告格式数量"
        )
    
    # 格式分布
    if stats['by_format']:
        st.subheader("📊 格式分布")
        
        format_data = []
        for format_type, format_stats in stats['by_format'].items():
            format_data.append({
                "格式": format_type,
                "数量": format_stats['count'],
                "大小(MB)": f"{format_stats['size'] / (1024 * 1024):.2f}",
                "占比": f"{format_stats['count'] / stats['total_reports']:.1%}"
            })
        
        format_df = pd.DataFrame(format_data)
        st.dataframe(format_df, use_container_width=True)
        
        # 可视化
        col1, col2 = st.columns(2)
        
        with col1:
            # 数量分布饼图
            import plotly.express as px
            fig1 = px.pie(
                format_df,
                values='数量',
                names='格式',
                title="报告格式数量分布"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # 大小分布柱状图
            sizes = [float(size.replace(' MB', '')) for size in format_df['大小(MB)']]
            fig2 = px.bar(
                x=format_df['格式'],
                y=sizes,
                title="报告格式大小分布",
                labels={'x': '格式', 'y': '大小 (MB)'}
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # 清理建议
    st.subheader("🧹 存储管理")
    
    if stats['total_reports'] > 0:
        avg_size = stats['total_size'] / stats['total_reports']
        st.write(f"**平均文件大小**: {avg_size / 1024:.2f} KB")
        
        if total_size_mb > 100:  # 超过100MB
            st.warning("⚠️ 报告存储空间较大，建议定期清理旧报告")
        
        if stats['total_reports'] > 50:  # 超过50个报告
            st.info("💡 建议定期归档或删除不需要的报告以节省存储空间")
    
    # 清理操作
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ 清理旧报告", help="删除30天前的报告"):
            st.warning("清理功能开发中...")
    
    with col2:
        if st.button("📦 归档报告", help="将报告打包归档"):
            st.warning("归档功能开发中...")

# 页面底部信息
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>📊 提示：定期生成和查看评估报告有助于跟踪模型安全性的变化趋势</small>
    </div>
    """,
    unsafe_allow_html=True
)