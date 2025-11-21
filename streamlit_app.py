"""
基于Streamlit的Web应用程序
允许用户上传Excel文件，选择字段并展示分析结果
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import json
from io import BytesIO
from datetime import datetime, timedelta
import plotly.io as pio
import base64

# 导入chansey模块
import chansey
from chansey.llm_analyzer import LLMAnalyzer

# 设置页面配置
st.set_page_config(
    page_title="Chansey数据分析工具",
    page_icon="📊",
    layout="wide"
)

# 初始化会话状态
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'sheet_name' not in st.session_state:
    st.session_state.sheet_name = None
if 'date_column' not in st.session_state:
    st.session_state.date_column = None
if 'dimension_columns' not in st.session_state:
    st.session_state.dimension_columns = []
if 'metric_configurations' not in st.session_state:
    st.session_state.metric_configurations = []

def main():
    st.title("📊 Chansey数据分析工具")
    st.markdown("---")
    
    # 侧边栏导航
    st.sidebar.title("导航")
    page = st.sidebar.radio("选择页面", ["上传数据", "字段配置", "数据分析", "结果展示"])
    
    if page == "上传数据":
        upload_page()
    elif page == "字段配置":
        configure_page()
    elif page == "数据分析":
        analysis_page()
    elif page == "结果展示":
        results_page()

def upload_page():
    st.header("上传数据文件")
    
    uploaded_file = st.file_uploader(
        "选择Excel文件", 
        type=["xlsx", "xls"],
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.success("文件上传成功！")
        
        # 读取Excel文件的工作表
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            st.subheader("选择工作表")
            selected_sheet = st.selectbox("选择要分析的工作表", sheet_names)
            
            if st.button("加载数据"):
                # 读取选定的工作表
                st.session_state.df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                st.session_state.sheet_name = selected_sheet
                
                st.success(f"成功加载工作表 '{selected_sheet}' 的数据")
                st.subheader("数据预览")
                st.dataframe(st.session_state.df.head(10))
                
                # 显示数据信息
                st.subheader("数据信息")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"行数: {st.session_state.df.shape[0]}")
                    st.write(f"列数: {st.session_state.df.shape[1]}")
                with col2:
                    st.write("列名:")
                    st.write(st.session_state.df.columns.tolist())
                
                st.session_state.metric_configurations = []  # 重置指标配置
        except Exception as e:
            st.error(f"读取Excel文件时出错: {str(e)}")
    else:
        st.info("请选择一个Excel文件上传")

def configure_page():
    st.header("字段配置")
    
    if st.session_state.df is None:
        st.warning("请先上传并加载数据")
        return
    
    df = st.session_state.df
    columns = df.columns.tolist()
    
    # 选择日期字段
    st.subheader("1. 选择日期字段")
    date_column = st.selectbox(
        "选择用于区分观察期和对比期的日期字段",
        columns,
        index=columns.index(st.session_state.date_column) if st.session_state.date_column in columns else 0
    )
    st.session_state.date_column = date_column
    
    # 显示日期信息
    if date_column:
        unique_dates = sorted(df[date_column].dropna().unique())
        st.write(f"可选日期范围: {min(unique_dates)} 到 {max(unique_dates)}")
        st.write(f"唯一日期数量: {len(unique_dates)}")
    
    # 选择维度字段
    st.subheader("2. 选择维度字段")
    dimension_columns = st.multiselect(
        "选择用于分组分析的维度字段（可多选）",
        columns,
        default=st.session_state.dimension_columns
    )
    st.session_state.dimension_columns = dimension_columns
    
    # 配置指标字段
    st.subheader("3. 配置指标字段")
    st.markdown("""
    指标可以是以下两种类型之一：
    - **数值指标**: 直接使用的数值字段
    - **比例指标**: 需要指定分子和分母字段计算得出的比例
    """)
    
    # 显示当前配置的指标
    if st.session_state.metric_configurations:
        st.write("当前配置的指标:")
        for i, metric_config in enumerate(st.session_state.metric_configurations):
            col1, col2, col3 = st.columns([3, 3, 1])
            with col1:
                st.write(f"**指标名称**: {metric_config['name']}")
            with col2:
                if metric_config['type'] == 'numeric':
                    st.write(f"**类型**: 数值指标 ({metric_config['field']})")
                else:
                    st.write(f"**类型**: 比例指标 ({metric_config['numerator']}/{metric_config['denominator']})")
            with col3:
                if st.button("删除", key=f"delete_metric_{i}"):
                    st.session_state.metric_configurations.pop(i)
                    st.rerun()
    
    # 添加新指标
    st.subheader("添加新指标")
    
    metric_name = st.text_input("指标名称")
    
    metric_type = st.radio(
        "指标类型",
        ["数值指标", "比例指标"],
        horizontal=True
    )
    
    if metric_type == "数值指标":
        numeric_field = st.selectbox("选择数值字段", columns)
        if st.button("添加数值指标"):
            if metric_name and numeric_field:
                st.session_state.metric_configurations.append({
                    'name': metric_name,
                    'type': 'numeric',
                    'field': numeric_field
                })
                st.success(f"已添加数值指标: {metric_name}")
                st.rerun()
            else:
                st.warning("请填写指标名称并选择数值字段")
    else:  # 比例指标
        numerator_field = st.selectbox("选择分子字段", columns)
        denominator_field = st.selectbox("选择分母字段", columns)
        if st.button("添加比例指标"):
            if metric_name and numerator_field and denominator_field:
                st.session_state.metric_configurations.append({
                    'name': metric_name,
                    'type': 'ratio',
                    'numerator': numerator_field,
                    'denominator': denominator_field
                })
                st.success(f"已添加比例指标: {metric_name}")
                st.rerun()
            else:
                st.warning("请填写指标名称并选择分子、分母字段")
    
    # 显示配置摘要
    if st.session_state.date_column and st.session_state.dimension_columns and st.session_state.metric_configurations:
        st.markdown("---")
        st.subheader("配置摘要")
        st.write(f"**日期字段**: {st.session_state.date_column}")
        st.write(f"**维度字段**: {', '.join(st.session_state.dimension_columns)}")
        st.write("**指标配置**:")
        for config in st.session_state.metric_configurations:
            if config['type'] == 'numeric':
                st.write(f"  - {config['name']} (数值: {config['field']})")
            else:
                st.write(f"  - {config['name']} (比例: {config['numerator']}/{config['denominator']})")

def analysis_page():
    st.header("数据分析")
    
    if not st.session_state.metric_configurations:
        st.warning("请先完成字段配置")
        return
    
    df = st.session_state.df
    date_column = st.session_state.date_column
    dimension_columns = st.session_state.dimension_columns
    metric_configs = st.session_state.metric_configurations
    
    # 确保日期列是字符串格式
    df[date_column] = df[date_column].astype(str)
    
    # 获取唯一日期
    unique_dates = sorted(df[date_column].unique())
    
    # 如果日期少于2个，无法进行对比分析
    if len(unique_dates) < 2:
        st.error("数据中至少需要包含两个不同的日期才能进行对比分析")
        return
    
    # 选择观察期和对比期（支持时间段选择）
    st.subheader("选择分析期间")
    col1, col2 = st.columns(2)
    with col1:
        # 对比期选择
        st.write("**对比期选择**")
        cmp_period_type = st.radio("对比期类型", ["单日", "时间段"], key="cmp_period_type", horizontal=True)
        if cmp_period_type == "单日":
            cmp_date = st.selectbox("对比期日期", unique_dates, index=max(0, len(unique_dates)-2), key="cmp_single_date")
            cmp_date_range = [cmp_date, cmp_date]
        else:
            # 转换日期字符串为datetime对象以便处理
            unique_date_objs = [datetime.strptime(d, '%Y-%m-%d') if '-' in d else datetime.strptime(d, '%Y/%m/%d') if '/' in d else pd.to_datetime(d) for d in unique_dates]
            min_date = min(unique_date_objs)
            max_date = max(unique_date_objs)
            
            c1, c2 = st.columns(2)
            with c1:
                cmp_start = st.date_input("对比期开始", min_date, min_value=min_date, max_value=max_date, key="cmp_start")
            with c2:
                cmp_end = st.date_input("对比期结束", max_date, min_value=min_date, max_value=max_date, key="cmp_end")
            cmp_date_range = [cmp_start.strftime('%Y-%m-%d'), cmp_end.strftime('%Y-%m-%d')]
    
    with col2:
        # 观察期选择
        st.write("**观察期选择**")
        obs_period_type = st.radio("观察期类型", ["单日", "时间段"], key="obs_period_type", horizontal=True)
        if obs_period_type == "单日":
            obs_date = st.selectbox("观察期日期", unique_dates, index=len(unique_dates)-1, key="obs_single_date")
            obs_date_range = [obs_date, obs_date]
        else:
            # 转换日期字符串为datetime对象以便处理
            unique_date_objs = [datetime.strptime(d, '%Y-%m-%d') if '-' in d else datetime.strptime(d, '%Y/%m/%d') if '/' in d else pd.to_datetime(d) for d in unique_dates]
            min_date = min(unique_date_objs)
            max_date = max(unique_date_objs)
            
            o1, o2 = st.columns(2)
            with o1:
                obs_start = st.date_input("观察期开始", min_date, min_value=min_date, max_value=max_date, key="obs_start")
            with o2:
                obs_end = st.date_input("观察期结束", max_date, min_value=min_date, max_value=max_date, key="obs_end")
            obs_date_range = [obs_start.strftime('%Y-%m-%d'), obs_end.strftime('%Y-%m-%d')]
    
    # 保存分析配置到会话状态
    st.session_state.analysis_config = {
        'obs_period_type': obs_period_type,
        'obs_date_range': obs_date_range,
        'cmp_period_type': cmp_period_type,
        'cmp_date_range': cmp_date_range
    }
    
    # 执行分析
    if st.button("执行分析", type="primary"):
        with st.spinner("正在分析数据..."):
            # 准备分析结果
            analysis_results = []
            
            # 对每个指标进行分析
            for metric_config in metric_configs:
                try:
                    metric_name = metric_config['name']
                    
                    # 根据指标类型处理数据
                    if metric_config['type'] == 'numeric':
                        # 数值指标
                        field = metric_config['field']
                        df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)
                        
                        # 对每个维度分别进行分析
                        for dim_col in dimension_columns:
                            # 筛选观察期和对比期数据
                            if obs_period_type == "单日":
                                obs_data = df[df[date_column] == obs_date_range[0]]
                            else:
                                # 获取观察期范围内的所有日期
                                obs_dates = [d for d in unique_dates if obs_date_range[0] <= d <= obs_date_range[1]]
                                obs_data = df[df[date_column].isin(obs_dates)]
                            
                            if cmp_period_type == "单日":
                                cmp_data = df[df[date_column] == cmp_date_range[0]]
                            else:
                                # 获取对比期范围内的所有日期
                                cmp_dates = [d for d in unique_dates if cmp_date_range[0] <= d <= cmp_date_range[1]]
                                cmp_data = df[df[date_column].isin(cmp_dates)]
                            
                            # 按维度分组计算指标值
                            obs_summary = obs_data.groupby(dim_col)[field].sum().reset_index()
                            obs_summary = obs_summary.rename(columns={field: f'{field}_obs'})
                            
                            cmp_summary = cmp_data.groupby(dim_col)[field].sum().reset_index()
                            cmp_summary = cmp_summary.rename(columns={field: f'{field}_cmp'})
                            
                            # 合并数据
                            merged = pd.merge(obs_summary, cmp_summary, on=dim_col, how='outer').fillna(0)
                            
                            # 计算分诊器贡献度（基于分诊器原理）
                            # 总体指标值
                            obs_total = merged[f'{field}_obs'].sum()
                            cmp_total = merged[f'{field}_cmp'].sum()
                            
                            # 计算权重（占比）
                            merged['obs_weight'] = np.where(obs_total != 0, merged[f'{field}_obs'] / obs_total, 0)
                            merged['cmp_weight'] = np.where(cmp_total != 0, merged[f'{field}_cmp'] / cmp_total, 0)
                            
                            # 计算贡献度（分诊器核心算法）
                            # 量的贡献（占比变化导致的变化）
                            merged['weight_contribution'] = (merged['obs_weight'] - merged['cmp_weight']) * cmp_total
                            
                            # 率的贡献（子类数值变化导致的变化）
                            merged['value_contribution'] = merged['cmp_weight'] * (merged[f'{field}_obs'] - merged[f'{field}_cmp'])
                            
                            # 总贡献
                            merged['total_contribution'] = merged['weight_contribution'] + merged['value_contribution']
                            
                            # 重新组织列结构
                            result_df = pd.DataFrame({
                                dim_col: merged[dim_col],
                                '观察期_数值': merged[f'{field}_obs'],
                                '对比期_数值': merged[f'{field}_cmp'],
                                '数值变化': merged[f'{field}_obs'] - merged[f'{field}_cmp'],
                                '观察期_权重': merged['obs_weight'],
                                '对比期_权重': merged['cmp_weight'],
                                '权重变化': merged['obs_weight'] - merged['cmp_weight'],
                                '量的贡献': merged['weight_contribution'],
                                '率的贡献': merged['value_contribution'],
                                '总贡献': merged['total_contribution']
                            })
                            
                            # 添加到结果中
                            result = {
                                'metric_name': metric_name,
                                'metric_type': 'numeric',
                                'field': field,
                                'dimension': dim_col,
                                'data': result_df.round(4).to_dict(orient='records'),
                                'obs_date_range': obs_date_range,
                                'cmp_date_range': cmp_date_range,
                                'obs_total': obs_total,
                                'cmp_total': cmp_total
                            }
                            analysis_results.append(result)
                        
                    else:
                        # 比例指标
                        numerator = metric_config['numerator']
                        denominator = metric_config['denominator']
                        
                        # 确保字段是数值类型
                        df[numerator] = pd.to_numeric(df[numerator], errors='coerce').fillna(0)
                        df[denominator] = pd.to_numeric(df[denominator], errors='coerce').fillna(1)
                        
                        # 避免除以零
                        df[denominator] = df[denominator].replace(0, 1)
                        
                        # 对每个维度分别进行分析
                        for dim_col in dimension_columns:
                            # 筛选观察期和对比期数据
                            if obs_period_type == "单日":
                                obs_data = df[df[date_column] == obs_date_range[0]]
                            else:
                                # 获取观察期范围内的所有日期
                                obs_dates = [d for d in unique_dates if obs_date_range[0] <= d <= obs_date_range[1]]
                                obs_data = df[df[date_column].isin(obs_dates)]
                            
                            if cmp_period_type == "单日":
                                cmp_data = df[df[date_column] == cmp_date_range[0]]
                            else:
                                # 获取对比期范围内的所有日期
                                cmp_dates = [d for d in unique_dates if cmp_date_range[0] <= d <= cmp_date_range[1]]
                                cmp_data = df[df[date_column].isin(cmp_dates)]
                            
                            # 按维度分组计算指标值
                            obs_summary = obs_data.groupby(dim_col)[[numerator, denominator]].sum().reset_index()
                            obs_summary[f'{metric_name}_obs_rate'] = obs_summary[numerator] / obs_summary[denominator]
                            
                            cmp_summary = cmp_data.groupby(dim_col)[[numerator, denominator]].sum().reset_index()
                            cmp_summary[f'{metric_name}_cmp_rate'] = cmp_summary[numerator] / cmp_summary[denominator]
                            
                            # 合并数据
                            merged = pd.merge(
                                obs_summary[[dim_col, numerator, denominator, f'{metric_name}_obs_rate']], 
                                cmp_summary[[dim_col, numerator, denominator, f'{metric_name}_cmp_rate']], 
                                on=dim_col, 
                                suffixes=('_obs', '_cmp'), 
                                how='outer'
                            ).fillna(0)
                            
                            # 计算分诊器贡献度（基于分诊器原理）
                            # 总体指标值
                            obs_numerator_total = merged[f'{numerator}_obs'].sum()
                            obs_denominator_total = merged[f'{denominator}_obs'].sum()
                            cmp_numerator_total = merged[f'{numerator}_cmp'].sum()
                            cmp_denominator_total = merged[f'{denominator}_cmp'].sum()
                            
                            obs_total_rate = obs_numerator_total / obs_denominator_total if obs_denominator_total != 0 else 0
                            cmp_total_rate = cmp_numerator_total / cmp_denominator_total if cmp_denominator_total != 0 else 0
                            
                            # 计算权重（占比）
                            merged['obs_weight'] = np.where(obs_denominator_total != 0, merged[f'{denominator}_obs'] / obs_denominator_total, 0)
                            merged['cmp_weight'] = np.where(cmp_denominator_total != 0, merged[f'{denominator}_cmp'] / cmp_denominator_total, 0)
                            
                            # 计算贡献度（分诊器核心算法）
                            # 量的贡献（占比变化导致的变化）
                            merged['weight_contribution'] = (merged['obs_weight'] - merged['cmp_weight']) * cmp_total_rate
                            
                            # 率的贡献（子类比率变化导致的变化）
                            merged['rate_contribution'] = merged['cmp_weight'] * (merged[f'{metric_name}_obs_rate'] - merged[f'{metric_name}_cmp_rate'])
                            
                            # 总贡献
                            merged['total_contribution'] = merged['weight_contribution'] + merged['rate_contribution']
                            
                            # 重新组织列结构
                            result_df = pd.DataFrame({
                                dim_col: merged[dim_col],
                                '观察期_数值': merged[f'{metric_name}_obs_rate'],
                                '对比期_数值': merged[f'{metric_name}_cmp_rate'],
                                '数值变化': merged[f'{metric_name}_obs_rate'] - merged[f'{metric_name}_cmp_rate'],
                                '观察期_权重': merged['obs_weight'],
                                '对比期_权重': merged['cmp_weight'],
                                '权重变化': merged['obs_weight'] - merged['cmp_weight'],
                                '量的贡献': merged['weight_contribution'],
                                '率的贡献': merged['rate_contribution'],
                                '总贡献': merged['total_contribution']
                            })
                            
                            # 添加到结果中
                            result = {
                                'metric_name': metric_name,
                                'metric_type': 'ratio',
                                'numerator': numerator,
                                'denominator': denominator,
                                'dimension': dim_col,
                                'data': result_df.round(4).to_dict(orient='records'),
                                'obs_date_range': obs_date_range,
                                'cmp_date_range': cmp_date_range,
                                'obs_total': obs_total_rate,
                                'cmp_total': cmp_total_rate
                            }
                            analysis_results.append(result)
                        
                except Exception as e:
                    st.error(f"分析指标 {metric_name} 时出错: {str(e)}")
            
            # 保存分析结果到会话状态
            st.session_state.analysis_results = analysis_results
            
            # 显示分析结果
            if analysis_results:
                st.success("分析完成！")
                
                # 创建标签页显示不同指标的结果
                unique_combinations = list(set([(r['metric_name'], r['dimension']) for r in analysis_results]))
                tab_labels = [f"{combo[0]} ({combo[1]})" for combo in unique_combinations]
                tabs = st.tabs(tab_labels)
                
                for i, (tab, combo) in enumerate(zip(tabs, unique_combinations)):
                    with tab:
                        metric_name, dimension = combo
                        # 找到匹配的结果
                        result = next((r for r in analysis_results 
                                     if r['metric_name'] == metric_name and r['dimension'] == dimension), None)
                        
                        if result:
                            st.subheader(f"{result['metric_name']} 按 {result['dimension']} 分析结果")
                            
                            if result['metric_type'] == 'numeric':
                                st.write(f"**指标类型**: 数值指标")
                                st.write(f"**数据字段**: {result['field']}")
                            else:
                                st.write(f"**指标类型**: 比例指标")
                                st.write(f"**分子字段**: {result['numerator']}")
                                st.write(f"**分母字段**: {result['denominator']}")
                            
                            st.write(f"**分析维度**: {result['dimension']}")
                            if len(result['obs_date_range']) == 1 or result['obs_date_range'][0] == result['obs_date_range'][1]:
                                st.write(f"**观察期**: {result['obs_date_range'][0]}")
                            else:
                                st.write(f"**观察期**: {result['obs_date_range'][0]} 到 {result['obs_date_range'][1]}")
                                
                            if len(result['cmp_date_range']) == 1 or result['cmp_date_range'][0] == result['cmp_date_range'][1]:
                                st.write(f"**对比期**: {result['cmp_date_range'][0]}")
                            else:
                                st.write(f"**对比期**: {result['cmp_date_range'][0]} 到 {result['cmp_date_range'][1]}")
                            
                            # 显示数据表
                            df_result = pd.DataFrame(result['data'])
                            st.dataframe(df_result, use_container_width=True)
                            
                            # 添加结果解释
                            st.subheader("结果解释")
                            total_obs = result['obs_total']
                            total_cmp = result['cmp_total']
                            total_delta = total_obs - total_cmp
                            total_delta_pct = (total_delta / total_cmp * 100) if total_cmp != 0 else np.inf
                            
                            # 计算主要贡献
                            weight_contrib_sum = df_result['量的贡献'].sum()
                            rate_contrib_sum = df_result['率的贡献'].sum()
                            
                            st.markdown(f"""
                            **整体变化分析**:
                            - 对比期总值: {total_cmp:.4f}
                            - 观察期总值: {total_obs:.4f}
                            - 绝对变化量: {total_delta:.4f}
                            - 相对变化率: {total_delta_pct:.2f}% ({'增加' if total_delta > 0 else '减少'})
                            
                            **贡献分解**:
                            - 量的变化贡献（占比变化导致）: {weight_contrib_sum:.4f}
                            - 率的变化贡献（子类指标变化导致）: {rate_contrib_sum:.4f}
                            
                            **维度贡献分析**:
                            - 最大正向贡献: {df_result.loc[df_result['总贡献'].idxmax(), dimension]} (贡献值: {df_result['总贡献'].max():.4f})
                            - 最大负向贡献: {df_result.loc[df_result['总贡献'].idxmin(), dimension]} (贡献值: {df_result['总贡献'].min():.4f})
                            - 主要贡献维度: 根据总贡献绝对值排序，前3名分别是 {
                                ', '.join(df_result.iloc[df_result['总贡献'].abs().nlargest(3).index][dimension].tolist())
                            }
                            """)

def results_page():
    st.header("分析结果展示")
    
    if 'analysis_results' not in st.session_state or not st.session_state.analysis_results:
        st.warning("请先执行数据分析")
        return
    
    analysis_results = st.session_state.analysis_results
    analysis_config = st.session_state.get('analysis_config', {})
    
    # 创建标签页
    tabs = st.tabs(["基尼系数分析", "详细结果"])
    
    # 基尼系数分析标签页
    with tabs[0]:
        st.subheader("基尼系数分析")
        
        st.markdown("""
        ## 基尼系数说明
        
        基尼系数用于衡量一个维度的贡献是否"集中"：
        - **基尼系数越大** → 贡献越不均衡 → 某些维度贡献格外大（重点维度）
        - **基尼系数越小** → 贡献越均衡 → 各维度贡献类似（无明显重点）
        
        计算公式：
        ```
        Gini = (1/n) * (n - (2 * Σ(Cum_i)) / Cum_n + 1)
        ```
        其中：
        - n: 组数
        - Cum_i: 前 i 组的累计贡献量
        - Cum_n: 所有分组总贡献量
        """)
        
        # 计算每个维度的基尼系数
        gini_data = []
        
        for result in analysis_results:
            df_result = pd.DataFrame(result['data'])
            
            # 获取总贡献并按绝对值排序
            contributions = df_result['总贡献'].abs().sort_values(ascending=False)
            
            if len(contributions) > 0:
                # 计算累计贡献
                cum_contributions = contributions.cumsum()
                cum_n = contributions.sum()  # 总贡献
                n = len(contributions)  # 组数
                
                # 计算基尼系数
                if cum_n != 0:
                    sum_cum = cum_contributions.sum()
                    # 应用修正公式：对 n 加 1.5 次幂
                    n_corrected = np.power(n, 1.5)
                    gini = (1 / n_corrected) * (n_corrected - (2 * sum_cum) / cum_n + 1)
                else:
                    gini = 0
                
                gini_data.append({
                    '指标名称': result['metric_name'],
                    '维度': result['dimension'],
                    '基尼系数': round(gini, 4),
                    '组数': n
                })
        
        if gini_data:
            # 创建基尼系数DataFrame并按基尼系数降序排列
            gini_df = pd.DataFrame(gini_data).sort_values('基尼系数', ascending=False)
            
            st.subheader("各维度基尼系数（按降序排列）")
            st.dataframe(gini_df, use_container_width=True)
            
            # 创建基尼系数可视化
            fig = px.bar(
                gini_df,
                x='维度',
                y='基尼系数',
                color='指标名称',
                title='各维度基尼系数对比',
                labels={'基尼系数': '基尼系数', '维度': '维度'}
            )
            
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # 添加解释
            st.subheader("结果解读")
            highest_gini = gini_df.iloc[0]
            st.markdown(f"""
            **重点维度识别**:
            - 基尼系数最高的维度是 **{highest_gini['维度']}**（指标：{highest_gini['指标名称']}），基尼系数为 **{highest_gini['基尼系数']:.4f}**
            - 这表明该维度的贡献最不均衡，少数分类贡献了大部分变化
            - 如果需要深入分析，建议优先关注此维度
            
            **分析建议**:
            - 基尼系数 > 0.5：高度集中，少数分类贡献显著
            - 基尼系数 0.3-0.5：中度集中，需要关注主要贡献分类
            - 基尼系数 < 0.3：较为均衡，各分类贡献相对平均
            """)
        else:
            st.warning("无法计算基尼系数，请检查分析结果")
    
    # 详细结果标签页
    with tabs[1]:
        # 选择要可视化的指标和维度组合
        unique_combinations = list(set([(r['metric_name'], r['dimension']) for r in analysis_results]))
        selected_combo = st.selectbox(
            "选择要可视化的指标和维度组合", 
            [f"{combo[0]} ({combo[1]})" for combo in unique_combinations]
        )
        
        if selected_combo:
            # 解析选择的组合
            parts = selected_combo.rsplit(' (', 1)
            metric_name = parts[0]
            dimension = parts[1][:-1]  # 去掉最后的')'
            
            # 找到选中的结果
            selected_result = next((result for result in analysis_results 
                                  if result['metric_name'] == metric_name and result['dimension'] == dimension), None)
            
            if selected_result:
                st.subheader(f"{selected_result['metric_name']} 按 {dimension} 可视化")
                
                df_result = pd.DataFrame(selected_result['data'])
                
                # 创建瀑布图（基于总贡献）
                # 按总贡献排序
                df_result = df_result.sort_values('总贡献', ascending=False)
                
                # 准备绘图数据
                obs_total = selected_result['obs_total']
                cmp_total = selected_result['cmp_total']
                
                x_data = ['对比期'] + df_result[dimension].tolist() + ['观察期']
                y_data = [cmp_total] + df_result['总贡献'].tolist() + [obs_total]
                
                # 创建瀑布图
                fig = go.Figure(go.Waterfall(
                    name="变化贡献",
                    orientation="v",
                    measure=["absolute"] + ["relative"] * len(df_result) + ["total"],
                    x=x_data,
                    textposition="outside",
                    text=[f"{y:.4f}" for y in y_data],
                    y=y_data,
                    connector={"line": {"color": "rgb(63, 63, 63)"}}
                ))
                
                fig.update_layout(
                    title=f"{selected_result['metric_name']} 指标按 {dimension} 变化瀑布图",
                    showlegend=True,
                    waterfallgap=0.1,
                    xaxis_title=dimension,
                    yaxis_title=selected_result['metric_name']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 创建贡献分解图
                st.subheader(f"{selected_result['metric_name']} 贡献分解")
                
                # 创建堆积柱状图显示量的贡献和率的贡献
                df_melted = df_result.melt(
                    id_vars=[dimension],
                    value_vars=['量的贡献', '率的贡献'],
                    var_name='贡献类型',
                    value_name='贡献值'
                )
                
                fig2 = px.bar(
                    df_melted,
                    x=dimension,
                    y='贡献值',
                    color='贡献类型',
                    title=f"{selected_result['metric_name']} 各{dimension}贡献分解",
                    labels={'贡献값': '贡献값', dimension: dimension}
                )
                
                fig2.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig2, use_container_width=True)
                
                # 显示数据表
                st.subheader("详细数据")
                st.dataframe(df_result, use_container_width=True)
                
                # 添加可视化解释
                st.subheader("可视化解释")
                st.markdown(f"""
                **瀑布图解读**:
                - 左侧"对比期"表示选定时间范围内的基准值
                - 中间各柱表示各{dimension}对整体变化的总贡献
                - 右侧"观察期"表示选定时间范围内的实际值
                - 绿色柱表示正向贡献(增长)，红色柱表示负向贡献(下降)
                
                **贡献分解图解读**:
                - 显示各{dimension}分类对指标变化的两种贡献类型：
                  - 量的贡献（占比变化导致）
                  - 率的贡献（子类指标变化导致）
                - 可以直观看出每种贡献类型的大小和方向
                """)
                
                # 添加大模型分析解读功能
                st.markdown("---")
                st.subheader("AI智能解读")
                
                if st.button("让AI分析这些结果"):
                    with st.spinner("正在请求AI分析，请稍候..."):
                        # 获取当前分析配置
                        analysis_config = {
                            'obs_date_range': selected_result.get('obs_date_range'),
                            'cmp_date_range': selected_result.get('cmp_date_range'),
                            'dimensions': selected_result.get('dimension')
                        }
                        
                        # 调用大模型进行分析
                        analyzer = LLMAnalyzer()
                        llm_response = analyzer.analyze_data([selected_result], analysis_config)
                        
                        if llm_response:
                            st.success("AI分析完成！")
                            st.markdown("**AI解读结果:**")
                            st.info(llm_response)
                        else:
                            st.error("AI分析失败，请检查网络连接或认证信息")

    # 添加完整的分析报告下载功能
    st.markdown("---")
    st.subheader("完整分析报告下载")
    st.markdown("下载包含所有分析过程和结果的完整报告")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("下载HTML报告"):
            html_report = generate_html_report(analysis_results, analysis_config)
            b64 = base64.b64encode(html_report.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="chansey_analysis_report.html">点击下载HTML报告</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        if st.button("下载Excel报告"):
            excel_buffer = generate_excel_report(analysis_results, analysis_config)
            b64 = base64.b64encode(excel_buffer.getvalue()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="chansey_analysis_report.xlsx">点击下载Excel报告</a>'
            st.markdown(href, unsafe_allow_html=True)

    # 添加AI整体分析功能
    st.markdown("---")
    st.subheader("AI整体分析")
    st.markdown("使用AI对所有分析结果进行综合解读")
    
    if st.button("综合AI分析"):
        with st.spinner("正在进行综合AI分析，请稍候..."):
            # 构造分析配置
            analysis_config = {}
            if analysis_results:
                analysis_config = {
                    'obs_date_range': analysis_results[0].get('obs_date_range'),
                    'cmp_date_range': analysis_results[0].get('cmp_date_range'),
                    'dimensions': [r.get('dimension') for r in analysis_results]
                }
            
            # 调用大模型进行分析
            analyzer = LLMAnalyzer()
            llm_response = analyzer.analyze_data(analysis_results, analysis_config)
            
            if llm_response:
                st.success("AI综合分析完成！")
                st.markdown("**AI综合解读结果:**")
                st.info(llm_response)
            else:
                st.error("AI综合分析失败，请检查网络连接或认证信息")

    # 添加AI分析报告导出功能
    st.markdown("---")
    st.subheader("AI分析报告导出")
    st.markdown("导出包含完整分析过程、AI解读和优化建议的分析报告")
    
    if st.button("生成AI分析报告"):
        with st.spinner("正在生成AI分析报告，请稍候..."):
            # 构造分析配置
            analysis_config = {}
            if analysis_results:
                analysis_config = {
                    'obs_date_range': analysis_results[0].get('obs_date_range'),
                    'cmp_date_range': analysis_results[0].get('cmp_date_range'),
                    'dimensions': [r.get('dimension') for r in analysis_results]
                }
            
            # 调用大模型生成详细的分析报告
            analyzer = LLMAnalyzer()
            llm_response = analyzer.analyze_data(analysis_results, analysis_config)
            
            if llm_response:
                # 生成完整的HTML分析报告
                html_report = generate_ai_html_report(analysis_results, analysis_config, llm_response)
                
                # 提供下载链接
                b64 = base64.b64encode(html_report.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="chansey_ai_analysis_report.html">点击下载AI分析报告</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                st.success("AI分析报告生成完成！")
                st.markdown("**AI分析报告内容:**")
                # 直接在网页上显示报告内容
                st.markdown(html_report, unsafe_allow_html=True)
            else:
                st.error("AI分析报告生成失败，请检查网络连接或认证信息")

def generate_ai_html_report(analysis_results, analysis_config, ai_interpretation):
    """生成AI分析报告HTML"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chansey AI数据分析报告</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            h1, h2, h3 { color: #333; }
            h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }
            h2 { border-left: 4px solid #007acc; padding-left: 10px; }
            table { border-collapse: collapse; width: 100%; margin: 10px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .section { margin: 20px 0; }
            .chart { margin: 20px 0; }
            img { max-width: 100%; }
            pre { background-color: #f5f5f5; padding: 10px; overflow-x: auto; }
            ul { margin-top: 0; }
            li { margin-bottom: 10px; }
        </style>
    </head>
    <body>
        <h1>Chansey AI数据分析报告</h1>
        
        <div class="section">
            <h2>1. 报告基本信息</h2>
    """
    
    # 报告基本信息
    html_content += f"<p><strong>观察期:</strong> {analysis_config.get('obs_date_range', 'N/A')}</p>\n"
    html_content += f"<p><strong>对比期:</strong> {analysis_config.get('cmp_date_range', 'N/A')}</p>\n"
    html_content += f"<p><strong>分析维度:</strong> {', '.join(analysis_config.get('dimensions', []))}</p>\n"
    
    html_content += "</div>\n"
    
    # 分析过程和结果
    html_content += """
        <div class="section">
            <h2>2. 分析过程和结果</h2>
    """
    
    html_content += "\n"
    
    for i, result in enumerate(analysis_results):
        html_content += f"<h3>指标 {i+1}: {result.get('metric_name', 'N/A')}</h3>\n"
        html_content += "<ul>\n"
        html_content += f"  <li><strong>指标类型:</strong> {'数值指标' if result.get('metric_type') == 'numeric' else '比例指标'}</li>\n"
        html_content += f"  <li><strong>分析维度:</strong> {result.get('dimension', 'N/A')}</li>\n"
        html_content += f"  <li><strong>对比期数值:</strong> {result.get('cmp_total', 'N/A')}</li>\n"
        html_content += f"  <li><strong>观察期数值:</strong> {result.get('obs_total', 'N/A')}</li>\n"
        
        # 计算变化情况
        obs_total = result.get('obs_total', 0)
        cmp_total = result.get('cmp_total', 0)
        change = obs_total - cmp_total
        change_pct = (change / cmp_total * 100) if cmp_total != 0 else 0
        html_content += f"  <li><strong>变化量:</strong> {change:.4f}</li>\n"
        html_content += f"  <li><strong>变化率:</strong> {change_pct:.2f}%</li>\n"
        html_content += "</ul>\n"
        
        # 添加详细数据摘要
        data = result.get('data', [])
        if data:
            html_content += "<h4>主要贡献维度:</h4>\n"
            html_content += "<table>\n"
            html_content += "<thead><tr>"
            
            # 获取表头
            keys = list(data[0].keys())
            for key in keys[:6]:  # 只显示前几列
                html_content += f"<th>{key}</th>"
            html_content += "<th>总贡献</th><th>量的贡献</th><th>率的贡献</th>"
            html_content += "</tr></thead>\n"
            html_content += "<tbody>\n"
            
            # 按总贡献排序
            sorted_data = sorted(data, key=lambda x: abs(x.get('总贡献', 0)), reverse=True)
            for j, item in enumerate(sorted_data[:10]):  # 取前10个
                html_content += "<tr>"
                # 显示维度值和其他关键信息
                for key in keys[:6]:
                    html_content += f"<td>{item.get(key, 'N/A')}</td>"
                html_content += f"<td>{item.get('总贡献', 0):.4f}</td>"
                html_content += f"<td>{item.get('量的贡献', 0):.4f}</td>"
                html_content += f"<td>{item.get('率的贡献', 0):.4f}</td>"
                html_content += "</tr>\n"
            
            html_content += "</tbody></table>\n"
        html_content += "<br/>\n"
    
    html_content += "</div>\n"
    
    # AI解读部分
    html_content += """
        <div class="section">
            <h2>3. AI分析解读</h2>
    """
    
    # 将AI解读内容转换为纯HTML格式，处理换行和列表
    if ai_interpretation:
        # 将文本中的换行转换为HTML的<br>标签
        ai_html = ai_interpretation.replace('\n\n', '</p><p>').replace('\n', '<br>')
        html_content += f"<p>{ai_html}</p>\n"
    else:
        html_content += "<p>未能生成AI分析解读内容。</p>\n"
        
    html_content += "</div>\n"
    
    # 优化建议部分
    html_content += """
        <div class="section">
            <h2>4. 可优化的方向和建议</h2>
            <p>为进一步提升分析效果和业务价值，建议考虑以下几个方面:</p>
            
            <h3>1. 数据质量优化:</h3>
            <ul>
                <li>检查数据完整性，确保没有缺失值影响分析准确性</li>
                <li>定期校验数据一致性，避免异常值干扰分析结果</li>
            </ul>
            
            <h3>2. 分析维度拓展:</h3>
            <ul>
                <li>结合业务场景，探索更多潜在影响因素</li>
                <li>考虑引入外部数据，丰富分析维度</li>
            </ul>
            
            <h3>3. 分析方法改进:</h3>
            <ul>
                <li>对于重要指标，可建立长期趋势监控机制</li>
                <li>引入预测模型，提前预警指标异常变化</li>
            </ul>
            
            <h3>4. 业务应用深化:</h3>
            <ul>
                <li>将分析结论转化为具体可执行的业务动作</li>
                <li>建立反馈机制，评估分析结果对业务的实际影响</li>
            </ul>
        </div>
    """
    
    # 报告结尾
    html_content += """
        <div class="section">
            <h2>5. 报告说明</h2>
            <p>本报告基于Chansey数据分析工具生成，结合AI模型对业务指标变化进行了深度解读。<br>
            报告内容仅供参考，具体业务决策请结合实际情况综合判断。</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

def generate_html_report(analysis_results, analysis_config):
    """生成HTML格式的完整分析报告"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chansey数据分析报告</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin: 10px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .section { margin: 20px 0; }
            .chart { margin: 20px 0; }
            img { max-width: 100%; }
        </style>
    </head>
    <body>
        <h1>Chansey数据分析报告</h1>
        
        <div class="section">
            <h2>分析配置</h2>
    """
    
    # 添加分析配置信息
    html_content += f"<p><strong>观察期类型:</strong> {analysis_config.get('obs_period_type', '未设置')}</p>\n"
    html_content += f"<p><strong>观察期范围:</strong> {analysis_config.get('obs_date_range', ['未设置'])[0]} 到 {analysis_config.get('obs_date_range', ['未设置'])[1]}</p>\n"
    html_content += f"<p><strong>对比期类型:</strong> {analysis_config.get('cmp_period_type', '未设置')}</p>\n"
    html_content += f"<p><strong>对比期范围:</strong> {analysis_config.get('cmp_date_range', ['未设置'])[0]} 到 {analysis_config.get('cmp_date_range', ['未设置', '未设置'])[1]}</p>\n"
    
    html_content += "</div>\n"
    
    # 添加每个分析结果
    for i, result in enumerate(analysis_results):
        df_result = pd.DataFrame(result['data'])
        
        html_content += f"""
        <div class="section">
            <h2>分析结果 {i+1}: {result['metric_name']} 按 {result['dimension']} 分析</h2>
            <p><strong>指标类型:</strong> {'数值指标' if result['metric_type'] == 'numeric' else '比例指标'}</p>
            <p><strong>分析维度:</strong> {result['dimension']}</p>
            <p><strong>观察期:</strong> {result['obs_date_range'][0]} 到 {result['obs_date_range'][1] if len(result['obs_date_range']) > 1 and result['obs_date_range'][0] != result['obs_date_range'][1] else result['obs_date_range'][0]}</p>
            <p><strong>对比期:</strong> {result['cmp_date_range'][0]} 到 {result['cmp_date_range'][1] if len(result['cmp_date_range']) > 1 and result['cmp_date_range'][0] != result['cmp_date_range'][1] else result['cmp_date_range'][0]}</p>
        """
        
        # 添加数据表
        html_content += "<h3>详细数据</h3>\n"
        html_content += df_result.to_html(index=False, table_id=f"table_{i}", classes="data-table")
        html_content += "</div>\n"
    
    html_content += """
    </body>
    </html>
    """
    
    return html_content

def generate_excel_report(analysis_results, analysis_config):
    """生成Excel格式的完整分析报告"""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # 创建配置信息DataFrame
        config_data = {
            '配置项': ['观察期类型', '观察期范围', '对比期类型', '对比期范围'],
            '值': [
                analysis_config.get('obs_period_type', '未设置'),
                f"{analysis_config.get('obs_date_range', ['未设置'])[0]} 到 {analysis_config.get('obs_date_range', ['未设置', '未设置'])[1]}",
                analysis_config.get('cmp_period_type', '未设置'),
                f"{analysis_config.get('cmp_date_range', ['未设置'])[0]} 到 {analysis_config.get('cmp_date_range', ['未设置', '未设置'])[1]}"
            ]
        }
        config_df = pd.DataFrame(config_data)
        config_df.to_excel(writer, sheet_name='分析配置', index=False)
        
        # 为每个分析结果创建一个工作表
        for i, result in enumerate(analysis_results):
            sheet_name = f"{result['metric_name'][:15]}_{result['dimension'][:15]}"  # 限制工作表名称长度
            df_result = pd.DataFrame(result['data'])
            df_result.to_excel(writer, sheet_name=sheet_name, index=False)
    
    buffer.seek(0)
    return buffer

if __name__ == "__main__":
    main()