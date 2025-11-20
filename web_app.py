"""
Web界面应用程序
允许用户上传Excel文件，选择字段并展示分析结果
"""

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
import pandas as pd
import os
import json
from werkzeug.utils import secure_filename
import plotly
import plotly.graph_objs as go
import numpy as np

# 导入chansey模块
import chansey

app = Flask(__name__)
app.secret_key = 'chansey_secret_key_2023'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 最大16MB上传限制

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_sheet_names(filepath):
    """获取Excel文件中的工作表名称"""
    try:
        excel_file = pd.ExcelFile(filepath)
        return excel_file.sheet_names
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

def get_columns(filepath, sheet_name):
    """获取指定工作表的列名"""
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name, nrows=0)
        return list(df.columns)
    except Exception as e:
        print(f"Error reading sheet columns: {e}")
        return []

@app.route('/')
def index():
    """主页 - 文件上传页面"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    # 检查是否有文件在请求中
    if 'file' not in request.files:
        flash('没有选择文件')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('没有选择文件')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 获取工作表名称
        sheet_names = get_sheet_names(filepath)
        
        # 将文件路径保存到会话中
        session_data = {
            'filepath': filepath,
            'filename': filename
        }
        
        # 保存会话数据到临时文件（实际应用中应使用Flask-Session）
        session_file = os.path.join(app.config['UPLOAD_FOLDER'], 'session.json')
        with open(session_file, 'w') as f:
            json.dump(session_data, f)
        
        return render_template('configure.html', 
                             filename=filename, 
                             sheet_names=sheet_names)
    else:
        flash('只支持Excel文件格式 (xlsx, xls)')
        return redirect(url_for('index'))

@app.route('/columns/<sheet_name>')
def get_sheet_columns(sheet_name):
    """获取指定工作表的列名（API端点）"""
    # 读取会话数据
    session_file = os.path.join(app.config['UPLOAD_FOLDER'], 'session.json')
    try:
        with open(session_file, 'r') as f:
            session_data = json.load(f)
    except FileNotFoundError:
        return jsonify({'error': '会话数据丢失'}), 400
    
    filepath = session_data['filepath']
    columns = get_columns(filepath, sheet_name)
    
    return jsonify({'columns': columns})

@app.route('/configure', methods=['GET', 'POST'])
def configure_data():
    """配置数据字段"""
    # 读取会话数据
    session_file = os.path.join(app.config['UPLOAD_FOLDER'], 'session.json')
    try:
        with open(session_file, 'r') as f:
            session_data = json.load(f)
    except FileNotFoundError:
        flash('会话数据丢失，请重新上传文件')
        return redirect(url_for('index'))
    
    filepath = session_data['filepath']
    
    if request.method == 'POST':
        # 获取表单数据
        sheet_name = request.form.get('sheet_name')
        date_column = request.form.get('date_column')
        dimension_columns = request.form.getlist('dimension_columns')
        metric_columns = request.form.getlist('metric_columns')
        
        # 验证必需字段
        if not sheet_name or not date_column or not dimension_columns or not metric_columns:
            flash('请完整填写所有配置项')
            # 重新获取工作表名称以重新显示页面
            sheet_names = get_sheet_names(filepath)
            return render_template('configure.html', 
                                 filename=session_data['filename'], 
                                 sheet_names=sheet_names)
        
        # 读取数据
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
        except Exception as e:
            flash(f'读取数据时出错: {str(e)}')
            return redirect(url_for('index'))
        
        # 保存配置信息
        config_data = {
            'sheet_name': sheet_name,
            'date_column': date_column,
            'dimension_columns': dimension_columns,
            'metric_columns': metric_columns
        }
        config_file = os.path.join(app.config['UPLOAD_FOLDER'], 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # 保存会话数据
        session_data['config_file'] = config_file
        with open(session_file, 'w') as f:
            json.dump(session_data, f)
        
        # 显示数据预览
        preview_data = df.head(10).to_dict(orient='records')
        columns = list(df.columns)
        
        return render_template('preview.html', 
                             data=preview_data, 
                             columns=columns,
                             config=config_data)
    else:
        # GET请求，显示配置页面
        sheet_names = get_sheet_names(filepath)
        return render_template('configure.html',
                             filename=session_data['filename'],
                             sheet_names=sheet_names)

@app.route('/analyze')
def analyze():
    """执行分析并展示结果"""
    # 读取会话数据
    session_file = os.path.join(app.config['UPLOAD_FOLDER'], 'session.json')
    config_file = os.path.join(app.config['UPLOAD_FOLDER'], 'config.json')
    
    try:
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        with open(config_file, 'r') as f:
            config_data = json.load(f)
    except FileNotFoundError:
        flash('数据丢失，请重新开始')
        return redirect(url_for('index'))
    
    filepath = session_data['filepath']
    sheet_name = config_data['sheet_name']
    date_column = config_data['date_column']
    dimension_columns = config_data['dimension_columns']
    metric_columns = config_data['metric_columns']
    
    # 读取数据
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    except Exception as e:
        flash(f'读取数据时出错: {str(e)}')
        return redirect(url_for('index'))
    
    # 确保日期列是字符串格式
    df[date_column] = df[date_column].astype(str)
    
    # 获取唯一日期
    unique_dates = sorted(df[date_column].unique())
    
    # 如果日期少于2个，无法进行对比分析
    if len(unique_dates) < 2:
        flash('数据中至少需要包含两个不同的日期才能进行对比分析')
        return redirect(url_for('index'))
    
    # 默认使用最后两个日期作为观察期和对比期
    obs_date = unique_dates[-1]
    cmp_date = unique_dates[-2]
    
    # 创建var_diagnosis实例
    vd = chansey.var_diagnosis(df, date_column, [obs_date, obs_date], [cmp_date, cmp_date])
    
    # 准备分析结果
    analysis_results = []
    
    # 对每个指标进行分析
    for metric in metric_columns:
        try:
            # 确保指标列是数值类型
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
            
            # 按维度分组计算指标值
            obs_data = df[df[date_column] == obs_date]
            cmp_data = df[df[date_column] == cmp_date]
            
            obs_summary = obs_data.groupby(dimension_columns)[metric].sum().reset_index()
            cmp_summary = cmp_data.groupby(dimension_columns)[metric].sum().reset_index()
            
            # 合并数据
            merged = pd.merge(obs_summary, cmp_summary, on=dimension_columns, 
                            suffixes=('_obs', '_cmp'), how='outer').fillna(0)
            
            # 计算变化量和变化率
            merged['delta'] = merged[f'{metric}_obs'] - merged[f'{metric}_cmp']
            merged['delta_pct'] = np.where(merged[f'{metric}_cmp'] != 0,
                                         (merged['delta'] / merged[f'{metric}_cmp']) * 100,
                                         np.inf)
            
            # 添加到结果中
            result = {
                'metric': metric,
                'data': merged.round(2).to_dict(orient='records'),
                'obs_date': obs_date,
                'cmp_date': cmp_date
            }
            analysis_results.append(result)
            
        except Exception as e:
            flash(f'分析指标 {metric} 时出错: {str(e)}')
    
    return render_template('results.html', 
                         results=analysis_results,
                         obs_date=obs_date,
                         cmp_date=cmp_date)

@app.route('/waterfall/<metric>')
def waterfall_chart(metric):
    """生成瀑布图"""
    # 读取会话和配置数据
    session_file = os.path.join(app.config['UPLOAD_FOLDER'], 'session.json')
    config_file = os.path.join(app.config['UPLOAD_FOLDER'], 'config.json')
    
    try:
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        with open(config_file, 'r') as f:
            config_data = json.load(f)
    except FileNotFoundError:
        flash('数据丢失，请重新开始')
        return redirect(url_for('index'))
    
    filepath = session_data['filepath']
    sheet_name = config_data['sheet_name']
    date_column = config_data['date_column']
    dimension_columns = config_data['dimension_columns']
    
    # 读取数据
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    except Exception as e:
        flash(f'读取数据时出错: {str(e)}')
        return redirect(url_for('index'))
    
    # 确保日期列是字符串格式
    df[date_column] = df[date_column].astype(str)
    
    # 获取唯一日期
    unique_dates = sorted(df[date_column].unique())
    obs_date = unique_dates[-1]
    cmp_date = unique_dates[-2]
    
    # 按维度分组计算指标值
    obs_data = df[df[date_column] == obs_date]
    cmp_data = df[df[date_column] == cmp_date]
    
    obs_summary = obs_data.groupby(dimension_columns)[metric].sum().reset_index()
    cmp_summary = cmp_data.groupby(dimension_columns)[metric].sum().reset_index()
    
    # 合并数据
    merged = pd.merge(obs_summary, cmp_summary, on=dimension_columns, 
                    suffixes=('_obs', '_cmp'), how='outer').fillna(0)
    
    # 计算变化量
    merged['delta'] = merged[f'{metric}_obs'] - merged[f'{metric}_cmp']
    
    # 按变化量排序
    merged = merged.sort_values('delta', ascending=False)
    
    # 创建瀑布图数据
    obs_total = merged[f'{metric}_obs'].sum()
    cmp_total = merged[f'{metric}_cmp'].sum()
    total_delta = obs_total - cmp_total
    
    # 准备绘图数据
    x_data = ['对比期'] + merged[dimension_columns[0]].tolist() + ['观察期']
    y_data = [cmp_total] + merged['delta'].tolist() + [obs_total]
    
    # 创建瀑布图
    fig = go.Figure(go.Waterfall(
        name="变化贡献",
        orientation="v",
        measure=["absolute"] + ["relative"] * len(merged) + ["total"],
        x=x_data,
        textposition="outside",
        text=[f"{y:,.2f}" for y in y_data],
        y=y_data,
        connector={"line": {"color": "rgb(63, 63, 63)"}}
    ))
    
    fig.update_layout(
        title=f"{metric} 指标变化瀑布图",
        showlegend=True,
        waterfallgap=0.1
    )
    
    # 转换为JSON格式以便在模板中使用
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('waterfall.html', 
                         plot=graphJSON, 
                         metric=metric,
                         obs_date=obs_date,
                         cmp_date=cmp_date)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)