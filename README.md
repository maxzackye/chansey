# 从gitlab上import函数包


```python
from pokemon.chansey import var_diagnosis
```

# 起spark并获取数据


```python
spark = create_spark()

spark
```


```python
sdf = spark.sql('''select * from collection.collection_f_xyh_ai_test_04_03
''')

df  = sdf.toPandas() ###此处也可以使用spark_save_csv方法将查询结果存储到本地目录，再用pandas.read_csv读取
```

# 从Excel文件读取数据（新增功能）

```python
# 从Excel文件读取数据
import chansey

# 方式1：使用ExcelReader类
excel_reader = chansey.ExcelReader('your_excel_file.xlsx', ['Sheet1', 'Sheet2'])
multi_dataset = excel_reader.to_multi_dataset({
    'Sheet1': 'date_column',
    'Sheet2': 'date_column'
})

# 方式2：使用便捷函数
multi_dataset = chansey.read_excel_for_chansey(
    'your_excel_file.xlsx', 
    ['Sheet1', 'Sheet2'], 
    {'Sheet1': 'date_column', 'Sheet2': 'date_column'}
)

# 如果所有工作表都有相同的时间列名
multi_dataset = chansey.read_excel_for_chansey(
    'your_excel_file.xlsx', 
    None,  # 读取所有工作表
    'date_column'  # 所有工作表共用的时间列名
)
```

# Web应用程序（新增功能）

本项目包含两个Web应用程序，允许用户通过网页界面上传Excel文件、配置字段并查看分析结果。

## Flask Web应用程序

### 启动Flask应用程序

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动应用
python web_app.py
```

### 使用说明

1. 访问 http://localhost:5000
2. 上传包含业务数据的Excel文件
3. 选择工作表和字段配置：
   - 日期字段：用于区分观察期和对比期
   - 维度字段：用于分组分析的字段（如产品、地区、渠道等）
   - 指标字段：需要分析的数值型字段（如销售额、访问量等）
4. 预览数据确认无误
5. 执行分析并查看结果
6. 可以点击"查看瀑布图"按钮查看可视化结果

## Streamlit Web应用程序（推荐）

### 启动Streamlit应用程序

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动应用
streamlit run streamlit_app.py
```

### 使用说明

1. 在浏览器中打开Streamlit应用
2. 上传包含业务数据的Excel文件
3. 选择工作表
4. 配置字段：
   - 日期字段：用于区分观察期和对比期
   - 维度字段：用于分组分析的字段（可多选）
   - 指标字段：支持两种类型
     - 数值指标：直接使用的数值字段
     - 比例指标：需要指定分子和分母字段
5. 执行数据分析
6. 查看可视化结果

### 功能特点

- 支持Excel文件上传（.xlsx, .xls）
- 可视化字段配置界面
- 支持数值指标和比例指标
- 数据预览功能
- 多指标分析结果展示
- 交互式图表可视化（瀑布图、条形图等）
- 结果导出为CSV文件

# 实例化分诊器类


```python
vd = var_diagnosis(df,
                   obs_rng = ['2021-07-05','2021-07-05'],
                   cmp_rng = ['2021-07-04','2021-07-04'],
                   base_col = 'rec_dte')

'''
df:数据集
base_col: 区分对比期和观察期的筛选列
obs_rng: 观察集的筛选条件，用list表示，如果筛选列是日期格式，可以用[开始日期，结束日期]的形式输入
cmp_rng: 对比集的筛选条件，用list表示，如果筛选列是日期格式，可以用[开始日期，结束日期]的形式输入
'''
```

# 漏斗拆解方法


```python
mltplr_list = [] 
flow_list = ['ovd_prc_1','ovd_prc_2','ovd_prc_3','ovd_prc_4','ovd_prc_5'] 
flt = {} 
summary_fa,ctb_df = vd.flow_analyse(mltplr_list,flow_list,flt = flt)

'''返回业务流拆解,用循环替代法实现
mltplr_list: 数据级中连乘因子列名list
flow_list: 数据集中待拆解的业务流环节列名list,必须按照顺序排列
flt: 数据筛选条件，用字典表示，字典中每个key代表带筛选字段，key对应的value类型为list，代表需要保留的值，例如：{'lv1_cls':['今日头条','朋友圈']}
'''
```


```python
vd.waterfall_flow(summary_fa,ctb_df)
'''画图功能，生成瀑布图（仅在jupyter预发布环境可用）
summary_fa: flow_analyse输出变量一
ctb_df: flow_analyse输出变量二
'''
```

# 维度下钻方法


```python
label = ['ovd_prc_3','ovd_prc_2']
dim_col_list = ['que_nam','flw_nam']
flt = {}

summary,delta_drilldown,df_gini = vd.auto_drilldown(label,dim_col_list,flt = flt)

'''返回下钻维度拆解,用权重拆分法实现,并展示不同维度的基尼系数排序
label: 待拆分的指标, 用[分子,分母]的形式输入
dim_col_list:拆分维度列名list
flt: 数据筛选条件，用字典表示，字典中每个key代表带筛选字段，key对应的value类型为list，代表需要保留的值，例如：{'lv1_cls':['今日头条','朋友圈']}
'''
```


```python
##筛选结果数据集##

summary = summary[summary['dim_col'] == 'que_nam']

delta_drilldown = delta_drilldown[delta_drilldown['dim_col'] == 'que_nam']
```


```python
vd.waterfall_drilldown(summary,delta_drilldown,split = False)
'''画图功能，生成瀑布图（仅在jupyter预发布环境可用）
summary: auto_drilldown输出一
delta_drilldown: auto_drilldown输出二
split: 是否拆分rate贡献与weight贡献
'''
```