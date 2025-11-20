import chansey
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


"""
# 1. NumberMetric：数值类
chansey.NumberMetric(
        name: str, # 指标名（与数据集的字段同名）
        dataset: Optional[MultiDataset]=None, # multidataset数据集,自动继承父节点dataset
        dataset_key: str=None, # 数据集名称key,自动继承父节点dataset_key
        dimension: Optional[str]=None, # 拆解的维度
        children: Optional[Dict[str, MetricTreeNode]]=None # 子节点, key是维度的值, value是其他指标
        )

# 2. RatioMetric：比率类
chansey.RatioMetric(
    name: str,
    numerator: Metric, # 分子指标
    denominator: Metric, # 分母指标
    dataset: Optional[MultiDataset]=None,
    dataset_key: str=None,
    dimension: Optional[str]=None,
    children: Optional[Dict[str, MetricTreeNode]]=None
    )

# 3. Funnel：漏斗类
chansey.Funnel(
    name: str, # 漏斗整体指标的名字
    dataset: Optional[MultiDataset]=None,
    dataset_key: str=None,
    children: List[MetricTreeNode]=None # 漏斗指标list, 需要前一项的分母必须和后一项相同
    )

"""


def test():
    # 准备数据 - 从CSV读取（原方式）
    df1 = pd.read_csv('./chansey/df1.csv', encoding='gbk')
    df2 = pd.read_csv('./chansey/df2.csv', encoding='gbk')
    dict1 = chansey.MultiDataset({'df1': (df1, '月份'), 'df2': (df2, '月份')})

    # 准备数据 - 从Excel读取（新方式）
    # 假设你有一个Excel文件，其中包含df1和df2两个工作表
    # excel_reader = chansey.ExcelReader('your_excel_file.xlsx', ['df1', 'df2'])
    # dict1_from_excel = excel_reader.to_multi_dataset('月份')
    
    # 或者使用便捷函数
    # dict1_from_excel = chansey.read_excel_for_chansey('your_excel_file.xlsx', ['df1', 'df2'], '月份')

    fund_list = df2['资方'].drop_duplicates().to_list()
    scene_list = df1['场景'].drop_duplicates().to_list()

    # 构造拆解树
    ## 定义各资方下漏斗分析节点的函数
    def flow1(fund):
        flow1 = chansey.Funnel(name='总动支金额',
                               children=[chansey.NumberMetric(name='总授信额度')],
                               )
        flow1.children.extend(
            chansey.generate_metrics_by_funnel(
                number_names=['总授信额度', '总发起金额', '总动支金额'],
                ratio_names=['金额发起率', '金额通过率']
            )[1]
        )
        return flow1

    ## 构造第一层维度拆解（by资方）和第二层漏斗拆解
    business_formula = chansey.NumberMetric(name='总动支金额', dimension='资方'
                                            , dataset=dict1, dataset_key='df2'
                                            , children={str(key): flow1(key) for key in fund_list}
                                            )

    ## 构造第三层（by场景）维度拆解
    for fund in fund_list:
        business_formula.children[str(fund)].children[0] = chansey.NumberMetric(name='总授信额度'
                                                                                , dimension='场景'
                                                                                , dataset_key='df1'  # 第三层时更换了数据集key
                                                                                , children={
                str(key): chansey.NumberMetric(name='总授信额度') for key in scene_list}
                                                                                )

    # 获取下钻结果
    obs_period = ['2023-12']
    cmp_period = ['2023-11']
    output = business_formula.drilldown(obs_period, cmp_period, depth=4)
    """
    obs_period：观察期,枚举值list or ['开始日期','结束日期']
    cmp_period：对比期,枚举值list or ['开始日期','结束日期']
    depth：拆解最大层数，默认无穷大
    """

    return output


# 新增示例：展示如何使用Excel读取功能
def test_excel():
    """
    演示如何使用Excel读取功能
    注意：这需要你有一个实际的Excel文件才能运行
    """
    try:
        # 方式1：使用ExcelReader类
        excel_reader = chansey.ExcelReader('./example.xlsx', ['Sheet1', 'Sheet2'])
        multi_dataset = excel_reader.to_multi_dataset({
            'Sheet1': '月份',
            'Sheet2': '月份'
        })
        
        print("成功读取Excel文件:")
        print("工作表:", excel_reader.get_sheet_names())
        print("Sheet1列名:", excel_reader.list_columns('Sheet1'))
        print("Sheet2列名:", excel_reader.list_columns('Sheet2'))
        
        return multi_dataset
    except FileNotFoundError:
        print("示例Excel文件不存在，请创建一个Excel文件进行测试")
        return None
    except Exception as e:
        print(f"读取Excel文件时发生错误: {e}")
        return None


if __name__ == '__main__':
    test()