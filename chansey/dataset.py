"""
定义数据集
数据集包含四个元素：
    一、DataFrame
    二、时间字段名
    三、观察期时间
    四、对比期时间
"""
import pandas as pd
from typing import Optional, List, Union, Dict, Tuple, Any
from datetime import timedelta
from datetime import datetime
import time
import copy

class Dataset:
    """
    定义观察期和对比期的数据集
    """
    def __init__(
        self,
        df: pd.DataFrame,
        period_colname: str,
    ):
        """
        df: DataFrame
        period_colname: 时期列名
        obs_period: 观察期
        cmp_period: 对比期
        """
        assert isinstance(df, pd.DataFrame)
        assert period_colname in df

        self.df = df
        self.period_colname = period_colname

    def sub_dateset(self, dimension, value):
        """
        根据维度筛选子数据集
        dimension:
        value:
        """
        raise NotImplementedError

    def getBetweenDay(self,begin_date,end_date):
        '''返回两个日期之间所有日期的list
        begin_date:起始日期
        end_date:结束日期
        example: getBetweenDay('2021-01-01','2021-01-05')
        return: ['2021-01-01'.'2021-01-02','2021-01-03','2021-01-04','2021-01-05']
        '''
        date_list = []
        while begin_date <= end_date:
            date_str = begin_date.strftime("%Y-%m-%d")
            date_list.append(date_str)
            begin_date += timedelta(days=1)
        return date_list
    
    
    def filter_by_period(self, obs_period, cmp_period):
        '''
        筛选数据集，按照观察对比日期范围和flt,输出筛选后数据集,和观察集，对比集筛选条件（因为外层函数需要用到）
        '''
        df_new = copy.deepcopy(self.df)
        
        df_new[self.period_colname] = df_new[self.period_colname].astype(str)
        
        try:
            obs_stt_date = obs_period[0]
            obs_end_date = obs_period[-1]
            obs_stt_date = datetime.strptime(obs_stt_date,"%Y-%m-%d")
            obs_end_date = datetime.strptime(obs_end_date,"%Y-%m-%d")
    
            cmp_stt_date = cmp_period[0]
            cmp_end_date = cmp_period[-1]
            cmp_stt_date = datetime.strptime(cmp_stt_date,"%Y-%m-%d")
            cmp_end_date = datetime.strptime(cmp_end_date,"%Y-%m-%d")                                                               
            obs_period = self.getBetweenDay(obs_stt_date,obs_end_date)
            cmp_period = self.getBetweenDay(cmp_stt_date,cmp_end_date)
        except:
            obs_period = [str(x) for x in obs_period]
            cmp_period = [str(x) for x in cmp_period]
    
        df_new = df_new[(df_new[self.period_colname].isin(obs_period + cmp_period) == True)].reset_index(drop = True)
        
        obs_idx = df_new[df_new[self.period_colname].isin(obs_period) == True].index
        cmp_idx = df_new[df_new[self.period_colname].isin(cmp_period) == True].index
    
        df_new['grp'] = None
        
        df_new.loc[obs_idx,'grp'] = 'obs'
        df_new.loc[cmp_idx,'grp'] = 'cmp'
        
        return df_new, obs_period, cmp_period
    

class MultiDataset:
    """
    定义包含多个数据集集合
    """

    def __init__(self, dataset_dict: Dict[str, Tuple[pd.DataFrame, str]]):
        """
        dataset_dict: 字典形式的多个数据集，键为数据集标识符，值为 (DataFrame, period_colname) 元组
        """
        self.datasets = {}
        for key, dataset_obj  in dataset_dict.items():
            if dataset_obj is not None:
                df = dataset_obj[0]
                period_colname = dataset_obj[1]
                assert isinstance(df, pd.DataFrame)
                assert period_colname in df.columns
                self.datasets[key] = Dataset(df, period_colname)
            else:
                self.datasets[key] = None


    def sub_datasets_by_dimension(self, dimension: Union[str, List[str]], values: List[Any]) -> Dict[str, 'MultiDataset']:
        """
        根据指定维度将所有的数据集拆分成多个子数据集，每个子数据集的key用对应维度枚举值来命名。

        dimension: 要筛选的维度名称
        values: 筛选维度的取值列表

        return: 一个字典，键是新的数据集标识符（维度值），值是对应拆分后的 multidataset
        """
        # 当dimension是字符串格式时
        if isinstance(dimension, str):
            dimension = [dimension]
        cross_dim = '&'.join([str(i) for i in dimension])
        sub_datasets = {}
        for value in values:
            dict_i = {}
            for key, dataset in self.datasets.items():
                if dataset is not None and all([dim in dataset.df.columns for dim in dimension]):
                    for dim in dimension:
                        dataset.df[dim] = dataset.df[dim].astype('str')
                    df = dataset.df.copy()

                    if len(dimension)>1:
                        df[cross_dim] = df.apply(lambda row: {i: row[i] for i in dimension}, axis=1)
                        df[cross_dim] = df[cross_dim].astype(str)
                        sub_df = df[df[cross_dim] == str(value)]
                    else:
                        sub_df = df[df[dimension[0]] == str(value)]
                        
                    if not sub_df.empty:
                        dict_i[key] = (sub_df, dataset.period_colname)
                    else:
                        dict_i[key] = None
                elif dataset is not None and all([dim in dataset.df.columns for dim in dimension])==False:
                    dict_i[key] = (dataset.df, dataset.period_colname)
                else:
                    dict_i[key] = None
                
            sub_datasets[value] = MultiDataset(dict_i)
        return sub_datasets
        
        
