"""
将指标拆解的逻辑定义成一颗指标树
树节点分为两类：
    一、DimensionNode: 维度节点, 实现维度拆解的功能
    二、FunnelNode: 漏斗节点, 实现漏斗拆解的功能
"""
from typing import Optional, List, Union, Dict
from functools import wraps
from random import choice
import pandas as pd
import numpy as np
from itertools import product
import copy

from .dataset import Dataset,MultiDataset


class MetricTreeNode:
    """指标拆解基类"""
    def __init__(
        self,
        name: str,
        dataset: Optional[MultiDataset]=None,
        dataset_key: str=None,
        children: Optional[Union[list, dict]] = None
    ):
        """
        
        name: 节点名
        dataset: 数据集
        """
        assert isinstance(name, str)
        if dataset is not None:
            assert isinstance(dataset, MultiDataset)
            for key, value in dataset.datasets.items():
                assert isinstance(value, Dataset)

        self.name = name
        self.dataset = dataset
        self.dataset_key = dataset_key
        self.children = children

    @property
    def children(self):
        return self._children
    
    @children.setter
    def children(self, children):
        raise NotImplemented

    def drilldown(self, obs_period, cmp_period, depth=np.inf, dataset=None) -> dict:
        """
        下钻
        obs_period: 观察期
        cmp_period: 对比期
        depth: 下钻深度
        dataset: 数据集, 但是优先级低于self.dataset
        """
        raise NotImplementedError
    
    def repr(self):
        return f'{self.__class__.__name__}({self.name})'
    
    def __repr__(self):
        """定义本节点和子节点print的内容"""
        output = [self.repr()]
        if isinstance(self.children, list):
            for child in self.children:
                output.extend('  ' + x for x in repr(child).split('\n'))

        if isinstance(self.children, dict):
            for name, child in self.children.items():
                temp = f'{name}: {repr(child)}'
                output.extend('  ' + x for x in temp.split('\n'))

        return '\n'.join(output)
    
    @staticmethod
    def flatten_dataframe(df, column_name='nodes'):
        '''
        解开嵌套df并拼接成一个大df
        '''
        result_df = pd.DataFrame()
        for i in range(len(df)):
            if not isinstance(df[column_name][i], pd.DataFrame):
                # 如果某行值不是DataFrame，则与原来的dataframe合并
                result_df = pd.concat([result_df,df.loc[[i],:]])
            else:
                # 如果是DataFrame，先保留parent_node_name,再回调函数
                # parent_node_name = df.loc[i,'parent_node_name']
                df[column_name][i].loc[df[column_name][i]['parent_node_name'].isna(),'parent_node_name'] = df.loc[i,'parent_node_name']
                result_df = pd.concat([result_df,Metric.flatten_dataframe(df[column_name][i],column_name)])
        return result_df
    
    @staticmethod
    def add_color_column(df, category_column):
        # 获取唯一的类别列表
        categories = df[category_column].unique()
        colors = ['red', 'brown', 'hotpink','orange','purple','skyblue','tomato','mediumblue','blueviolet','goldenrod','darkgreen','forestgreen']
        # 随机生成颜色列表
        colors = [choice(colors) for _ in range(len(categories))]
        colors[0] = 'blue'
        # 创建类别和对应颜色的字典
        color_dict = dict(zip(categories, colors))
        # 使用map()函数将类别映射为颜色，并添加为新列'Color'
        df['color'] = df[category_column].map(color_dict)
        return df
    
    def get_name_df(self,i):
        '''
        获取每个节点名称，类型，层数及父节点名称，返回一个嵌套df
        '''
        names = pd.DataFrame(columns=['name','nodes','level','parent_node_name'])
        if self is None:
            return names
        ## children类型是字典
        if isinstance(self.children, dict):
            names.loc[len(names),:] = [self.name,list(self.children.keys()),i,None]
            for k, v in self.children.items():
                if v.children is not None:
                    names.loc[len(names),:] = [v.name,v.get_name_df(i+1),i+1,k]
        ## children类型是List
        elif isinstance(self.children, list):
            names.loc[len(names),:] = [self.name,tuple([f.name for f in self.children]),i,None]
            for funnel_child in self.children:
                if funnel_child.children is not None:
                    if isinstance(self, NumberMetric):
                        names.loc[len(names),:] = [funnel_child.name, funnel_child.get_name_df(i+1), i+1,self.name]
                    else:
                        names.loc[len(names),:] = [funnel_child.name, funnel_child.get_name_df(i+1), i+1,funnel_child.name]
        return names
    

    def to_dot(self, i=0, rankdir= 'TB'):
        names_df = self.get_name_df(i)
        # 调用flatten_dataframe函数进行展开操作
        df = self.flatten_dataframe(names_df,'nodes')
        df = df.sort_values('level')
        df.reset_index(drop=True, inplace= True)
        df.loc[0,'parent_node_name'] = df['name'][0]
        ## 加一列颜色
        df = self.add_color_column(df, 'name')
        ## 根节点和第一行单独画
        dot_string = f"rankdir = {rankdir};\nnode [shape = record,height=.1,fontname = \"KaiTi\"];\nroot_node[label=\"<f0> {df['name'][0]}\",color=blue,style=filled, fontcolor=white];\n"
        node = df['nodes'][0]
        color = df['color'][0]
        level = df['level'][0]
        if isinstance(node, list):
            for o in range(len(node)):
                ## 先画节点
                node_name = f"nd_{level}_0_{o}"
                dot_string += f"{node_name}[label=\"<f0> {node[o]}\",color = {color}];\n"
                ## 再画边
                dot_string += f"root_node -> {node_name}:f0;\n"
        else:## 漏斗，元组类型，整体是一个元素
            ## 先画节点
            node_name = f"nd_{level}_0_0"
            dot_string += f"{node_name}[label="
            for l in range(len(node)):
                if l == 0:
                    dot_string += f"\"{{<f{l}> {node[l]}"
                else:
                    dot_string += f"|<f{l}> {node[l]}"
            dot_string += f"}}\", color={color}, style=dashed];\n"
            ## 再画边
            dot_string += f"root_node -> {node_name}[dir=none];\n"
        ## 剩余的for循环画
        levels = list(df['level'].value_counts().sort_index().index)
        pairs = [(levels[i], levels[i+1]) for i in range(len(levels)-1)]
        ## 根据分组循环，每次只画df_level_2
        for pair in pairs:
            df_level_1 = df[df['level']==pair[0]]
            df_level_2 = df[df['level']==pair[1]]
            df_level_1.reset_index(drop= True, inplace= True)
            df_level_2.reset_index(drop= True, inplace= True)
            for i in range(len(df_level_2)):
                node = df_level_2['nodes'][i]
                parent_node_name = df_level_2['parent_node_name'][i]
                color = df_level_2['color'][i]
                level = df_level_2['level'][i]
                ## 维度，list类型，每个元素一个节点
                if isinstance(node, list):
                    for j in range(len(node)):
                        ## 先画节点
                        node_name = f"nd_{level}_{i}_{j}"
                        dot_string += f"{node_name}[label=\"<f0> {node[j]}\",color = {color}];\n"
                        ## 再画边,找到parent_node_name在df_level_1的哪个位置
                        for l in range(len(df_level_1)):
                            try:
                                loc = df_level_1['nodes'][l].index(parent_node_name)
                                name_loc = l
                            except:
                                continue
                        if isinstance(df_level_1['nodes'][name_loc],list):
                            parent_node_name = f"nd_{df_level_1['level'][name_loc]}_{name_loc}_{loc}"
                            dot_string += f"{parent_node_name}:f0 -> {node_name}:f0;\n"
                        else:
                            parent_node_name = f"nd_{df_level_1['level'][name_loc]}_{name_loc}_0"
                            dot_string += f"{parent_node_name}:f{loc} -> {node_name}:f0;\n"
                        
                else:## 漏斗，元组类型，整体是一个元素
                    ## 先画节点
                    node_name = f"nd_{level}_{i}_0"
                    dot_string += f"{node_name}[label="
                    for j in range(len(node)):
                        if j == 0:
                            dot_string += f"\"{{<f{j}> {node[j]}"
                        else:
                            dot_string += f"|<f{j}> {node[j]}"
                    dot_string += f"}}\", color={color}, style=dashed];\n"
                    ## 再画边,漏斗的parent不会是漏斗,找到parent_node_name在df_level_1的哪个位置
                    for l in range(len(df_level_1)):
                        try:
                            loc = df_level_1['nodes'][l].index(parent_node_name)
                            name_loc = l
                        except:
                            continue
                    parent_node_name = f"nd_{df_level_1['level'][name_loc]}_{name_loc}_{loc}"
                    dot_string += f"{parent_node_name}:f0 -> {node_name}[dir=none];\n"
            
        return "digraph g {\n" + dot_string + "}"




    @staticmethod
    def flatten_dataframe(df, column_name='nodes'):
        '''
        解开嵌套df并拼接成一个大df
        '''
        result_df = pd.DataFrame()
        for i in range(len(df)):
            if not isinstance(df[column_name][i], pd.DataFrame):
                # 如果某行值不是DataFrame，则与原来的dataframe合并
                result_df = pd.concat([result_df,df.loc[[i],:]])
            else:
                # 如果是DataFrame，先保留parent_node_name,再回调函数
                # parent_node_name = df.loc[i,'parent_node_name']
                df[column_name][i].loc[df[column_name][i]['parent_node_name'].isna(),'parent_node_name'] = df.loc[i,'parent_node_name']
                result_df = pd.concat([result_df,Metric.flatten_dataframe(df[column_name][i],column_name)])
        return result_df
    
    @staticmethod
    def add_color_column(df, category_column):
        # 获取唯一的类别列表
        categories = df[category_column].unique()
        colors = ['red', 'brown', 'hotpink','orange','purple','skyblue','tomato','mediumblue','blueviolet','goldenrod','darkgreen','forestgreen']
        # 随机生成颜色列表
        colors = [choice(colors) for _ in range(len(categories))]
        colors[0] = 'blue'
        # 创建类别和对应颜色的字典
        color_dict = dict(zip(categories, colors))
        # 使用map()函数将类别映射为颜色，并添加为新列'Color'
        df['color'] = df[category_column].map(color_dict)
        return df
    
    def get_name_df(self,i):
        '''
        获取每个节点名称，类型，层数及父节点名称，返回一个嵌套df
        '''
        names = pd.DataFrame(columns=['name','nodes','level','parent_node_name'])
        if self is None:
            return names
        ## children类型是字典
        if isinstance(self.children, dict):
            names.loc[len(names),:] = [self.name,list(self.children.keys()),i,None]
            for k, v in self.children.items():
                if v.children is not None:
                    names.loc[len(names),:] = [v.name,v.get_name_df(i+1),i+1,k]
        ## children类型是List
        elif isinstance(self.children, list):
            names.loc[len(names),:] = [self.name,tuple([f.name for f in self.children]),i,None]
            for funnel_child in self.children:
                if funnel_child.children is not None:
                    if isinstance(self, NumberMetric):
                        names.loc[len(names),:] = [funnel_child.name, funnel_child.get_name_df(i+1), i+1,self.name]
                    else:
                        names.loc[len(names),:] = [funnel_child.name, funnel_child.get_name_df(i+1), i+1,funnel_child.name]
        return names
    

    def to_dot(self, i=0, rankdir= 'TB'):
        names_df = self.get_name_df(i)
        # 调用flatten_dataframe函数进行展开操作
        df = self.flatten_dataframe(names_df,'nodes')
        df = df.sort_values('level')
        df.reset_index(drop=True, inplace= True)
        df.loc[0,'parent_node_name'] = df['name'][0]
        ## 加一列颜色
        df = self.add_color_column(df, 'name')
        ## 根节点和第一行单独画
        dot_string = f"rankdir = {rankdir};\nnode [shape = record,height=.1,fontname = \"KaiTi\"];\nroot_node[label=\"<f0> {df['name'][0]}\",color=blue,style=filled, fontcolor=white];\n"
        node = df['nodes'][0]
        color = df['color'][0]
        level = df['level'][0]
        if isinstance(node, list):
            for o in range(len(node)):
                ## 先画节点
                node_name = f"nd_{level}_0_{o}"
                dot_string += f"{node_name}[label=\"<f0> {node[o]}\",color = {color}];\n"
                ## 再画边
                dot_string += f"root_node -> {node_name}:f0;\n"
        else:## 漏斗，元组类型，整体是一个元素
            ## 先画节点
            node_name = f"nd_{level}_0_0"
            dot_string += f"{node_name}[label="
            for l in range(len(node)):
                if l == 0:
                    dot_string += f"\"{{<f{l}> {node[l]}"
                else:
                    dot_string += f"|<f{l}> {node[l]}"
            dot_string += f"}}\", color={color}, style=dashed];\n"
            ## 再画边
            dot_string += f"root_node -> {node_name}[dir=none];\n"
        ## 剩余的for循环画
        levels = list(df['level'].value_counts().sort_index().index)
        pairs = [(levels[i], levels[i+1]) for i in range(len(levels)-1)]
        ## 根据分组循环，每次只画df_level_2
        for pair in pairs:
            df_level_1 = df[df['level']==pair[0]]
            df_level_2 = df[df['level']==pair[1]]
            df_level_1.reset_index(drop= True, inplace= True)
            df_level_2.reset_index(drop= True, inplace= True)
            for i in range(len(df_level_2)):
                node = df_level_2['nodes'][i]
                parent_node_name = df_level_2['parent_node_name'][i]
                color = df_level_2['color'][i]
                level = df_level_2['level'][i]
                ## 维度，list类型，每个元素一个节点
                if isinstance(node, list):
                    for j in range(len(node)):
                        ## 先画节点
                        node_name = f"nd_{level}_{i}_{j}"
                        dot_string += f"{node_name}[label=\"<f0> {node[j]}\",color = {color}];\n"
                        ## 再画边,找到parent_node_name在df_level_1的哪个位置
                        for l in range(len(df_level_1)):
                            try:
                                loc = df_level_1['nodes'][l].index(parent_node_name)
                                name_loc = l
                            except:
                                continue
                        if isinstance(df_level_1['nodes'][name_loc],list):
                            parent_node_name = f"nd_{df_level_1['level'][name_loc]}_{name_loc}_{loc}"
                            dot_string += f"{parent_node_name}:f0 -> {node_name}:f0;\n"
                        else:
                            parent_node_name = f"nd_{df_level_1['level'][name_loc]}_{name_loc}_0"
                            dot_string += f"{parent_node_name}:f{loc} -> {node_name}:f0;\n"
                        
                else:## 漏斗，元组类型，整体是一个元素
                    ## 先画节点
                    node_name = f"nd_{level}_{i}_0"
                    dot_string += f"{node_name}[label="
                    for j in range(len(node)):
                        if j == 0:
                            dot_string += f"\"{{<f{j}> {node[j]}"
                        else:
                            dot_string += f"|<f{j}> {node[j]}"
                    dot_string += f"}}\", color={color}, style=dashed];\n"
                    ## 再画边,漏斗的parent不会是漏斗,找到parent_node_name在df_level_1的哪个位置
                    for l in range(len(df_level_1)):
                        try:
                            loc = df_level_1['nodes'][l].index(parent_node_name)
                            name_loc = l
                        except:
                            continue
                    parent_node_name = f"nd_{df_level_1['level'][name_loc]}_{name_loc}_{loc}"
                    dot_string += f"{parent_node_name}:f0 -> {node_name}[dir=none];\n"
            
        return "digraph g {\n" + dot_string + "}"
    
    @staticmethod
    def complete_data(df,dimension_dict):
        """返回补全所有维度笛卡尔积的dataframe
        dimension_dict: 需要笛卡尔积补全的字典{字段名1:[枚举值,...],字段名2:[枚举值,...]}
        """
        cartesian_product = list(product(*dimension_dict.values()))
        df_left = pd.DataFrame(cartesian_product, columns=dimension_dict.keys())
        df = pd.merge(df_left,df,how='left',on=list(dimension_dict.keys()))
        df = df.fillna(0)
        return df
    
    @staticmethod
    def fill_funnel_df(df, flow_list):
        empty_indices = []  # 存储需要填充且后面有非空值的环节索引
        for i in range(1, len(flow_list)-1):  # 遍历所有中间环节
            current_col = flow_list[i]
            # 找到该环节为空的所有行的索引
            null_rows = df[df[current_col].isnull()].index.tolist()
            if not null_rows:  # 如果当前环节没有空值，则继续下一个环节
                continue
            # 获取后面环节的列名
            next_col = flow_list[i+1:]
            # 检查该环节后面是否有非空指标
            non_zero_mask = ~df.loc[null_rows,next_col].isnull()
            # 如果有非空指标，标记该环节数据有问题
            if any(non_zero_mask.any().values):
                empty_indices.append(i)
            # 如果后面有非空指标，则将当前环节为空的所有行的值填充为上一个环节的数据
            df.loc[non_zero_mask.any(axis=1)[non_zero_mask.any(axis=1)].index,current_col] = df.loc[non_zero_mask.any(axis=1)[non_zero_mask.any(axis=1)].index, flow_list[i - 1]]
                    
        return df, empty_indices
    
    @staticmethod
    def get_dimension_value(df: pd.DataFrame,dimension: Union[str, List[str]]):
        """
        获取df中多个维度交叉的新维度的枚举值
        返回枚举值list，格式为json字符串
        """
        if isinstance(dimension,str):
            dimension = [dimension]
        if len(dimension)>1:
            for dim in dimension:
                df[dim] = df[dim].astype(str)
                df[dim] = df[dim].fillna('NA')
            cross_dimension_value_list = df[dimension].drop_duplicates().apply(lambda row:{i:row[i] for i in dimension}, axis = 1).to_list()
            cross_dimension_value_list = [str(i) for i in cross_dimension_value_list]
        else:
            dim = dimension[0]
            df[dim] = df[dim].astype(str)
            df[dim] = df[dim].fillna('NA')
            cross_dimension_value_list = df[dim].drop_duplicates().to_list()
        return cross_dimension_value_list
    
    @staticmethod
    def dimension_drilldown(df,name,label,dimension):
        '''返回下钻维度拆解,用权重拆分法实现
        df: 筛选出的观察期和对比期的数据
        name: 指标名string
        label: 待拆分的指标, 用[分母,分子]的形式输入
        dimension:拆分维度列名，可以是字符串，也可以是多个维度的列表
        '''   
        # 处理dimension和label     
        if isinstance(dimension, str):
            dimension = [dimension]
        for dim in dimension:
            df[dim] = df[dim].astype(str)
            df[dim] = df[dim].fillna('NA')
        cross_dim = '&'.join([str(i) for i in dimension])

        if len(dimension)>1:
            df[cross_dim] = df[dimension].apply(lambda row: {i: row[i] for i in dimension}, axis=1)
        else:
            df[cross_dim] = df[dimension[0]]
        df[cross_dim] = df[cross_dim].astype(str)

        dimension_list = df[cross_dim].unique().tolist()
        for c in label:
            try:
                df[c] = pd.to_numeric(df[c],errors='coerce')
                df[[c]] = df[[c]].fillna(0)
            except:
                raise Exception("维度拆解失败,指标名不在数据集中！请检查数据集是否有",c)
        df['label'] = df[label[-1]]
        df['cnt'] = df[label[0]]

        # 聚合并处理缺失维度
        summary = df.groupby(['grp',cross_dim])[['label','cnt']].sum().reset_index()
        summary = Metric.complete_data(summary,{'grp':['obs','cmp'],cross_dim:dimension_list})

        if len(label) != 1:
            summary['rate'] = summary['label']/summary['cnt']
        else:
            summary['rate'] = summary['label']
        
        # 开始计算delta（数值类和比率类两种情况）
        summary['weight'] = None
        summary['weight_value'] = None
        summary['weight_value_rate'] = None
        summary['rate'] = summary['rate'].fillna(0)
        summary = summary.sort_values(by = ['grp',cross_dim])
        summary = summary.reset_index(drop = True)
        summary_total = summary.groupby('grp')[['label','cnt']].sum()
        if len(label) != 1:
            summary_total['rate'] = summary_total['label']/summary_total['cnt']
        else:
            summary_total['rate'] = summary_total['label']

        idx_cmp = summary[summary['grp'] == 'cmp'].index
        idx_obs = summary[summary['grp'] == 'obs'].index
        summary.loc[idx_cmp,'weight'] = summary.loc[idx_cmp,'cnt']/summary.groupby('grp')['cnt'].sum()['cmp']
        summary.loc[idx_obs,'weight'] = summary.loc[idx_obs,'cnt']/summary.groupby('grp')['cnt'].sum()['obs']
        sum_temp = pd.merge(summary[summary['grp'] == 'obs'],summary[summary['grp'] == 'cmp'],on = cross_dim,how = 'inner')
        if len(label) != 1:
            sum_temp['delta_rate'] = sum_temp['weight_y']*(sum_temp['rate_x'] - sum_temp['rate_y'])
            sum_temp['delta_weight'] = (sum_temp['weight_x'] - sum_temp['weight_y'])*(sum_temp['rate_x'] - summary_total.loc['obs','rate'])
            sum_temp['delta'] = sum_temp['delta_rate'] + sum_temp['delta_weight']
        else:
            sum_temp['delta'] = sum_temp['label_x'] - sum_temp['label_y']
            sum_temp['delta_rate'] = sum_temp['delta']
            sum_temp['delta_weight'] = sum_temp['delta']
        
        # 保存信息
        delta_drilldown = sum_temp[[cross_dim,'delta','delta_rate','delta_weight','weight_x','weight_y','rate_x','label_x','cnt_x','rate_y','label_y','cnt_y']]
        delta_drilldown.loc[:,'obs_total'] = summary_total.loc['obs','rate']
        delta_drilldown.loc[:,'cmp_total'] = summary_total.loc['cmp','rate']
        delta_drilldown.loc[:,'name'] = name
        delta_drilldown.loc[:,'funnel_has_null'] = 0
        delta_drilldown.columns = ['grp','delta','delta_rate','delta_weight','obs_weight','cmp_weight','obs','obs_numerator','obs_denominator','cmp','cmp_numerator','cmp_denominator','obs_total','cmp_total','name','funnel_has_null']
        delta_drilldown = delta_drilldown[['name','grp','delta','delta_rate','delta_weight','obs','obs_numerator','obs_denominator','cmp','cmp_numerator','cmp_denominator','obs_total','cmp_total','obs_weight','cmp_weight','funnel_has_null']]
        delta_drilldown['is_leaf'] = 1
        delta_drilldown.set_index('grp',inplace=True)
        delta_drilldown

        return delta_drilldown,dimension_list
    
    @staticmethod
    def flow_analyse(df,name,number_names,ratio_names):
        '''返回业务流拆解,用循环替代法实现
        df: 筛选出的观察期和对比期的数据
        name: 指标名string
        number_names: 漏斗各业务流程指标名list
        ratio_names: 漏斗各转化指标名list
        '''
        mltplr_list = [number_names[0]]
        flow_list = number_names
        for c in number_names:
            try:
                df[c] = pd.to_numeric(df[c],errors='coerce')
            except:
                raise Exception("漏斗分析失败，指标名不在数据集中！请检查数据集是否有",c)
            
        df,empty_indices = Metric.fill_funnel_df(df,flow_list)

        summary = df.groupby(['grp'])[number_names].sum().reset_index()

        # 若缺失对比期或观察期的数据，先补为0
        summary = Metric.complete_data(summary,{'grp':['obs','cmp']})
        summary.set_index('grp',inplace=True)

        # 保存obs_total和cmp_total
        obs_total = summary.loc['obs',name]
        cmp_total = summary.loc['cmp',name]

        # 计算漏斗环节的所有转化率
        for i in range(len(number_names)-1):
            summary[ratio_names[i]] = summary.apply(lambda row: row[number_names[i+1]] / row[number_names[i]] if row[number_names[i]] != 0 else 0, axis=1)

        # 计算漏斗环节整体转化率
        summary['label'] = summary.apply(lambda row: row[number_names[-1]] / row[number_names[0]] if row[number_names[0]] != 0 else 0, axis=1)
        summary = summary.drop(flow_list[1:],axis = 1)
        summary_sub = summary.drop('label',axis = 1)
        obs_label = summary.loc['obs',mltplr_list+ratio_names].to_list()
        cmp_label = summary.loc['cmp',mltplr_list+ratio_names].to_list()
        
        # 计算delta
        ctb_list = []
        for l in mltplr_list+ratio_names:
            ctb = np.product(summary_sub.loc['cmp',[i for i in mltplr_list+ratio_names if i != l]])*summary_sub.loc['obs',l]-np.product(summary_sub.loc['cmp',:])
            ctb_list.append(ctb)
            summary_sub.loc['cmp',l] = summary_sub.loc['obs',l]

        ctb_list = [round(x,6) for x in ctb_list]

        # 保存信息
        df_ctb = pd.DataFrame({'label':mltplr_list+ratio_names,
                        'ctb':ctb_list})
        df_ctb['obs'] = obs_label
        df_ctb['cmp'] = cmp_label
        df_ctb['obs_total'] = obs_total
        df_ctb['cmp_total'] = cmp_total
        df_ctb['name'] = name
        for i in ['obs_numerator','obs_denominator','cmp_numerator','cmp_denominator','delta_rate','delta_weight','obs_weight','cmp_weight']:
            df_ctb[i] = None
        df_ctb['funnel_has_null'] = 0
        df_ctb.loc[empty_indices,'funnel_has_null'] = 1
        empty_flow_list = [flow_list[i] for i in empty_indices]

        df_ctb.columns = ['grp','delta','obs','cmp','obs_total','cmp_total','name','obs_numerator','obs_denominator','cmp_numerator','cmp_denominator','delta_rate','delta_weight','obs_weight','cmp_weight','funnel_has_null']
        df_ctb = df_ctb[['name','grp','delta','delta_rate','delta_weight','obs','obs_numerator','obs_denominator','cmp','cmp_numerator','cmp_denominator','obs_total','cmp_total','obs_weight','cmp_weight','funnel_has_null']]
        df_ctb['is_leaf'] = 1
        df_ctb.set_index('grp',inplace=True)
        df_ctb.index = r'(' + df_ctb.index + r')'
        return df_ctb,obs_label,cmp_label,empty_flow_list  # 新增返回有空值的漏斗环节索引
    

class Metric(MetricTreeNode):
    """指标基类，数值指标"""
    def __init__(
        self,
        name: str,
        dataset: Optional[MultiDataset]=None,
        dataset_key: str=None,
        dimension: Union[str, List[str]]=None,
        children: Optional[Dict[str, MetricTreeNode]]=None
    ) -> None:
        """

        name: 指标名
        dataset: multidataset数据集
        dataset_key：数据集名称key
        dimension: 拆解的维度,可以是一个维度字符串，也可以是一个列表，如果是列表则将列表中所有维度交叉成一个新的维度拆解
        children: 子节点, key是维度的值, value是其他指标
        """
        if dimension is not None:
            assert isinstance(dimension, (str, list))

        super().__init__(name, dataset,dataset_key,children)
        self.dimension = dimension

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, children):
        if children is not None:
            assert isinstance(children, dict)
            for k, v in children.items():
                assert isinstance(k, str)
                assert isinstance(v, MetricTreeNode)
        self._children = children


class NumberMetric(Metric):
    """数值类指标"""
    # 引用父类方法的docstring
    @wraps(Metric.drilldown)
    def drilldown(self, obs_period, cmp_period, depth=np.inf, dataset=None, dataset_key = None, depth_i=1, father_grp ='', later_pc=1) -> dict:
        assert isinstance(self, NumberMetric)

        output_df = pd.DataFrame(columns=['name', 'delta', 'delta_rate', 'delta_weight','obs','obs_numerator','obs_denominator','cmp','cmp_numerator','cmp_denominator',
       'obs_total', 'cmp_total', 'obs_weight', 'cmp_weight', 'funnel_has_null', 'is_leaf', 'type','depth','later_pc','final_delta'])
        if self is None:
            return output_df
        # 获取数据集
        if dataset is None:
            dataset = self.dataset
        if dataset is None:
            return output_df
        if self.dataset_key is not None:
            dataset_key = self.dataset_key

        dimension = self.dimension
        dataset_get = dataset.datasets.get(dataset_key, None)

        if dataset_key is None or (dataset_get is None) or dimension is None or len(dimension)==0:
            return output_df

        assert isinstance(dataset, MultiDataset)
        if isinstance(dimension, str):
            assert dimension in dataset_get.df.columns
        elif isinstance(dimension, list):
            for dim in dimension:
                assert dim in dataset_get.df.columns

        name = self.name
        label = [name]
        
        # 获取观察期和对比期数据
        df, obs_rng, cmp_rng= dataset_get.filter_by_period(obs_period,cmp_period)
        if len(df)>0:
            output_df,dimension_list = self.dimension_drilldown(df,name,label,dimension)

            # 保存信息
            output_df['type'] = 'NumberMetric' # 拆解类型
            output_df['depth'] = depth_i # 当前下钻深度
            if depth_i>1:
                father_grp = father_grp+'_' # 继承父节点名称
            output_df.index = father_grp + output_df.index
            output_df['later_pc'] = later_pc # 继承父节点delta转化因子
            output_df['final_delta'] = output_df['delta']*output_df['later_pc'] #计算对根节点指标的delta

            # 下层拆解
            depth_i=depth_i+1
            if self.children is not None and depth_i <= depth:
                child_dataset_dict = dataset.sub_datasets_by_dimension(dimension,dimension_list) # 按dimension拆分数据集
                for dimension_key, child in self.children.items():
                    if dimension_key in dimension_list:
                        assert isinstance(child, MetricTreeNode)
                        dataset_new = child_dataset_dict[dimension_key]
                        father_grp_new = father_grp+dimension_key # 更新父节点名称
                        deeper_df = child.drilldown(obs_period, cmp_period, depth=depth, dataset = dataset_new, dataset_key=dataset_key, depth_i=depth_i, father_grp=father_grp_new, later_pc=later_pc)
                        if len(deeper_df)>0:
                            output_df.loc[father_grp_new,'is_leaf']=0 # 将父节点标记为非叶子结点
                        output_df = pd.concat([output_df, deeper_df])

        return output_df


class RatioMetric(NumberMetric):
    """比例指标，由分子指标和分母指标组成"""
    def __init__(
        self,
        name: str,
        numerator: Metric,
        denominator: Metric,
        dataset: Optional[MultiDataset]=None,
        dataset_key: str=None,
        dimension: Union[str, List[str]]=None,
        children: Optional[Dict[str, MetricTreeNode]]=None
    ) -> None:
        """

        name: 指标名
        numerator: 分子指标
        denominator: 分母指标
        dataset: multidataset数据集数据集
        dataset_key: 数据集名称key
        dimension: 拆解的维度,可以是一个维度字符串，也可以是一个列表，如果是列表则将列表中所有维度交叉成一个新的维度拆解
        children: 子节点, key是维度的值, value是其他指标
        """
        assert isinstance(numerator, Metric)
        assert isinstance(denominator, Metric)
        super().__init__(name, dataset, dataset_key, dimension, children)
        self.numerator = numerator
        self.denominator = denominator
        # 保证RatioMetric节点的children不能是RatioMetric
        if children is not None:
            for child_key, child_value in children.items():
                if isinstance(child_value, RatioMetric):
                    raise TypeError(f"Invalid configuration: Child node with key '{child_key}' in the 'children' dictionary of RatioMetric '{name}' cannot be another RatioMetric instance. Nested RatioMetrics are not supported.")
        
        
    # 引用父类方法的docstring
    @wraps(NumberMetric.drilldown)
    def drilldown(self, obs_period, cmp_period, depth=np.inf, dataset=None, dataset_key = None, depth_i=1, father_grp ='', later_pc=1) -> dict:
        assert isinstance(self, RatioMetric)
        
        output_df = pd.DataFrame(columns=['name', 'delta', 'delta_rate', 'delta_weight','obs','obs_numerator','obs_denominator','cmp','cmp_numerator','cmp_denominator',
       'obs_total', 'cmp_total', 'obs_weight', 'cmp_weight', 'funnel_has_null', 'is_leaf', 'type','depth','later_pc','final_delta'])
        if self is None:
            return output_df
        # 获取数据集
        if dataset is None:
            dataset = self.dataset
        if dataset is None:
            return output_df
        if self.dataset_key is not None:
            dataset_key = self.dataset_key

        dimension = self.dimension
        dataset_get = dataset.datasets.get(dataset_key, None)

        if dataset_key is None or (dataset_get is None) or dimension is None or len(dimension)==0:
            return output_df

        assert isinstance(dataset, MultiDataset)
        if isinstance(dimension, str):
            assert dimension in dataset_get.df.columns
        elif isinstance(dimension, list):
            for dim in dimension:
                assert dim in dataset_get.df.columns
        
        name = self.name
        label = [self.denominator.name,self.numerator.name]
        
        df, obs_rng, cmp_rng= dataset_get.filter_by_period(obs_period,cmp_period)
        
        if len(df)>0:
            output_df,dimension_list = self.dimension_drilldown(df,name,label,dimension)
            # 保存信息
            output_df['type'] = 'RatioMetric'
            output_df['depth'] = depth_i
            if depth_i>1:
                father_grp = father_grp+'_'
            output_df.index = father_grp + output_df.index
            output_df['later_pc'] = later_pc
            output_df['final_delta'] = output_df['delta']*output_df['later_pc']

            # 下层拆解
            depth_i=depth_i+1
            if self.children is not None and depth_i <= depth:
                child_dataset_dict = dataset.sub_datasets_by_dimension(dimension,dimension_list) # 按dimension拆分数据集
                for dimension_key, child in self.children.items():
                    if dimension_key in dimension_list:
                        assert isinstance(child, MetricTreeNode)
                        dataset_new = child_dataset_dict[dimension_key]
                        father_grp_new = father_grp+dimension_key
                        deeper_df = child.drilldown(obs_period, cmp_period, depth=depth, dataset = dataset_new, dataset_key=dataset_key, depth_i=depth_i, father_grp=father_grp_new, later_pc=later_pc)
                        if len(deeper_df)>0:
                            output_df.loc[father_grp_new,'is_leaf']=0
                        output_df = pd.concat([output_df, deeper_df])


        return output_df
    
    def repr(self) -> str:
        return f'{self.__class__.__name__}({self.name} = {self.numerator.name} / {self.denominator.name})'


def generate_metrics_by_funnel(
    number_names: List[str],
    ratio_names: List[str]
) -> (List[NumberMetric], List[RatioMetric]):
    """
    输入一个漏斗每个环节的数值类指标名, 批量创建指标
    node_names: 漏斗环节名
    ratio_name: 比例指标名, 长度必须比funnel少一

    return: 
        返回两个list
        list1: 数值指标list
        list2: 比例指标list
    """
    # 校验输入项合法性
    assert isinstance(number_names, list)
    assert isinstance(ratio_names, list)
    assert len(number_names) == len(ratio_names) + 1

    for item in number_names:
        assert isinstance(item, str)

    for item in ratio_names:
        assert isinstance(item, str)

    # 生成数值类指标
    number_metrics = list(map(NumberMetric, number_names))

    # 生成比例类指标
    funnel_metrics = []
    for i, item in enumerate(ratio_names):
        funnel_metrics.append(RatioMetric(item, number_metrics[i+1], number_metrics[i]))

    return number_metrics, funnel_metrics

class Funnel(MetricTreeNode):
    """漏斗拆解, 按照加法拆解贡献"""
    def __init__(
        self,
        name: str,
        dataset: Optional[MultiDataset]=None,
        dataset_key: str=None,
        children: List[MetricTreeNode]=None,
    ):
        """
        定义指标拆解的数据集和根结点指标
        root: 根节点指标
        dataset: 数据集
        metric_name: 漏斗整体指标的名字
        children: 漏斗指标(也可以是漏斗拆解或者维度拆解), 需要前一项的分母必须和后一项相同
        """
        super().__init__(name, dataset, dataset_key, children)

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, children):
        if children is not None:
            assert isinstance(children, list)
            for item in children:
                    assert isinstance(item, (Metric, MetricTreeNode))
        self._children = children
            
    # 引用父类方法的docstring
    @wraps(MetricTreeNode.drilldown)
    def drilldown(self, obs_period, cmp_period, depth=np.inf, dataset=None, dataset_key = None, depth_i=1, father_grp ='', later_pc=1) -> dict:
        assert isinstance(self, Funnel)
        
        output_df = pd.DataFrame(columns=['name', 'delta', 'delta_rate', 'delta_weight','obs','obs_numerator','obs_denominator','cmp','cmp_numerator','cmp_denominator',
       'obs_total', 'cmp_total', 'obs_weight', 'cmp_weight', 'funnel_has_null', 'is_leaf', 'type','depth','later_pc','final_delta'])
        if self is None:
            return output_df
        # 获取数据及
        if dataset is None:
            dataset = copy.deepcopy(self.dataset)
        if dataset is None:
            return output_df
        if self.dataset_key is not None:
            dataset_key = self.dataset_key

        dataset_get = dataset.datasets.get(dataset_key, None)
        if dataset_key is None or (dataset_get is None):
            return output_df
        period_colname = dataset_get.period_colname

        assert isinstance(dataset, MultiDataset)
        # 获取漏斗信息
        name = self.name
        funnel_len = len(self.children)
        number_names = [self.children[0].name]+[self.children[i].numerator.name for i in range(1,funnel_len)]
        ratio_names = [self.children[i].name for i in range(1,funnel_len)]
        label_list = [number_names[0]]+ratio_names
        
        # 获取观察期与对比期数据
        df, obs_rng, cmp_rng= dataset_get.filter_by_period(obs_period,cmp_period)
        if len(df)>0:
            output_df,obs_label,cmp_label,empty_flow_list = self.flow_analyse(df,name,number_names,ratio_names)
            # 将dataset中对应的数据集更新成处理过漏斗环节空值的新数据
            df = df.drop('grp',axis = 1)
            dataset.datasets[dataset_key] = Dataset(df, period_colname)

            # 保存信息
            output_df['type'] = 'Funnel'
            output_df['depth'] = depth_i
            if depth_i>1:
                father_grp = father_grp+'_'
            output_df.index = father_grp + output_df.index
            output_df['later_pc'] = later_pc
            output_df['final_delta'] = output_df['delta']*output_df['later_pc']
            
            # 下层拆解
            depth_i=depth_i+1
            for child in self.children:
                if child.dimension is not None and depth_i <= depth and child.name in label_list:
                    index = label_list.index(child.name) # 获取子节点指标在漏斗中的位置
                    # 存储later_pc_new
                    later_pc_new = later_pc*np.product([1]+obs_label[:index])*np.product([1]+cmp_label[index+1:])
                    # 存储父节点father_grp_new
                    father_grp_new = father_grp+'('+child.name+')'
                    
                    deeper_df = child.drilldown(obs_period, cmp_period, depth=depth, dataset = dataset, dataset_key=dataset_key,  depth_i=depth_i, father_grp=father_grp_new, later_pc=later_pc_new)
                    if len(deeper_df)>0:
                            output_df.loc[father_grp+'('+child.name+')','is_leaf']=0 # 指标所在漏斗指标节点置为非叶子节点
                    # 如果该节点的漏斗环节有填充过空值，那么传给下层的funnel_has_null
                    if number_names[index] in empty_flow_list:
                        deeper_df['funnel_has_null'] = 1
                    output_df = pd.concat([output_df, deeper_df])
                else :
                    continue

        return output_df

    def repr(self):
        # 将本节点标识为子节点的连乘
        if self.children is not None:
            output = ' x '.join(child.name for child in self.children)
            output = f'{self.__class__.__name__}({self.name} = {output})'
            return output
        else:
            return super().repr()


def sort_f(x,y):
    df_xy = pd.DataFrame({'x':x,'y':y})
    df_xy = df_xy.sort_values(by = 'y',ascending = False)
    x = df_xy['x'].to_list()
    y = df_xy['y'].to_list()
    return x,y

def draw_f(x,y, title=None, yaxis_title=None):
    import plotly.graph_objects as go
        
    fig = go.Figure()
    x_delta,y_delta = sort_f(x[1:-1],y[1:-1])
    x = [x[0]]+x_delta+[x[-1]]
    y = [y[0]]+y_delta+[y[-1]]
    fig.add_trace(go.Waterfall(
                    name = "contribution", orientation = "v",
                    measure = ["initial"] + ['relative']*(len(y)-2)+["total"],
                    x = x,
                    textposition = "auto",
                    text = [str(x) for x in y],
                    y = y,
                    connector = {"line":{"color":"rgb(63, 63, 63)"}}))
    if title is None:
        fig.update_layout(showlegend = True,yaxis_title=yaxis_title)
        fig.update_traces(textfont=dict(size=13))  # 设置字体大小为14
    else:
        fig.update_layout(title=title,showlegend = True,yaxis_title=yaxis_title)
        fig.update_traces(textfont=dict(size=13))  # 设置字体大小为14
    return fig
