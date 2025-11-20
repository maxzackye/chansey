"""
Excel数据读取模块
用于支持从Excel文件中读取数据并转换为chansey所需的格式
"""

import pandas as pd
from typing import Dict, List, Optional, Union
from .dataset import MultiDataset


class ExcelReader:
    """
    Excel文件读取器
    支持读取Excel文件中的多个工作表并转换为MultiDataset格式
    """

    def __init__(self, file_path: str, sheet_names: Optional[Union[str, List[str]]] = None):
        """
        初始化Excel读取器
        
        Parameters:
        file_path: Excel文件路径
        sheet_names: 需要读取的工作表名称，如果为None则读取所有工作表
        """
        self.file_path = file_path
        self.sheet_names = sheet_names
        self.dataframes = {}
        self._read_excel()

    def _read_excel(self):
        """读取Excel文件"""
        try:
            # 读取Excel文件
            if self.sheet_names is None:
                # 读取所有工作表
                self.dataframes = pd.read_excel(self.file_path, sheet_name=None)
            elif isinstance(self.sheet_names, str):
                # 读取单个工作表
                self.dataframes = {self.sheet_names: pd.read_excel(self.file_path, sheet_name=self.sheet_names)}
            elif isinstance(self.sheet_names, list):
                # 读取多个指定工作表
                self.dataframes = {}
                for sheet in self.sheet_names:
                    df = pd.read_excel(self.file_path, sheet_name=sheet)
                    self.dataframes[sheet] = df
        except Exception as e:
            raise Exception(f"读取Excel文件失败: {str(e)}")

    def to_multi_dataset(self, period_columns: Union[str, Dict[str, str]]) -> MultiDataset:
        """
        将读取的Excel数据转换为MultiDataset格式
        
        Parameters:
        period_columns: 时间列名
            - 如果所有工作表使用相同的时间列名，传入字符串
            - 如果不同工作表使用不同的时间列名，传入字典，格式为 {'工作表名': '时间列名'}
            
        Returns:
        MultiDataset对象
        """
        dataset_dict = {}
        
        for sheet_name, df in self.dataframes.items():
            if isinstance(period_columns, str):
                period_col = period_columns
            elif isinstance(period_columns, dict):
                period_col = period_columns.get(sheet_name, None)
                if period_col is None:
                    raise ValueError(f"工作表 '{sheet_name}' 未指定时间列名")
            else:
                raise ValueError("period_columns 参数必须是字符串或字典")
                
            dataset_dict[sheet_name] = (df, period_col)
            
        return MultiDataset(dataset_dict)

    def get_sheet_names(self) -> List[str]:
        """获取所有工作表名称"""
        return list(self.dataframes.keys())

    def get_dataframe(self, sheet_name: str) -> pd.DataFrame:
        """获取指定工作表的数据"""
        if sheet_name not in self.dataframes:
            raise ValueError(f"工作表 '{sheet_name}' 不存在")
        return self.dataframes[sheet_name]

    def list_columns(self, sheet_name: str) -> List[str]:
        """列出指定工作表的所有列名"""
        if sheet_name not in self.dataframes:
            raise ValueError(f"工作表 '{sheet_name}' 不存在")
        return list(self.dataframes[sheet_name].columns)


def read_excel_for_chansey(file_path: str, 
                          period_columns: Union[str, Dict[str, str]],
                          sheet_names: Optional[Union[str, List[str]]] = None) -> MultiDataset:
    """
    便捷函数：直接从Excel文件读取数据并转换为MultiDataset格式
    
    Parameters:
    file_path: Excel文件路径
    period_columns: 时间列名
        - 如果所有工作表使用相同的时间列名，传入字符串
        - 如果不同工作表使用不同的时间列名，传入字典，格式为 {'工作表名': '时间列名'}
    sheet_names: 需要读取的工作表名称，如果为None则读取所有工作表
        
    Returns:
    MultiDataset对象
    """
    reader = ExcelReader(file_path, sheet_names)
    return reader.to_multi_dataset(period_columns)