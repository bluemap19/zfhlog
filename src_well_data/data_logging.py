import os
import sys

import pandas as pd
from src_logging.curve_preprocess import get_resolution_by_depth, data_Normalized
import xml.etree.ElementTree as ET

class data_logging:
    def __init__(self, path='', well_name=''):
        self._data = pd.DataFrame()
        self._data_normed = pd.DataFrame()
        self._data_with_type = pd.DataFrame()        # 选取的self.data[]+table_type（data_table数据）， 测井数据＋类别数据 结果
        self._data_normed_with_type = pd.DataFrame()
        self._curve_names = []
        self._file_path = path
        self._logging_name = well_name
        self._resolution = -1
        # self.mapping_dict = {
        #     'depth': ['#Depth', 'Depth', 'DEPTH', 'depth'],
        #     'CAL': ['CAL', 'CALC', 'CALX', 'CALY'],
        #     'SP': ['SP', 'Sp'],
        #     'GR': ['GR', 'GRC'],
        #     'CNL': ['CNL', 'CN'],
        #     'DEN': ['DEN', 'DENC'],
        #     'DT': ['DT', 'DT24', 'DTC', 'AC', 'Ac'],
        #     'RXO': ['RXO', 'Rxo', 'RXo', 'RS', 'Rs'],
        #     'RD': ['RD', 'Rd', 'Rt', 'RT'],
        # }
        self.mapping_dict = self.load_config("COLS_MAPPIMG.xml")

    def read_data(self, file_path='', table_name=''):
        if file_path == '':
            file_path = self._file_path
        if table_name == '':
            # table_name = self._logging_name
            table_name = 0

        if file_path.endswith('.csv'):
            try:
                self._data = pd.read_csv(file_path)
            except Exception as e:
                print('文件读取失败', e)
                self._data = pd.DataFrame()
        elif file_path.endswith('.xlsx'):
            try:
                self._data = pd.read_excel(file_path, sheet_name=table_name)
            except Exception as e:
                print('文件读取失败', e)
                self._data = pd.DataFrame()

        # 分辨率初始化
        self._resolution = get_resolution_by_depth(self._data.iloc[:, 0].values)
        # 表格头初始化
        self._curve_names = list(self._data.columns)

    def save_data(self, file_path='', table_name=''):

        pass

    # 数据直接读取
    def get_data(self, curve_names=[], depth_delete=[]):
        # 判断数据体是否为空
        if self._data.empty:
            self.read_data()
        # 判断是否需要 初始化 曲线名称列表
        if curve_names == []:
            curve_names = self._curve_names
        # 判断 曲线是否包含深度特征 不包含的话，一定给他自动加上
        if not curve_names[0].lower().__contains__('depth'):
            curve_names = [self._curve_names[0]] + curve_names

        curve_names = self.input_cols_mapping(curve_names, self._curve_names)
        if len(depth_delete) == 0:
            return self._data[curve_names]
        else:
            data_temp = self._data[curve_names]


    # 获得 归一化数据
    def get_data_normed(self, curve_names=[]):
        # 如果归一化数据为零，则初始化归一化数据
        if self._data_normed.empty:
            data_target = self.get_data(curve_names)
            cols_new_temp = list(data_target.columns)
            data_normed_temp = data_Normalized(data_target.values, DEPTH_USE=True, logging_range=[-999, 9999], max_ratio=0.01, local_normalized=False)
            self._data_normed = pd.DataFrame(data_normed_temp, columns=cols_new_temp)
        else:
            # 如果 self._data_normed 中含有 curve_names，则直接返回
            if set(curve_names) < set(list(self._data_normed.columns)):
                return self._data_normed[curve_names]
            # 否则的话，重新进行正则化
            else:
                data_target = self.get_data(curve_names)
                cols_new_temp = list(data_target.columns)
                data_normed_temp = data_Normalized(data_target.values, DEPTH_USE=True, logging_range=[-999, 9999], max_ratio=0.01, local_normalized=False)
                self._data_normed = pd.DataFrame(data_normed_temp, columns=cols_new_temp)

        return self._data_normed


    # 测井曲线名称映射 例如 把DT24 -映射为- AC 等等
    def input_cols_mapping(self, input_cols=[], target_cols=[]):
        if target_cols == []:
            target_cols = self._curve_names
        input_mapping_cols = input_cols.copy()

        # 遍历 l1，保留不在 l2 中的元素
        unindex_result = [element for element in input_cols if element not in target_cols]

        processed_cols = []
        for element in reversed(unindex_result):
            idx = input_mapping_cols.index(element)
            for target, replacement in self.mapping_dict.items():
                if element in replacement:
                    # 转换为集合求交集
                    intersection = list(set(replacement) & set(target_cols))
                    print(element, '--->', intersection)
                    input_mapping_cols[idx] = intersection[0]
                    processed_cols.append(element)
                    break

        unprocessed_cols = [element for element in unindex_result if element not in processed_cols]
        if unprocessed_cols:
            print('Exist unprocessable cols:', unprocessed_cols)
            exit(0)

        return input_mapping_cols

    def get_curve_names(self):
        if len(self._curve_names) == 0:
            self.read_data()
        return self._curve_names


    def load_config(self, config_name):
        print(sys.argv[0], os.getcwd())
        """从 XML 文件加载配置"""
        # path_current = os.getcwd()
        path_current = r'C:\Users\ZFH\Documents\GitHub\zfhlog\src_well_data'
        filepath = path_current + f"\{config_name}"

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"配置文件不存在: {filepath}")

        # 解析 XML 文件
        tree = ET.parse(filepath)
        root = tree.getroot()

        # 创建空字典
        data_dict = {}

        # 遍历根元素的所有子元素
        for log_type in root.findall("LogType"):
            # 获取键名
            key = log_type.get("name")

            # 获取所有别名
            aliases = [alias.text for alias in log_type.findall("Alias")]

            # 添加到字典
            data_dict[key] = aliases

        return data_dict



if __name__ == '__main__':
    test_data = data_logging(path=r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV\城96\城96_logging_data.csv', well_name='城96')

    col_names = ['GRC', 'CN', 'DT', 'Sp']
    print(test_data.get_data(curve_names=col_names).describe())
    print(test_data.mapping_dict)
