import warnings

import numpy as np
import pandas as pd
from src_logging.curve_preprocess import get_resolution_by_depth
from src_table.table_process import table_2_to_3, get_replace_dict, table_3_to_2


class data_table:
    def __init__(self, path='', well_name='', resolution=0.0025):
        self._table_2 = pd.DataFrame()
        self._table_2_replaced = pd.DataFrame()
        self._table_3 = pd.DataFrame()
        self._table_3_replaced = pd.DataFrame()
        self._table_resolution = resolution
        self._file_path = path
        self._well_name = well_name
        self._data = pd.DataFrame()     # 存放的是从文件中读取的原始dataframe
        self._replace_dict_local = {}

    def read_data(self, file_path='', table_name='', TYPE2=False, TYPE3=False):
        if self._data.empty:
            # 文件名称
            if file_path == '':
                file_path = self._file_path
            # 表格名称
            if table_name == '':
                table_name = self._well_name

            if file_path.endswith('.csv'):
                try:
                    # 尝试中文编码
                    self._data = pd.read_csv(file_path, encoding='gbk')
                except UnicodeDecodeError:
                    # 尝试其他编码
                    self._data = pd.read_csv(file_path, encoding='utf-8-sig')  # 处理 BOM 头
                # self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                try:
                    self._data = pd.read_excel(file_path, sheet_name=table_name)
                except Exception as e:
                    warnings.warn('xlsx表格数据读取失败,报错{}，直接读取表格数据的第一个表格'.format(e))
                    self._data = pd.read_excel(file_path, sheet_name=0)

            if TYPE2:
                self._table_2 = self._data
                self.check_table_2()
            if TYPE3:
                self._table_3 = self._data
                self.check_table_3()
            else:
                if self._data.shape[1] == 2:
                    # 表格 2 初始化
                    self._table_2 = self._data
                    self.check_table_2()
                    # 分辨率初始化
                    self._table_resolution = get_resolution_by_depth(self._table_2.iloc[:, 0].values)
                    # 表格2 转 表格3
                    self.table_2_to_3()
                elif self._data.shape[1] == 3:
                    # 表格3初始化
                    self._table_3 = self._data
                    self.check_table_3()
                    # 分辨率初始化
                    if self._table_resolution < 0:
                        self._table_resolution = 0.1
                        print('reset table resolution as 0.1m')
                    # 表格3 转 表格2
                    self.table_3_to_2()
                elif self._data.shape[1] >= 4:
                    # 表格4初始化
                    self._table_3 = self._data.iloc[:, 1:4]
                    self.check_table_3()
                    # 分辨率初始化
                    if self._table_resolution < 0:
                        self._table_resolution = 0.1
                        print('reset table resolution as 0.1m')
                    # 表格3 转 表格2
                    self.table_3_to_2()
                else:
                    print('\033[31m' + 'ALARM TABLE　FORMAT AS:{},{}, YOU MUST ALTER IT AS (DEP_START-DEP_END-TYPE) or (DEPTH-TYPE)'.format(self._data.columns, self._data.shape) + '\033[0m')
                    print(self._data)
                    self._table_3 = self._data
                    self.check_table_3()

            # 读取完表格数据之后，先对 replace_dict 进行赋值
            if not self._data.empty:
                try:
                    self._replace_dict_local = get_replace_dict(self._data.iloc[:, -1].values)
                except Exception as e:
                    print('\033[31m' + 'file path:{}, as exception:{}'.format(self._file_path, e) + '\033[0m')
                    exit(0)
        else:
            pass

    # 对table2进行详细的数据格式检查
    def check_table_2(self):
        if self._table_2.empty:
            print('\033[31m' + 'table file is not empty' + '\033[0m')

        if self._table_2.shape[1] != 2:
            print('\033[31m' + 'table file is not legal 2 cols:{}'.format(self._table_2.columns) + '\033[0m')

        # 检查整个 self._table_2 是否有空值
        has_nulls = self._table_2.isnull().values.any()
        if has_nulls:
            null_counts = self._table_2.isnull().sum()
            print('\033[31m' + f"self._table_2 包含空值:\n{null_counts}\n直接丢弃相应空数据" + '\033[0m')
            self._table_2 = self._table_2.dropna(how='all').reset_index(drop=True)
        for i in range(self._table_2.shape[0]):
            if i != 0:
                depth_last = self._table_2.iloc[i-1, 0]
                depth_now = self._table_2.iloc[i, 0]
                if depth_last > depth_now:
                    print('\033[31m' + 'error depth setting in table:{} in depth:{} & {}'.format(self._file_path, depth_last, depth_now) + '\033[0m')


    # 对table3进行详细的数据格式检查
    def check_table_3(self):
        if self._table_3.empty:
            print('\033[31m' + 'table file is not empty' + '\033[0m')

        if self._table_3.shape[1] != 3:
            print('\033[31m' + 'table file is not legal 3 cols:{}'.format(self._table_3.columns) + '\033[0m')

        # 检查整个 self._table_3 是否有空值
        has_nulls = self._table_3.isnull().values.any()
        if has_nulls:
            null_counts = self._table_3.isnull().sum()
            print('\033[31m' + f"self._table_3 包含空值: \n{null_counts}\n直接丢弃相应空数据" + '\033[0m')
            self._table_3 = self._table_3.dropna(how='all').reset_index(drop=True)
        for i in range(self._table_3.shape[0]):
            if i != 0:
                depth_1 = self._table_3.iloc[i-1, 0]
                depth_2 = self._table_3.iloc[i-1, 1]
                depth_3 = self._table_3.iloc[i, 0]
                depth_4 = self._table_3.iloc[i, 1]
                if depth_1 >= depth_2:
                    print('\033[31m' + 'error depth setting in table:{} in depth:{} & {}'.format(self._file_path, depth_1, depth_2) + '\033[0m')
                if depth_3 >= depth_4:
                    print('\033[31m' + 'error depth setting in table:{} in depth:{} & {}'.format(self._file_path, depth_3, depth_4) + '\033[0m')
                if depth_3 < depth_2:
                    print('\033[31m' + 'error depth setting in table:{} in depth:{} & {}'.format(self._file_path, depth_3, depth_4) + '\033[0m')

    # 表格3转2
    def table_2_to_3(self):
        if self._table_3.shape[0] == 0:
            cols_temp = ['Depth_start', 'Depth_end', 'Type']

            table_3_temp = table_2_to_3(self._table_2.values)
            self._table_3 = pd.DataFrame(table_3_temp, columns=cols_temp)

    # 表格2转3
    def table_3_to_2(self, resolution=-1):
        if resolution < 0:
            resolution = self._table_resolution
        if self._table_2.shape[0] == 0:
            table_2_temp = table_3_to_2(self._table_3.values, step=resolution)
            cols_temp = ['Depth', 'Type']
            self._table_2 = pd.DataFrame(table_2_temp, columns=cols_temp)
        else:
            return

    def get_table_3(self, curve_names=[]):
        if self._table_3.shape[0] == 0:
            self.read_data()
            self.table_2_to_3()
        if len(curve_names) >= 3:
            return self._table_3[curve_names]
        if self._table_3.shape[1] >= 3:
            return self._table_3.iloc[:, [0, 1, -1]]
        return self._table_3
    def get_table_3_replaced(self, curve_names=[]):
        if self._table_3_replaced.shape[0] == 0:
            self.table_2_to_3()
            self.table_type_replace()
        if len(curve_names) >= 3:
            return self._table_3_replaced[curve_names]
        if self._table_3_replaced.shape[1] >= 3:
            return self._table_3_replaced.iloc[:, [0, 1, -1]]
        return self._table_3_replaced

    def get_table_2(self, curve_names=[]):
        if self._table_2.shape[0] == 0:
            self.read_data()
            self.table_3_to_2()
        if len(curve_names) == 2:
            return self._table_2[curve_names]
        if self._table_2.shape[1] >= 2:
            return self._table_2.iloc[:, [0, -1]]
        return self._table_2
    def get_table_2_replaced(self, curve_names=[]):
        if self._table_2_replaced.shape[0] == 0:
            self.table_3_to_2()
            self.table_type_replace()
        if len(curve_names) >= 2:
            return self._table_2_replaced[curve_names]
        if self._table_2_replaced.shape[1] >= 2:
            return self._table_2_replaced.iloc[:, [0, -1]]
        return self._table_2_replaced

    def set_table_resolution(self, resolution):
        if resolution > 0:
            self._table_resolution = resolution

    def table_type_replace(self, replace_dict={}, new_col=''):
        if not replace_dict:
            replace_dict = self._replace_dict_local
        else:
            self._replace_dict_local = replace_dict

        # print('current replace dict: {}'.format(replace_dict))
        if new_col == '':
            new_col = 'Type'

        self._table_2_replaced = self._table_2.copy()
        self._table_2_replaced[new_col] = self._table_2.iloc[:, -1].map(replace_dict)
        self._table_3_replaced = self._table_3.copy()
        self._table_3_replaced[new_col] = self._table_3.iloc[:, -1].map(replace_dict)

        # print(replace_dict)
        # print(self.table_2)
        # print(self.table_3)

if __name__ == '__main__':
    test_table = data_table(path=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\FY1-15\FY1-15_LITHO_TYPE.csv', well_name='FY1-15')
    # test_table = data_table(path=r'F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\城96__纹理岩相划分原始数据.csv', well_name='城96')

    print(test_table.get_table_3().describe())
    print(test_table.get_table_2().describe())
    print(test_table.get_table_2_replaced().describe())
    print(test_table.get_table_3_replaced().describe())
    print(test_table._replace_dict_local)

