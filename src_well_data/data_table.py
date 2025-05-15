import pandas as pd

from src_logging.curve_preprocess import get_resolution_by_depth
from src_table.table_process import table_2_to_3, get_replace_dict, table_3_to_2


class data_table:
    def __init__(self, path='', well_name='', resolution=-1):
        self._table_2 = pd.DataFrame()
        self._table_3 = pd.DataFrame()
        self._table_resolution = resolution
        self._file_path = path
        self._well_name = well_name
        self._data = pd.DataFrame()
        self._replace_dict = {}

    def read_data(self, file_path='', table_name='', TYPE2=False, TYPE3=False):
        if file_path == '':
            file_path = self._file_path
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
                print('文件读取失败', e)
                self._data = pd.DataFrame()
        # 读取完表格数据之后，先对replace_dict进行赋值
        if not self._data.empty:
            try:
                self._replace_dict = get_replace_dict(self._data.iloc[:, -1].values)
            except Exception as e:
                print(e)
                print('file path:{}'.format(self._file_path))
                exit(0)

        if TYPE2:
            self._table_2 = self._data
        if TYPE3:
            self._table_3 = self._data
        else:
            if self._data.shape[1] == 2:
                self._table_2 = self._data
            elif self._data.shape[1] == 3:
                self._table_3 = self._data
            else:
                print('ALARM TABLE　FORMAT:{},{}'.format(self._data.columns, self._data.shape))
                self._table_3 = self._data

        if self._table_3.shape[0] > 0:
            self.table_3_to_2()
        elif self._table_2.shape[0] > 0:
            self._table_resolution = get_resolution_by_depth(self._table_2.iloc[:, 0].values)
            self.table_2_to_3()

    def table_2_to_3(self):
        if self._table_3.shape[0] == 0:
            cols_temp = ['Depth_start', 'Depth_end', 'Type']

            table_3_temp = table_2_to_3(self._table_2.values)
            self._table_3 = pd.DataFrame(table_3_temp, columns=cols_temp)

    def table_3_to_2(self, resolution=-1):
        if resolution < 0:
            resolution = self._table_resolution
        if self._table_2.shape[0] == 0:
            table_2_temp = table_3_to_2(self._table_3.values, step=resolution)
            cols_temp = ['Depth', 'Type']
            self._table_2 = pd.DataFrame(table_2_temp, columns=cols_temp)


    def get_table_3(self, curve_names=[]):
        if self._table_3.shape[0] == 0:
            self.table_2_to_3()
        if len(curve_names) == 3:
            return self._table_3[curve_names]
        if self._table_2.shape[1] > 3:
            return self._table_2.iloc[:, [0, 1, -1]]
        return self._table_3

    def get_table_2(self, curve_names=[]):
        if self._table_2.shape[0] == 0:
            self.table_3_to_2()
        if len(curve_names) == 2:
            return self._table_2[curve_names]
        if self._table_2.shape[1] > 2:
            return self._table_2.iloc[:, [0, -1]]
        return self._table_2

    def set_table_resolution(self, resolution):
        if resolution > 0:
            self._table_resolution = resolution

    def table_type_replace(self, replace_dict={}, new_col=''):
        if replace_dict == {}:
            replace_dict = self._replace_dict

        print('current replace dict: {}'.format(replace_dict))
        if new_col == '':
            new_col = 'Type'
        self._table_2[new_col] = self._table_2.iloc[:, -1].map(replace_dict)
        self._table_3[new_col] = self._table_3.iloc[:, -1].map(replace_dict)

        # print(replace_dict)
        # print(self.table_2)
        # print(self.table_3)