import pandas as pd

from src_logging.curve_preprocess import get_resolution_by_depth
from src_table.table_process import table_2_to_3, get_replace_dict, table_3_to_2


class data_table:
    def __init__(self, path='', well_name='', resolution=-1):
        self.table_2 = pd.DataFrame()
        self.table_3 = pd.DataFrame()
        self.table_resolution = resolution
        self.file_path = path
        self.well_name = well_name
        self.data = pd.DataFrame()
        self.replace_dict = {}

    def read_data(self, file_path='', table_name='', TYPE2=False, TYPE3=False):
        if file_path == '':
            file_path = self.file_path
        if table_name == '':
            table_name = self.well_name

        if file_path.endswith('.csv'):
            try:
                # 尝试中文编码
                self.data = pd.read_csv(file_path, encoding='gbk')
            except UnicodeDecodeError:
                # 尝试其他编码
                self.data = pd.read_csv(file_path, encoding='utf-8-sig')  # 处理 BOM 头
            # self.data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            try:
                self.data = pd.read_excel(file_path, sheet_name=table_name)
            except Exception as e:
                print('文件读取失败', e)
                self.data = pd.DataFrame()
        # 读取完表格数据之后，先对replace_dict进行赋值
        if not self.data.empty:
            self.replace_dict = get_replace_dict(self.data.iloc[:, -1].values)

        if TYPE2:
            self.table_2 = self.data
        if TYPE3:
            self.table_3 = self.data
        else:
            if self.data.shape[1] == 2:
                self.table_2 = self.data
            elif self.data.shape[1] == 3:
                self.table_3 = self.data
            else:
                print('ALARM TABLE　FORMAT:{},{}'.format(self.data.columns, self.data.shape))
                self.table_3 = self.data

        if self.table_3.shape[0] > 0:
            # self.table_3_to_2()
            pass

        elif self.table_2.shape[0] > 0:
            self.table_resolution = get_resolution_by_depth(self.table_2.iloc[:, 0].values)
            # self.table_2_to_3()

    def table_2_to_3(self):
        if self.table_3.shape[0] == 0:
            cols_temp = ['Depth_start', 'Depth_end', 'Type']

            table_3_temp = table_2_to_3(self.table_2.values)
            self.table_3 = pd.DataFrame(table_3_temp, columns=cols_temp)

    def table_3_to_2(self, resolution=-1):
        if resolution < 0:
            resolution = self.table_resolution
        table_3_to_2(self.table_3.values, step=resolution)


    def get_table_3(self, curve_names=[]):
        if self.table_3.shape[0] == 0:
            self.table_2_to_3()
        if len(curve_names) == 3:
            return self.table_3[curve_names]
        if self.table_2.shape[1] > 3:
            return self.table_2.iloc[:, [0, 1, -1]]
        return self.table_3

    def get_table_2(self, curve_names=[]):
        if self.table_2.shape[0] == 0:
            self.table_3_to_2()
        if len(curve_names) == 2:
            return self.table_2[curve_names]
        if self.table_2.shape[1] > 2:
            return self.table_2.iloc[:, [0, -1]]
        return self.table_2

    def set_table_resolution(self, resolution):
        if resolution > 0:
            self.table_resolution = resolution

    def table_type_replace(self, replace_dict={}, new_col=''):
        if replace_dict == {}:
            replace_dict = self.replace_dict

        if new_col == '':
            new_col = 'Type'
        self.table_2[new_col] = self.table_2.iloc[:, -1].map(replace_dict)
        self.table_3[new_col] = self.table_3.iloc[:, -1].map(replace_dict)

        # print(replace_dict)
        # print(self.table_2)
        # print(self.table_3)