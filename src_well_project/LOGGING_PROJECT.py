import os

import pandas as pd

from src_file_op.dir_operation import get_all_subfolder_paths
from src_table.table_process import get_replace_dict
from src_well_data.DATA_WELL import WELL


class LOGGING_PROJECT:
    def __init__(self):
        self.PROJECT_PATH=r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV'
        # ['元543', '元552', '悦235', '悦88', '悦92', '珠201', '珠202', '珠23', '珠45', '珠74', '珠79', '珠80', '白159', '白291', '白292', '白294', '白300', '白75']
        self.WELL_NAMES = []
        self.WELL_PATH = {}
        self.target_curves = {}
        self.WELL_DATA = {}
        self.replace_dict_all = {}
        self.data_vc = pd.DataFrame()

        self.init_well_path()

    def init_well_path(self):

        path_all = get_all_subfolder_paths(self.PROJECT_PATH)

        if not self.WELL_NAMES:
            self.WELL_NAMES = [os.path.basename(p) for p in path_all]
            self.WELL_PATH = {name: path for name, path in zip(self.WELL_NAMES, path_all)}
        else:
            valid_paths = {}
            WELL_NAMES = []
            for p in path_all:
                folder_name = os.path.basename(p)
                if folder_name in self.WELL_NAMES:
                    valid_paths[folder_name] = p
                    WELL_NAMES.append(folder_name)
            self.WELL_PATH = valid_paths
            self.WELL_NAMES = WELL_NAMES

        for well_name in self.WELL_PATH.keys():
            self.WELL_DATA[well_name] = WELL(path_folder=self.WELL_PATH[well_name], WELL_NAME=well_name)

        # print(self.WELL_NAMES)
        # print(self.WELL_PATH)
        # print(self.WELL_DATA)
    def get_default_dict(self, dict={}, key_default=''):
        if not dict:
            print('Empty dictionary get')
            return
        if key_default == '':
            key_default = list(dict.keys())[0]
            value_default = dict[key_default]
        else:
            value_default = dict[key_default]
        return value_default

    def get_well_data(self, well_name='', file_path='', curve_names=[]):
        if well_name not in self.WELL_NAMES:
            if len(well_name) == 0:
                well_name = self.WELL_NAMES[0]
            else:
                print('Error well name:{}'.format(well_name))
                exit(0)

        WELL_temp = self.get_default_dict(self.WELL_DATA, well_name)
        data = WELL_temp.get_logging_data(well_key=file_path, curve_names=curve_names)
        return data

    def get_table_3_data(self, well_name='', file_path='', curve_names=[]):
        WELL_temp = self.get_default_dict(self.WELL_DATA, well_name)
        type_3 = WELL_temp.get_type_3(table_key=file_path, curve_names=curve_names)
        return type_3
    def get_table_2_data(self, well_name='', file_path='', curve_names=[]):
        WELL_temp = self.get_default_dict(self.WELL_DATA, well_name)
        type_2 = WELL_temp.get_type_2(table_key=file_path, curve_names=curve_names)
        return type_2
    def get_table_3_all_data(self, well_names=[], file_path={}, curve_names=[]):
        if well_names == []:
            well_names = self.WELL_NAMES

        if not file_path:
            for well_name in well_names:
                file_path[well_name] = self.WELL_PATH[well_name]

        table_3_list = []
        for well_name in well_names:
            table_3_df_t = self.get_table_3_data(well_name=well_name, file_path=file_path[well_name], curve_names=curve_names)
            table_3_list.append(table_3_df_t)
        self.data_vc = self.data_vertical_cohere(data_list=table_3_list, well_names=well_names)
        return self.data_vc

    def data_vertical_cohere(self, data_list=[], well_names=[]):
        if data_list == []:
            data_list = self.data_norm_list
        if well_names == []:
            well_names = self.WELL_NAMES

        for i in range(len(data_list)):
            well_temp = data_list[i]
            # 对每一口井都先添加井名，再进行合并
            well_temp['Well_Name'] = well_names[i]
            if i == 0:
                data_vertical_combined = well_temp
            else:
                # 合并并添加来源标识
                data_vertical_combined = pd.concat(
                    [data_vertical_combined, well_temp], axis=0
                )

        print('data_vertical_combined shape :', data_vertical_combined.shape)
        data_vertical_combined.reset_index(drop=True, inplace=True)
        return data_vertical_combined

    def get_all_table_replace_dict(self):
        table_value = self.get_table_3_all_data()
        self.replace_dict_all = get_replace_dict(table_value['Type'])
        return self.replace_dict_all

    def set_all_table_replace_dict(self, dict={}):
        if dict:
            self.replace_dict_all = dict
            print('set all table replace dict as:{}'.format(self.replace_dict_all))
        else:
            print('No replace dict')
            exit(0)

LG = LOGGING_PROJECT()
a = LG.get_well_data(curve_names=['AC', 'CNL', 'GR'])
# print(a)

data_vc = LG.get_table_3_all_data()
print(data_vc.head(30))
b = LG.get_all_table_replace_dict()
print(b)
LG.set_all_table_replace_dict(dict={'中GR长英黏土质': 0, '中GR长英黏土质（泥岩）': 0, '中低GR长英质': 1, '中低GR长英质（砂岩）': 1,
                                   '富有机质长英质': 2, '富有机质长英质页岩': 2, '富有机质黏土质': 3, '富有机质黏土质页岩': 3,
                                   '高GR富凝灰长英质': 4, '高GR富凝灰长英质（沉凝灰岩）': 4})
# data_t = B.get_well_data('白75', ['AC', 'CNL', 'GR'])
# print(data_t)
# data_t = B.get_well_data('白75', ['DEN', 'SP'])
# print(data_t)
# data_t = B.get_well_data('白75', ['AC', 'CNL', 'GR', 'DEN', 'SP'])
# print(data_t)
#
# data_t = B.get_well_data('白291', ['AC', 'CNL', 'GR', 'DEN', 'SP'])
# print(data_t)
# data_t = B.get_well_data('白291', ['DEN', 'SP'])
# print(data_t)
# data_t = B.get_well_data('白291', ['AC', 'CNL'])
# print(data_t)


