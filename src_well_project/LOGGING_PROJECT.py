import os

import pandas as pd

from src_data_process.data_correction_analysis import data_correction_analyse_by_tree
from src_data_process.data_filter import pdnads_data_drop, pandas_data_filtration
from src_file_op.dir_operation import get_all_subfolder_paths
from src_table.table_process import get_replace_dict
from src_well_data.DATA_WELL import WELL


class LOGGING_PROJECT:
    def __init__(self):
        self.PROJECT_PATH=r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV'
        # ['元543', '元552', '悦235', '悦88', '悦92', '珠201', '珠202', '珠23', '珠45', '珠74', '珠79', '珠80', '白159', '白291', '白292', '白294', '白300', '白75']
        self.WELL_NAMES = []
        self.WELL_PATH = {}
        self.well_logging_path_dict = {}
        self.well_table_path_dict = {}
        self.well_FMI_path_dict = {}
        self.well_NMR_path_dict={}

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
            well_temp = WELL(path_folder=self.WELL_PATH[well_name], WELL_NAME=well_name)
            self.well_logging_path_dict[well_name] = well_temp.get_logging_path_list()
            self.well_table_path_dict[well_name] = well_temp.get_table_path_list()
            # self.well_FMI_path_dict[well_name] = well_temp.get_FMI_path_list()
            # self.well_NMR_path_dict = well_temp.get_NMR_path_list()

            self.WELL_DATA[well_name] = well_temp

        # print(self.WELL_NAMES)
        # print(self.WELL_PATH)
        # print(self.WELL_DATA)

        print(self.well_logging_path_dict)
        print(self.well_table_path_dict)
        # print(self.well_FMI_path_dict)
        # print(self.well_NMR_path_dict)
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

        if file_path == '':
            file_path = self.well_logging_path_dict[well_name][0]
            print('current file path set as: {}'.format(file_path))

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
                file_path[well_name] = self.well_table_path_dict[well_name][0]

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

    def combined_all_logging_with_type(self, well_names=[], file_path_logging={}, file_path_table={}, curve_names_logging=[], curve_names_type=[], replace_dict={}, type_col_name='', Norm=False):
        if well_names == []:
            well_names = self.WELL_NAMES
        if not file_path_logging:
            for well_name in well_names:
                file_path_logging[well_name] = self.well_logging_path_dict[well_name][0]
        if not file_path_table:
            for well_name in well_names:
                file_path_table[well_name] = self.well_table_path_dict[well_name][0]

        if not replace_dict:
            replace_dict = self.replace_dict_all

        data_logging_type_list = []
        for well_name in well_names:
            well_data = self.get_default_dict(self.WELL_DATA, well_name)
            data_logging_type = well_data.combine_logging_table(well_key=file_path_logging[well_name],
                                                                curve_names_logging=curve_names_logging,
                                                                table_key=file_path_table[well_name],
                                                                curve_names_table=curve_names_type,
                                                                replace_dict=replace_dict, new_col=type_col_name, Norm=Norm)
            data_logging_type_list.append(data_logging_type)
            print(data_logging_type.describe())
        data_final = self.data_vertical_cohere(data_list=data_logging_type_list, well_names=well_names)
        print(data_final.describe())
        return data_final


LG = LOGGING_PROJECT()
a = LG.get_well_data(well_name='白75', curve_names=['AC', 'CNL', 'GR'])
# print(a)

data_vc = LG.get_table_3_all_data()
b = LG.get_all_table_replace_dict()
print(b)
dict = {'中GR长英黏土质': 0, '中GR长英黏土质（泥岩）': 0, '中低GR长英质': 1, '中低GR长英质（砂岩）': 1,
                                   '富有机质长英质': 2, '富有机质长英质页岩': 2, '富有机质黏土质': 3, '富有机质黏土质页岩': 3,
                                   '高GR富凝灰长英质': 4, '高GR富凝灰长英质（沉凝灰岩）': 4}
LG.set_all_table_replace_dict(dict=dict)

c = LG.combined_all_logging_with_type(well_names=['元543', '元552', '悦235', '珠23'], curve_names_logging=['AC', 'CNL', 'GR'], Norm=True)
d = pdnads_data_drop(c)
print(d.describe())
e = pandas_data_filtration(d)
f = d.iloc[e]
print(f.describe())

data_correction_analyse_by_tree(f, ['AC', 'CNL', 'GR'], 'Type', 10,
                                y_replace_dict=dict, title_string='fffffffffuck')
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


