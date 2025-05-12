import os
from src_well_data.DATA_WELL import WELL


class LOGGING_PROJECT:
    def __init__(self):
        self.PROJECT_PATH=r'C:\Users\Administrator\Desktop\算法测试-长庆数据收集\Code_input-O'
        self.WELL_NAMES = ['白75', '白159', '白291', '白300']
        self.WELL_PATH = {}
        self.target_curves = {}
        self.WELL_DATA = {}

    def init_well_path(self):
        for well in self.WELL_NAMES:
            print(well)

    def get_well_data(self, well, curve_names=[]):
        # 如果找不到数据，就重新初始化WELL类
        if well not in list(self.WELL_DATA.keys()):
            print('NOW LOADING　ADN NEW WELL DATA:{}'.format(well))
            self.WELL_DATA[well] = WELL(self.PROJECT_PATH, well)

        return self.WELL_DATA[well].get_logging_data(curve_names)

    def get_table_data(self, well):
        if well not in list(self.WELL_DATA.keys()):
            print('NOW LOADING　ADN NEW WELL DATA:{}'.format(well))
            self.WELL_DATA[well] = WELL(self.PROJECT_PATH, well)

        return self.WELL_DATA[well].get_table_2()
B = LOGGING_PROJECT()
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


data_t = B.get_table_data('白75')
print(data_t)
data_t = B.get_table_data('白291')
print(data_t)