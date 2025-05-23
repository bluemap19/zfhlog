import os
from datetime import datetime

import pandas as pd
from src_file_op.dir_operation import search_files_by_criteria
from src_logging.curve_preprocess import get_resolution_by_depth, data_combine_new, data_combine_table2col
from src_well_data.data_logging import data_logging
from src_well_data.data_table import data_table


class WELL:
    def __init__(self, path_folder='', WELL_NAME=''):
        self.describe_logging = {}
        self.describe_FMI = {}
        self.describe_NMR = {}
        self.describe_table = {}

        self.logging_dict = {}
        self.table_dict = {}
        self.FMI_dict = {}
        self.NMR_dict = {}
        self.logging_table = {}
        self.resolution = {}

        self.well_path = path_folder
        self.curve_names = {}
        self.curve_names_target = {}
        if WELL_NAME == '':
            self.WELL_NAME = path_folder.split('/')[-1].split('\\')[-1].split('.')[0]
        else:
            self.WELL_NAME = WELL_NAME
        self.file_charter_dict = {'logging':['_logging'], 'FMI':['FMI_dyna', 'FMI_stat'], 'NMR':['NMR'], 'table':['_LITHO_TYPE']} # 保存目标文件关键字,分别存放 测井数据文件关键词logging，表格数据关键词table，电成像数据关键词FMI， 核磁数据NMR
        self.file_path_dict = {}                # 保存目标文件路径,分别存放 测井数据文件路径键logging，表格数据路径键table，电成像数据路径键FMI， 核磁数据路径键NMR

        # 初始化之后，直接寻找一下本地路径下是否含有需要的文件
        self.search_data_file()

    # 初始化self.file_path_dict字典，存在四种key，分别是‘logging’，‘table’，‘FMI’，‘NMR’每一个key对应一个list，list里面装的是对应目标测井文件格式的数据路径
    def search_data_file(self):
        # 搜寻测井数据所在文件
        file_list = search_files_by_criteria(self.well_path, name_keywords=[self.WELL_NAME]+self.file_charter_dict['logging'], file_extensions=['.xlsx', '.csv'])
        if len(file_list) == 0:
            file_list = search_files_by_criteria(self.well_path, name_keywords=['Logging_ALL'], file_extensions=['.xlsx', '.csv'])

        if len(file_list) == 0:
            print('No logging files found,in path:{}, by charter:{} {}'.format(self.well_path, self.WELL_NAME, self.file_charter_dict['logging']))
            self.file_path_dict['logging'] = []
        if len(file_list) >= 1:
            print('Successed searching logging file as:{}'.format(file_list))
            self.file_path_dict['logging'] = file_list


        # 搜寻表格数据所在文件
        file_list = search_files_by_criteria(self.well_path, name_keywords=[self.WELL_NAME]+self.file_charter_dict['table'], file_extensions=['.xlsx', '.csv'])
        if len(file_list) == 0:
            file_list = search_files_by_criteria(self.well_path, name_keywords=['_LITHO_TYPE'], file_extensions=['.xlsx', '.csv'])

        if len(file_list) >= 1:
            print('successed searching table file as:{}'.format(file_list))
            self.file_path_dict['table'] = file_list
        else:
            print('No table file found,in path:{}, by charter:{}'.format(self.well_path, [self.WELL_NAME]+self.file_charter_dict['table']))
            self.file_path_dict['table'] = []

    # 测井文件读取，data_logging类型的初始化
    def data_logging_init(self, path):
        if path in list(self.logging_dict.keys()):
            return
        else:
            logging_data = data_logging(path=path, well_name=self.WELL_NAME)
            logging_data.read_data()
            # 井类型dataframe的data初始化
            self.logging_dict[path] = logging_data
            # 井表头初始化
            self.curve_names[path] = logging_data.get_curve_names()
            self.resolution[path] = logging_data._resolution

    # 表格资料读取，table_data类型的初始化
    def data_table_init(self, path):
        if path in list(self.table_dict.keys()):
            return
        else:
            table_data = data_table(path=path, well_name=self.WELL_NAME)
            table_data.read_data()
            # 如果是三线表的话，要初始化三线表的分辨率，分辨率就是用logging默认的分辨率
            if table_data._table_resolution < 0:
                table_data.set_table_resolution(self.get_logging_resolution())
            self.table_dict[path] = table_data

    # 获取测井数据
    def get_logging_data(self, well_key='', curve_names=[], Norm=False):
        # 如果测井资料为空，初始化测井资料
        if not self.logging_dict:
            # 判断是否存在合适的测井数据文件
            if len(self.file_path_dict['logging']) > 0:
                for path in self.file_path_dict['logging']:
                    self.data_logging_init(path=path)
            else:
                print('No Logging Data Found:{}'.format(self.file_path_dict['logging']))
                self.search_data_file()
                return pd.DataFrame()

        # 如果不为空的话，就要取第一个文件作为数据输出文件了
        well_value = self.get_default_dict(dict=self.logging_dict, key_default=well_key)
        if Norm:
            data_temp = well_value.get_data_normed(curve_names)
        else:
            data_temp = well_value.get_data(curve_names)
        return data_temp

    # def get_logging_data_normed(self, well_key='', curve_names=[]):
    #     # 如果测井资料为空，初始化测井资料
    #     if not self.logging_dict:
    #         # 判断是否存在合适的测井数据文件
    #         if len(self.file_path_dict['logging']) > 0:
    #             for path in self.file_path_dict['logging']:
    #                 self.data_logging_init(path=path)
    #         else:
    #             print('No Logging Data Found:{}'.format(self.file_path_dict['logging']))
    #             self.search_data_file()
    #             return pd.DataFrame()
    #
    #     # 如果不为空的话，就要取第一个文件作为数据输出文件了
    #     well_value = self.get_default_dict(dict=self.logging_dict, key_default=well_key)
    #     return

    # 获得dep-s，dep-e，type类型的表格数据
    def get_type_3(self, table_key='', curve_names=[]):
        # 如果字典表格数据为空，初始化表格数据字典
        if not self.table_dict:
            if len(self.file_path_dict['table']) > 0:
                for path in self.file_path_dict['table']:
                    self.data_table_init(path=path)
            else:
                self.search_data_file()
                return pd.DataFrame()

        table_value = self.get_default_dict(dict=self.table_dict, key_default=table_key)
        if table_value._data.shape[0] == 0:
            table_value.read_data()

        return table_value.get_table_3(curve_names=curve_names)

    # 获得depth，type类型的表格数据
    def get_type_2(self, table_key='', curve_names=[]):
        # 如果字典表格数据为空，初始化表格数据字典
        if not self.table_dict:
            if len(self.file_path_dict['table']) > 0:
                for path in self.file_path_dict['table']:
                    self.data_table_init(path=path)
            else:
                self.search_data_file()
                return pd.DataFrame()

        table_value = self.get_default_dict(dict=self.table_dict, key_default=table_key)
        if table_value._data.shape[0] == 0:
            table_value.read_data()

        return table_value.get_table_2(curve_names=curve_names)

    def combine_logging_table(self, well_key='', curve_names_logging=[], table_key='', curve_names_table=[], replace_dict={}, new_col='', Norm=False):
        if well_key == '':
            well_key = list(self.logging_dict.keys())[0]
        if table_key == '':
            table_key = list(self.table_dict.keys())[0]

        logging_value = self.get_logging_data(well_key=well_key, curve_names=curve_names_logging, Norm=Norm)

        # 现根据replace_dict更新表格数据table_value
        self.table_replace_update(table_key=table_key, replace_dict=replace_dict, new_col=new_col)

        # 再获得table_value，进行测井数据-表格数据的合并
        table_value = self.get_type_2(table_key=table_key, curve_names=curve_names_table)

        logging_columns = list(logging_value.columns)
        table_columns = list(table_value.columns)
        data_new = data_combine_table2col(logging_value.values, table_value.values, drop=True)

        data_columns = logging_columns + [table_columns[-1]]
        df_new = pd.DataFrame(data_new, columns=data_columns)
        self.update_data_with_type(well_key=well_key, df=df_new)
        return df_new

    # 更新well_value._data_with_type数据到well_value类里面
    def update_data_with_type(self, well_key='', df=pd.DataFrame()):
        well_value = self.get_default_dict(self.logging_dict, well_key)
        well_value._data_with_type = df

    def data_save(self, path='', new_sheet_name='', df=pd.DataFrame()):
        if path.endswith('.xlsx'):
            if df.shape[0] > 0:
                # 检查文件是否已存在
                try:
                    # 加载现有工作簿获取所有Sheet名称
                    with pd.ExcelFile(path) as xls:
                        existing_sheets = xls.sheet_names

                    # 生成唯一Sheet名称（避免覆盖）
                    if new_sheet_name in existing_sheets:
                        i = 1
                        while f"{new_sheet_name}_{i}" in existing_sheets:
                            i += 1
                        new_sheet_name = f"{new_sheet_name}_{i}"

                    # 以追加模式写入
                    with pd.ExcelWriter(path, engine='openpyxl', mode='a') as writer:
                        df.to_excel(writer, sheet_name=new_sheet_name, index=False)

                    print(f"成功添加新Sheet: [{new_sheet_name}]")

                except FileNotFoundError:
                    # 如果文件不存在则新建
                    df.to_excel(path, sheet_name=new_sheet_name, index=False)
                    print(f"新建文件并添加Sheet: [{new_sheet_name}]")

        elif path.endswith('.csv'):
            if df.shape[0] > 0:
                if os.path.exists(path):
                    print(f"已存在文件，未保存：{path}")
                else:
                    df.to_csv( path,
                               index=False,  # 不保存索引
                               encoding='utf-8-sig',  # 支持中文的编码格式
                               sep=',',  # CSV分隔符
                               quotechar='"',  # 文本限定符
                               float_format='%.4f')  # 浮点数格式
                    print(f"已创建新文件：{path}")

    def save_logging_data(self):
        # 保存的是保留了Type类型的测井数据
        for key in list(self.logging_dict.keys()):
            data_logging = self.get_default_dict(self.logging_dict, key)
            if data_logging._data.shape[0] > 0:
                # 原始测井数据的合并
                self.data_save(path=key, new_sheet_name=self.WELL_NAME, df=data_logging._data)
                path_combined = self.generate_new_path(key)
                # 筛选测井数据＋分类数据的合并
                self.data_save(path=path_combined, new_sheet_name=self.WELL_NAME, df=data_logging._data_with_type)



    # 通过读取测井数据文件，来初始化分辨率
    def get_logging_resolution(self, well_key=''):
        if not self.resolution:
            self.get_logging_data()

        # 获取默认分辨率
        resolution_default = self.get_default_dict(dict=self.resolution, key_default=well_key)
        return resolution_default


    def table_replace_update(self, table_key='', replace_dict={}, new_col=''):
        if table_key == '':
            table_key = list(self.table_dict.keys())[0]
        table_value = self.table_dict[table_key]
        table_value.table_type_replace(replace_dict=replace_dict, new_col=new_col)

    def get_default_dict(self, dict={}, key_default=''):
        if not dict:
            print('Empty dictionary get')
            return pd.DataFrame()
        if key_default == '':
            key_default = list(dict.keys())[0]
            value_default = dict[key_default]
        else:
            if key_default not in list(dict.keys()):
                for key in list(dict.keys()):
                    if key.__contains__(key_default):
                        key_default = key
                        value_default = dict[key]
            else:
                value_default = dict[key_default]
        return value_default

    def get_table_replace_dict(self, table_key=''):
        table_value = self.get_default_dict(self.table_dict, table_key)
        return table_value.replace_dict

    def get_curve_names(self, well_key=''):
        return self.get_default_dict(self.curve_names, well_key)

    # 生成新文件路径（自动添加时间戳保证唯一性）
    def generate_new_path(self, original_path):
        dir_path = os.path.dirname(original_path)  # 获取原文件目录
        base_name = os.path.basename(original_path)  # 获取原文件名
        name_part, ext = os.path.splitext(base_name)  # 拆分文件名和扩展名

        # 添加时间戳
        timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
        new_name = f"{name_part}{timestamp}{ext}"

        return os.path.join(dir_path, new_name)

    def get_logging_path_list(self):
        return self.file_path_dict['logging']
    def get_table_path_list(self):
        return self.file_path_dict['table']
    def get_FMI_path_list(self):
        return self.file_path_dict['FMI']
    def get_NMR_path_list(self):
        return self.file_path_dict['NMR']

# a = WELL(path_folder=r'C:\Users\Administrator\Desktop\算法测试-长庆数据收集\Code_input-O', well_charter='白75')
# print(a.get_logging_data().describe())


# WELL_TEST = WELL(path_folder=r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV\白75', WELL_NAME='白75')
# a = WELL_TEST.get_logging_data(curve_names=['AC', 'CNL', 'DEN', 'GR'])
# print(WELL_TEST.logging_dict)
# b = WELL_TEST.get_type_2()
# print(WELL_TEST.table_dict)
# print(b.head(10))
# c = WELL_TEST.get_type_3()
# print(c.head(10))
# # # print(WELL_TEST.get_logging_resolution())
# # # print(WELL_TEST.get())
# e = WELL_TEST.get_logging_data_normed(curve_names=['AC', 'CNL', 'DEN', 'GR'])
# print(e.head(10), '\n', a.head(10), '\n', e.describe())
#
# d = WELL_TEST.combine_logging_table(curve_names_logging=['AC', 'CNL', 'DEN'],
#     replace_dict={'中GR长英黏土质（泥岩）': 4, '中低GR长英质（砂岩）': 3, '富有机质长英质': 2, '富有机质黏土质': 1, '高GR富凝灰长英质（沉凝灰岩）': 1},
#                                     Norm=True)
# print(d.describe())


# print(WELL_TEST.get_table_replace_dict())
# print(WELL_TEST.resolution)
# print(WELL_TEST.get_curve_names())
# WELL_TEST.save_logging_data()
