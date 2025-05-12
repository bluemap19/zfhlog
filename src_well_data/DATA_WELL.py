import pandas as pd

from src_file_op.dir_operation import search_files_by_criteria
from src_logging.curve_preprocess import get_resolution_by_depth
from src_well_data.data_logging import data_logging
from src_well_data.data_table import data_table


class WELL:
    def __init__(self, path_folder='', WELL_NAME=''):
        self.describe_logging = {}
        self.describe_FMI = {}
        self.describe_NMR = {}
        self.describe_table = {}

        self.logging = {}
        self.table = {}
        self.resolution = {}

        self.well_path = path_folder
        self.curve_names = []
        self.curve_names_target = []
        if WELL_NAME == '':
            self.WELL_NAME = path_folder.split('/')[-1].split('\\')[-1].split('.')[0]
        else:
            self.WELL_NAME = WELL_NAME
        self.file_charter_dict = {'logging':[''], 'FMI':['FMI_dyna', 'FMI_stat'], 'NMR':['NMR'], 'table':['_LITHO_TYPE']} # 保存目标文件关键字,分别存放 测井数据文件关键词logging，表格数据关键词table，电成像数据关键词FMI， 核磁数据NMR
        self.file_path_dict = {}                # 保存目标文件路径,分别存放 测井数据文件路径键logging，表格数据路径键table，电成像数据路径键FMI， 核磁数据路径键NMR

        # 初始化之后，直接寻找一下本地路径下是否含有需要的文件
        self.search_logging_data_file()

    def get_logging_data(self, curve_names=[]):
        # 如果测井资料为空，初始化测井资料
        if not self.logging:
            # 判断是否存在合适的测井数据文件
            if len(self.file_path_dict['logging']) > 0:
                for path in self.file_path_dict['logging']:
                    logging_data = data_logging(path=path, well_name=self.WELL_NAME)
                    logging_data.read_data()
                    self.logging[path] = logging_data
                    self.resolution[path] = get_resolution_by_depth(logging_data.data.iloc[:, 0].values)
            else:
                print('No Logging Data Found:{}'.format(self.file_path_dict['logging']))
                return None

        # 如果不为空的话，就要取第一个文件作为数据输出文件了
        first_well_key = next(iter(self.logging))
        first_well_value = self.logging[first_well_key]
        if first_well_value.data.shape[0] == 0:
            first_well_value.read_data()

        return first_well_value.get_data(curve_names)

    def get_table_3(self):
        # 如果字典表格数据为空，初始化表格数据字典
        if not self.table:
            if len(self.file_path_dict['table']) > 0:
                for path in self.file_path_dict['table']:
                    table_data = data_table(path=path, well_name=self.WELL_NAME, resolution=self.get_logging_resolution())
                    table_data.read_data()
                    self.table[path] = table_data

        # 如果不为空的话，就要取第一个文件作为数据输出文件了
        first_table_key = next(iter(self.table))
        first_table_value = self.table[first_table_key]
        if first_table_value.data.shape[0] == 0:
            first_table_value.read_data()

        return first_table_value.get_table_3()

    def get_table_2(self):
        # 如果字典表格数据为空，初始化表格数据字典
        if not self.table:
            if len(self.file_path_dict['table']) > 0:
                for path in self.file_path_dict['table']:
                    table_data = data_table(path=path, well_name=self.WELL_NAME, resolution=self.get_logging_resolution())
                    table_data.read_data()
                    self.table[path] = table_data

        # 如果不为空的话，就要取第一个文件作为数据输出文件了
        first_table_key = next(iter(self.table))
        first_table_value = self.table[first_table_key]
        if first_table_value.data.shape[0] == 0:
            first_table_value.read_data()

        return first_table_value.get_table_2()

    # 通过读取测井数据文件，来初始化分辨率
    def get_logging_resolution(self):
        if not self.resolution:
            self.get_logging_data()

        # 获取第一个键
        first_key = next(iter(self.resolution))
        resolution_first = self.resolution[first_key]
        return resolution_first

    def search_logging_data_file(self):
        # 搜寻测井数据所在文件
        file_list = search_files_by_criteria(self.well_path, name_keywords=[self.WELL_NAME]+self.file_charter_dict['logging'], file_extensions=['.xlsx'])
        if len(file_list) == 0:
            print('No files found,in path:{}, by charter:{} and {}'.format(self.well_path, self.WELL_NAME, self.file_charter_dict['logging']))
            print('Now searching file use charter \'Logging_ALL\'')
            file_list = search_files_by_criteria(self.well_path, name_keywords=['Logging_ALL'], file_extensions=['.xlsx'])
        if len(file_list) >= 1:
            print('successed searching file as:{}'.format(file_list))
            self.file_path_dict['logging'] = file_list
        else:
            print('no file found,in path:{}, by charter:{}'.format(self.well_path, file_list))


        # 搜寻表格数据所在文件
        file_list = search_files_by_criteria(self.well_path, name_keywords=[self.WELL_NAME]+self.file_charter_dict['table'], file_extensions=['.xlsx'])
        if len(file_list) == 0:
            print('No table files found, in path:{}, by charter:{} and {}'.format(self.well_path, self.WELL_NAME, self.file_charter_dict['table']))
            print('Now searching table file use charter \'Table_ALL\'')
            file_list = search_files_by_criteria(self.well_path, name_keywords=['_LITHO_TYPE'], file_extensions=['.xlsx'])
        if len(file_list) >= 1:
            print('successed searching file as:{}'.format(file_list))
            self.file_path_dict['table'] = file_list
        else:
            print('no table file found,in path:{}, by charter:{}'.format(self.well_path, [self.WELL_NAME]+self.file_charter_dict['table']))

# a = WELL(path_folder=r'C:\Users\Administrator\Desktop\算法测试-长庆数据收集\Code_input-O', well_charter='白75')
# print(a.get_logging_data().describe())