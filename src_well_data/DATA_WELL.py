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

        self.logging = {}
        self.table = {}
        self.resolution = {}
        self.logging_table = {}

        self.well_path = path_folder
        self.curve_names = []
        self.curve_names_target = []
        if WELL_NAME == '':
            self.WELL_NAME = path_folder.split('/')[-1].split('\\')[-1].split('.')[0]
        else:
            self.WELL_NAME = WELL_NAME
        self.file_charter_dict = {'logging':['_logging'], 'FMI':['FMI_dyna', 'FMI_stat'], 'NMR':['NMR'], 'table':['_LITHO_TYPE']} # 保存目标文件关键字,分别存放 测井数据文件关键词logging，表格数据关键词table，电成像数据关键词FMI， 核磁数据NMR
        self.file_path_dict = {}                # 保存目标文件路径,分别存放 测井数据文件路径键logging，表格数据路径键table，电成像数据路径键FMI， 核磁数据路径键NMR

        # 初始化之后，直接寻找一下本地路径下是否含有需要的文件
        self.search_logging_data_file()

    def get_logging_data(self, well_key='', curve_names=[]):
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
                self.search_logging_data_file()
                return pd.DataFrame()

        # 如果不为空的话，就要取第一个文件作为数据输出文件了
        if well_key == '':
            well_key = list(self.logging.keys())[0]
        print(well_key, self.logging)
        well_value = self.logging[well_key]
        if well_value.data.shape[0] == 0:
            well_value.read_data()

        return well_value.get_data(curve_names)

    def get_table_3(self, table_key=''):
        # 如果字典表格数据为空，初始化表格数据字典
        if not self.table:
            if len(self.file_path_dict['table']) > 0:
                for path in self.file_path_dict['table']:
                    table_data = data_table(path=path, well_name=self.WELL_NAME)
                    table_data.read_data()
                    if table_data.table_resolution < 0:
                        table_data.set_table_resolution(self.get_logging_resolution())
                    self.table[path] = table_data
            else:
                self.search_logging_data_file()
                return pd.DataFrame()

        if table_key == '':
            table_key = list(self.table.keys())[0]
        table_value = self.table[table_key]
        if table_value.data.shape[0] == 0:
            table_value.read_data()

        return table_value.get_table_3()

    def get_table_2(self, table_key=''):
        # 如果字典表格数据为空，初始化表格数据字典
        if not self.table:
            if len(self.file_path_dict['table']) > 0:
                for path in self.file_path_dict['table']:
                    table_data = data_table(path=path, well_name=self.WELL_NAME)
                    table_data.read_data()
                    if table_data.table_resolution < 0:
                        table_data.set_table_resolution(self.get_logging_resolution())
                    self.table[path] = table_data
            else:
                self.search_logging_data_file()
                return pd.DataFrame()

        if table_key == '':
            table_key = list(self.table.keys())[0]
        table_value = self.table[table_key]
        if table_value.data.shape[0] == 0:
            table_value.read_data()

        return table_value.get_table_2()

    def combine_logging_table(self, logging_key='', table_key='', curve_names=[], replace_dict={}, new_col=''):
        if logging_key == '':
            well_key = next(iter(self.logging))
        if table_key == '':
            table_key = next(iter(self.table))

        logging_value = self.get_logging_data(well_key=well_key, curve_names=curve_names)
        table_value = self.get_table_2(table_key=table_key)
        # self.table_replace_update(table_key=table_key, replace_dict=replace_dict, new_col=new_col)
        logging_columns = list(logging_value.columns)
        table_columns = list(table_value.columns)
        data_new = data_combine_table2col(logging_value.values, table_value.values, drop=True)
        # data_new = data_combine_new([logging_value.values, table_value.values], step='MIN')

        data_columns = logging_columns + [table_columns[-1]]
        df_new = pd.DataFrame(data_new, columns=data_columns)
        self.logging_table[well_key] = df_new
        return df_new

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
            file_list = search_files_by_criteria(self.well_path, name_keywords=['Logging_ALL'], file_extensions=['.xlsx'])

        if len(file_list) == 0:
            print('No logging files found,in path:{}, by charter:{} {}'.format(self.well_path, self.WELL_NAME, self.file_charter_dict['logging']))
            self.file_path_dict['logging'] = []
        if len(file_list) >= 1:
            print('Successed searching logging file as:{}'.format(file_list))
            self.file_path_dict['logging'] = file_list


        # 搜寻表格数据所在文件
        file_list = search_files_by_criteria(self.well_path, name_keywords=[self.WELL_NAME]+self.file_charter_dict['table'], file_extensions=['.xlsx'])
        if len(file_list) == 0:
            file_list = search_files_by_criteria(self.well_path, name_keywords=['_LITHO_TYPE'], file_extensions=['.xlsx'])

        if len(file_list) >= 1:
            print('successed searching file as:{}'.format(file_list))
            self.file_path_dict['table'] = file_list
        else:
            print('No table file found,in path:{}, by charter:{}'.format(self.well_path, [self.WELL_NAME]+self.file_charter_dict['table']))
            self.file_path_dict['table'] = []

    def table_replace_update(self, table_key='', replace_dict={}, new_col=''):
        if table_key == '':
            table_key = list(self.table.keys())[0]
        table_value = self.table[table_key]
        table_value.table_replace(replace_dict=replace_dict, new_col=new_col)

# a = WELL(path_folder=r'C:\Users\Administrator\Desktop\算法测试-长庆数据收集\Code_input-O', well_charter='白75')
# print(a.get_logging_data().describe())

WELL_TEST = WELL(path_folder=r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\Code_input-O\白75', WELL_NAME='白75')
a = WELL_TEST.get_logging_data()
# # print(a.head(10))
# print(a.head(10))
b = WELL_TEST.get_table_2()
# print(b.head(10))
# print(b.head(10))
c = WELL_TEST.get_table_3()
# print(c.head(10))
# print(WELL_TEST.get_logging_resolution())
# print(WELL_TEST.get())
d = WELL_TEST.combine_logging_table(curve_names=['AC', 'CNL', 'DEN'])
print(d.describe())