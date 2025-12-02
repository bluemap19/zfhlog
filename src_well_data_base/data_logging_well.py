from src_file_op.dir_operation import search_files_by_criteria
from src_well_data_base.data_logging_FMI import DataFMI
from src_well_data_base.data_logging_normal import DataLogging
from src_well_data_base.data_logging_table import DataTable


class DATA_WELL:
    def __init__(self, path_folder='', WELL_NAME=''):
        # 初始化四种类型数据（测井数据、电成像数据、核磁数据、表格分类数据）的 数据实体存放，默认保存成{路径：dataframe}的字典格式
        # 存放格式为{logging_path1:dataframe1, logging_path2:dataframe2}
        self.logging_dict = {}
        self.table_dict = {}
        self.FMI_dict = {}
        self.NMR_dict = {}

        # 初始化每一个path对应的dataframe对应的目标文件header表头，即曲线名称{路径：[depth, GR, DEN]}的字典格式，这个保存的是所有文件的目标表头
        self.curve_names_target = {}
        # 初始化 测井数据+表格数据 的保存，依旧是{路径：dataframe}的字典格式保存
        self.logging_table = {}

        # 数据的分辨率保存，默认以{路径：float}字典格式
        self.resolution = {}

        # 初始化井名所在的路径
        self.well_path = path_folder

        # 如果没有传入井名，那么就默认取路径最后一个文件夹的名字作为井名称
        if WELL_NAME == '':
            self.WELL_NAME = path_folder.split('/')[-1].split('\\')[-1].split('.')[0]
        else:
            self.WELL_NAME = WELL_NAME

        # 配置四种格式文件的关键字，分别存放 测井数据文件关键词logging，表格数据关键词table，电成像数据关键词FMI， 核磁数据路径关键词NMR
        # 保存目标文件关键字,分别存放 测井数据文件关键词logging，表格数据关键词table，电成像数据关键词FMI， 核磁数据NMR
        self.LOGGING_CHARTER = ['logging']
        self.FMI_CHARTER = ['dyna', 'stat']
        self.NMR_CHARTER = ['nmr']
        self.TABLE_CHARTER = ['table', 'LITHO_TYPE']
        # 保存目标文件路径,分别存放 测井数据文件路径键logging，表格数据路径键table，电成像数据路径键FMI， 核磁数据路径键NMR

        # 存放格式为logging:[logging_path1, logging_path2], table:[table_path1, table_path2], FMI:[FMI_path1, FMI_path2], NMR:[NMR_path1, NMR_path2]
        self.path_list_logging = []
        self.path_list_table = []
        self.path_list_fmi = []
        self.path_list_NMR = []

        # 初始化之后，直接初始化本地路径下目标文件
        self.search_data_file()

    # 初始化self.file_path_dict字典，存在四种key，分别是‘logging’，‘table’，‘FMI’，‘NMR’每一个key对应一个list，list里面装的是对应目标测井文件格式的数据路径
    def search_data_file(self):
        # 搜寻测井数据所在文件
        file_list = search_files_by_criteria(self.well_path, name_keywords=self.LOGGING_CHARTER, file_extensions=['.xlsx', '.csv'])
        if len(file_list) == 0:
            print('No logging files found,in path:{}, by charter:{} {}'.format(self.well_path, self.WELL_NAME, self.LOGGING_CHARTER))
            self.path_list_logging = []
        if len(file_list) >= 1:
            print('Successed searching logging file as:{}'.format(file_list))
            self.path_list_logging = file_list

        # 搜寻表格数据所在文件
        file_list = search_files_by_criteria(self.well_path, name_keywords=self.TABLE_CHARTER, file_extensions=['.xlsx', '.csv'])
        if len(file_list) >= 1:
            print('successed searching table file as:{}'.format(file_list))
            self.path_list_table = file_list
        else:
            print('No table file found,in path:{}, by charter:{}'.format(self.well_path, self.TABLE_CHARTER))
            self.path_list_table = []

        # 搜索电成像图像数据
        file_list_fmi_dyna = search_files_by_criteria(self.well_path, name_keywords=self.FMI_CHARTER, file_extensions=['.txt'], all_keywords=False)
        if len(file_list_fmi_dyna) >= 1:
            print('successed searching fmi dyna file as:{}'.format(file_list_fmi_dyna))
            self.path_list_fmi = file_list_fmi_dyna
        else:
            print('No fmi dyna file found,in path:{}, by charter:{}'.format(self.well_path, self.FMI_CHARTER))
            self.path_list_fmi = []


    # 测井文件读取，data_logging类型的初始化
    def data_logging_init(self, path=''):
        # 数据体字典self.logging_dict已经存放了，初始化了，则直接返回就行了
        if path in list(self.logging_dict.keys()):
            return
        else:
            if path == '':
                path = self.path_list_logging[0]
            logging_data = DataLogging(path=path, well_name=self.WELL_NAME)
            # 井类型dataframe的data初始化
            self.logging_dict[path] = logging_data


    # 表格资料读取，table_data类型的初始化
    def data_table_init(self, path=''):
        # 数据体字典self.table_dict已经存放了，初始化了，则直接返回就行了
        if path in list(self.table_dict.keys()):
            return
        else:
            if path == '':
                path = self.path_list_table[0]
            table_data = DataTable(path=path, well_name=self.WELL_NAME)
            self.table_dict[path] = table_data

    # 成像资料读取， data_FMI 类型的初始化, 其key使用的是 path_stat+path_dyna
    def data_FMI_init(self, path_fmi=''):
        # 数据体字典 self.FMI_dict 已经存放了，初始化了，则直接返回就行了
        if path_fmi in list(self.FMI_dict.keys()):
            return
        else:
            if path_fmi == '':
                path_dyna = self.path_list_fmi[0]
            if (path_fmi is None):
                print('\033[31m' + 'Empty fmi file found,in path:{}'.format(path_fmi) + '\033[0m')
                exit(0)
            fmi_data = DataFMI(path_fmi=path_fmi)
            self.FMI_dict[path_fmi] = fmi_data

    def check_logging_files(self, well_key=''):
        # 如果测井资料为空，初始化测井资料
        if not self.logging_dict:
            if well_key == '':
                # 判断是否存在 目标测井数据文件
                if len(self.file_path_dict['logging']) > 0:
                    for path in self.file_path_dict['logging']:
                        self.data_logging_init(path=path)
                else:
                    print('No Logging Data Found:{}'.format(self.file_path_dict['logging']))
                    self.search_data_file()
                    return pd.DataFrame()
            else:
                self.data_logging_init(path=well_key)
        # logging data 类的存放器 dict 不为空
        else:
            # 已经初始化过了
            if well_key in list(self.logging_dict.keys()):
                return
            # 还没有初始化，则重新进行初始化，但是要求这个输入的的路径 well_key 在 file_path_dict['logging']中
            if well_key in self.file_path_dict['logging']:
                self.data_logging_init(path=well_key)

    # 获取测井数据，根据文件路径+曲线名称，返回dataframe格式的测井数据
    def get_logging_data(self, well_key='', curve_names=[], Norm=False):
        self.check_logging_files(well_key)

        # 如果不为空的话，就要取第一个文件作为数据输出文件了
        well_value = self.get_default_dict(dict=self.logging_dict, key_default=well_key)
        if Norm:
            data_temp = well_value.get_data_normed(curve_names)
        else:
            data_temp = well_value.get_data(curve_names)
        return data_temp

    def check_table_files(self, table_key=''):
        # 如果字典表格数据为空，初始化表格数据字典
        if not self.table_dict:
            if table_key == '':
                if len(self.file_path_dict['table']) > 0:
                    for path in self.file_path_dict['table']:
                        self.data_table_init(path=path)
                else:
                    self.search_data_file()
                    return
            else:
                self.data_table_init(path=table_key)
        # table data 类的存放器 dict 不为空
        else:
            # 已经初始化过了
            if table_key in list(self.table_dict.keys()):
                return
            # 还没有初始化，则重新进行初始化，但是要求这个输入的的路径 table_key 在 file_path_dict['table']中
            if table_key in self.file_path_dict['table']:
                self.data_table_init(path=table_key)

    # 获得dep-s，dep-e，type类型的表格数据
    def get_type_3(self, table_key='', curve_names=[]):
        self.check_table_files(table_key)

        table_value = self.get_default_dict(dict=self.table_dict, key_default=table_key)
        if table_value._data.empty:
            table_value.read_data()

        return table_value.get_table_3(curve_names=curve_names)

    def get_type_3_replaced(self, table_key='', curve_names=[], replace_dict={}, new_col=''):
        self.check_table_files(table_key)

        table_value = self.get_default_dict(dict=self.table_dict, key_default=table_key)
        if table_value._data.empty:
            table_value.read_data()

        table_value.table_type_replace(replace_dict=replace_dict, new_col=new_col)
        return table_value.get_table_3_replaced(curve_names=curve_names)

    # 获得depth，type类型的表格数据
    def get_type_2(self, table_key='', curve_names=[]):
        self.check_table_files(table_key)

        table_value = self.get_default_dict(dict=self.table_dict, key_default=table_key)
        if table_value._data.empty:
            table_value.read_data()

        return table_value.get_table_2(curve_names=curve_names)

    def get_type_2_replaced(self, table_key='', curve_names=[], replace_dict={}, new_col=''):
        self.check_table_files(table_key)

        table_value = self.get_default_dict(dict=self.table_dict, key_default=table_key)
        if table_value._data.empty:
            table_value.read_data()

        table_value.table_type_replace(replace_dict=replace_dict, new_col=new_col)
        return table_value.get_table_2_replaced(curve_names=curve_names)

    def check_fmi_files(self, fmi_stat_key='', fmi_dyna_key=''):
        # 如果字典表格数据为空，初始化表格数据字典
        if not self.FMI_dict:
            self.data_FMI_init(path_stat=fmi_stat_key, path_dyna=fmi_dyna_key)
        # FMI data 类的存放器 dict 不为空
        else:
            # 已经初始化过了
            if fmi_stat_key + fmi_dyna_key in list(self.FMI_dict.keys()):
                return
            # 还没有初始化，则重新进行初始化，但是要求这个输入的的路径 fmi_stat_key 在 file_path_dict['fmi_stat']中
            if (fmi_stat_key in self.file_path_dict['fmi_stat']) and (
                    fmi_dyna_key in self.file_path_dict['fmi_dyna']):
                self.data_FMI_init(path_stat=fmi_stat_key, path_dyna=fmi_dyna_key)
            if len(fmi_stat_key + fmi_dyna_key) < 2:
                self.data_FMI_init(path_stat=fmi_stat_key, path_dyna=fmi_dyna_key)
            else:
                print('\033[31m' + 'PATH {}&{} Not In This WELL PATH LIST:{}&{}'.format(fmi_stat_key, fmi_dyna_key,
                                                                                        self.file_path_dict[
                                                                                            'fmi_stat'],
                                                                                        self.file_path_dict[
                                                                                            'fmi_dyna']) + '\033[0m')

    def get_fmi_data(self, fmi_stat_key='', fmi_dyna_key='', mode='ALL'):
        self.check_fmi_files(fmi_stat_key=fmi_stat_key, fmi_dyna_key=fmi_dyna_key)

        fmi_value = self.get_default_dict(dict=self.FMI_dict, key_default=fmi_stat_key + fmi_dyna_key)
        data_all = fmi_value.get_data(mode=mode)

        return data_all