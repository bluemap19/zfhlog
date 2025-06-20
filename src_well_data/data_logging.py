import pandas as pd
from src_logging.curve_preprocess import get_resolution_by_depth, data_Normalized


class data_logging:
    def __init__(self, path='', well_name=''):
        self._data = pd.DataFrame()
        self._data_normed = pd.DataFrame()
        self._curve_names = []
        self._file_path = path
        self._logging_name = well_name
        self._data_with_type = pd.DataFrame()        # 选取的self.data[]+table_type（data_table数据）， 测井数据＋类别数据 结果
        self._resolution = -1
        self.mapping_dict = {
            'depth': ['#Depth', 'Depth', 'DEPTH', 'depth'],
            'CAL': ['CAL', 'CALC', 'CALX', 'CALY'],
            'SP': ['SP', 'Sp'],
            'GR': ['GR', 'GRC'],
            'CNL': ['CNL', 'CN'],
            'DEN': ['DEN', 'DENC'],
            'DT': ['DT', 'DT24', 'DTC', 'AC', 'Ac'],
            'RXO': ['RXO', 'Rxo', 'RXo', 'RS', 'Rs'],
            'RD': ['RD', 'Rd', 'Rt', 'RT'],
        }

    def read_data(self, file_path='', table_name=''):
        if file_path == '':
            file_path = self._file_path
        if table_name == '':
            table_name = self._logging_name

        if file_path.endswith('.csv'):
            self._data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            try:
                self._data = pd.read_excel(file_path, sheet_name=table_name)
            except Exception as e:
                print('文件读取失败', e)
                self._data = pd.DataFrame()

        # 分辨率初始化
        self._resolution = get_resolution_by_depth(self._data.iloc[:, 0].values)
        # 表格头初始化
        self._curve_names = list(self._data.columns)

    def save_data(self, file_path='', table_name=''):

        pass

    def get_data(self, curve_names=[]):
        if self._data.empty:
            self.read_data()
        if curve_names == []:
            curve_names = self._curve_names
        if not curve_names[0].lower().__contains__('depth'):
            curve_names = [self._curve_names[0]] + curve_names

        curve_names = self.input_cols_mapping(curve_names, list(self._data.columns))
        return self._data[curve_names]

    def get_data_normed(self, curve_names=[]):
        if self._data_normed.empty:
            data_target = self.get_data(curve_names)
            cols_new_temp = list(data_target.columns)
            data_normed_temp = data_Normalized(data_target.values, DEPTH_USE=True, logging_range=[-999, 9999], max_ratio=0.01, local_normalized=False)
            self._data_normed = pd.DataFrame(data_normed_temp, columns=cols_new_temp)
        else:
            pass

        return self._data_normed

    def input_cols_mapping(self, input_cols=[], target_cols=[]):
        if target_cols == []:
            target_cols = list(self._data.columns)
        input_mapping_cols = input_cols.copy()

        # 遍历 l1，保留不在 l2 中的元素
        unindex_result = [element for element in input_cols if element not in target_cols]

        processed_cols = []
        for element in reversed(unindex_result):
            idx = input_mapping_cols.index(element)
            for target, replacement in self.mapping_dict.items():
                if element in replacement:
                    # 转换为集合求交集
                    intersection = list(set(replacement) & set(target_cols))
                    print(element, '--->', intersection)
                    input_mapping_cols[idx] = intersection[0]
                    processed_cols.append(element)
                    break

        unprocessed_cols = [element for element in unindex_result if element not in processed_cols]
        if unprocessed_cols:
            print('Exist unprocessable cols:', unprocessed_cols)
            exit(0)

        return input_mapping_cols

    def get_curve_names(self):
        return self._curve_names
