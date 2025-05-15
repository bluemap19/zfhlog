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
        if curve_names[0].lower().__contains__('depth'):
            return self._data[curve_names]
        else:
            curve_names = [self._curve_names[0]] + curve_names
            return self._data[curve_names]

    def get_data_normed(self, curve_names=[]):
        if self._data_normed.empty:
            data_target = self.get_data(curve_names)
            cols_new_temp = list(data_target.columns)
            data_normed_temp = data_Normalized(data_target.values, DEPTH_USE=True, logging_range=[-999, 9999], max_ratio=0.01)
            self._data_normed = pd.DataFrame(data_normed_temp, columns=cols_new_temp)
        else:
            pass

        return self._data_normed

    def get_curve_names(self):
        return self._curve_names
