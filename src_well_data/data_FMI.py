import os
import numpy as np
import pandas as pd


class data_FMI:
    def __init__(self, path_dyna='', path_stat='', well_name=''):
        self._table_2 = pd.DataFrame()
        self._well_name = well_name
        self._data_dyna = pd.DataFrame()     # 存放的是从文件中读取的原始dataframe
        self._data_stat = pd.DataFrame()
        self._resolution = 0.0025
        self._texture_dyna = pd.DataFrame()
        self._texture_stat = pd.DataFrame()
        self._path_dyna = path_dyna
        self._path_stat = path_stat

        if not os.path.isfile(path_dyna):
            print('File not found or does not exist:{}'.format(path_dyna))
        if not os.path.isfile(path_stat):
            print('File not found or does not exist:{}'.format(path_stat))

    def read_data(self, file_path=''):
        if os.path.isfile(file_path):
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, index_col=0)
            elif file_path.endswith('.txt'):
                df = np.loadtxt(file_path, delimiter='\t', skiprows=8)
                data = np.array(df)
            else:
                print('Error file extension as:{}'.format(file_path.split('\\')[-1]))

            return data
        else:
            print('File path does not exist as:{}'.format(file_path))

    def get_data(self, mode='DYNA'):
        if mode == 'DYNA':
            if self._data_dyna.empty:
                self._data_dyna = self.read_data(self._path_dyna)
            return {'DYNA':self._data_dyna}
        elif mode == 'STAT':
            if self._data_stat.empty:
                self._data_stat = self.read_data(self._path_stat)
            return {'STAT':self._data_stat}
        elif mode == 'ALL':
            if self._data_dyna.empty:
                self._data_dyna = self.read_data(self._path_dyna)
            if self._data_stat.empty:
                self._data_stat = self.read_data(self._path_stat)

            return {'DYNA':self._data_dyna, 'STAT':self._data_stat}
        else:
            print('ERROR MODE')
            exit(0)

    def get_texture(self, ):

        pass

if __name__ == '__main__':
    test_table = data_FMI(
        path_dyna=r'F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\城96_FMI_Main 1700-2120_dyna.txt',
        path_stat=r'F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\城96_FMI_Main 1700-2120_stat.txt',
        well_name='城96')

    DYNA = test_table.get_data(mode='DYNA')
    print(DYNA['DYNA'].shape)
    STAT = test_table.get_data(mode='STAT')
    print(STAT['STAT'].shape)

