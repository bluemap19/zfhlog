import pandas as pd

from src_logging.curve_preprocess import get_resolution_by_depth


class data_table:
    def __init__(self, path='', well_name='', resolution=0.1):
        self.table_2 = pd.DataFrame()
        self.table_3 = pd.DataFrame()
        self.table_resolution = resolution
        self.file_path = path
        self.well_name = well_name
        self.data = pd.DataFrame()

    def read_data(self, file_path='', table_name='', TYPE2=False, TYPE3=False):
        if file_path == '':
            file_path = self.file_path
        if table_name == '':
            table_name = self.well_name

        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            self.data = pd.read_excel(file_path, sheet_name=table_name)

        if TYPE2:
            self.table_2 = self.data
        if TYPE3:
            self.table_3 = self.data
        else:
            if self.data.shape[1] == 2:
                self.table_2 = self.data
            elif self.data.shape[1] == 3:
                self.table_3 = self.data
            else:
                print('ALARM TABLE　FORMAT:{}'.format(self.data.columns))
                self.table_3 = self.data

        if self.table_3.shape[0] > 0:
            self.table_3_to_2()

        elif self.table_2.shape[0] > 0:
            self.table_resolution = get_resolution_by_depth(self.table_2.iloc[:, 0].values)
            self.table_2_to_3()

    def table_2_to_3(self):
        pass
    def table_3_to_2(self):
        pass
    def get_table_3(self):
        return self.table_3
    def get_table_2(self):
        return self.table_2