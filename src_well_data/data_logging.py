import pandas as pd


class data_logging:
    def __init__(self, path='', well_name=''):
        self.data = pd.DataFrame()
        self.curve_names = []
        self.file_path = path
        self.table_name = well_name

    def read_data(self, file_path='', table_name=''):
        if file_path == '':
            file_path = self.file_path
        if table_name == '':
            table_name = self.table_name

        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            self.data = pd.read_excel(file_path, sheet_name=table_name)

        self.curve_names = list(self.data.columns)

    def save_data(self, file_path='', table_name=''):

        pass

    def get_data(self, curve_names=[]):
        if self.data.empty:
            self.read_data()
        if curve_names == []:
            curve_names = self.curve_names
        return self.data[curve_names]