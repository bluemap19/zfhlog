from datetime import datetime

import pandas as pd
import numpy as np
from src_data_process.resistivity_correction import scale_by_quantiles_use_config
from src_file_op.dir_operation import search_files_by_criteria
from Remove_temp_influence import correction_by_tempture


def find_closest_match(df, temp_value, source_lower_value):
    """
    在DataFrame中查找与给定温度和数据低值最匹配的行

    参数:
    df: 包含数据的DataFrame
    temp_value: 目标温度值
    source_lower_value: 目标数据低值

    返回:
    最匹配行的Series
    """
    # 计算温度差异的绝对值
    temp_diff = np.abs(df['TEMP'] - temp_value)/df['TEMP'].values

    # 计算数据低值差异的绝对值
    source_lower_diff = np.abs(df['source_lower'] - source_lower_value)/df['source_lower'].values

    # 计算综合差异（加权平均）
    # 权重可根据实际需求调整
    total_diff = (0.95 * temp_diff) + (0.05 * source_lower_diff)

    # 找到最小差异的索引
    min_index = total_diff.idxmin()

    return df.loc[min_index]


def correct_res_by_tempture(df_source, df_config):
    df_source = correction_by_tempture(df_source)
    print(df_source.describe())

    quantile = 0.2
    windows_length = 10
    for i in range(df_source.shape[0]):
        tempture_t = df_source['TEMP'][i]

        windows_s = max(0, i-windows_length)
        windows_e = min(i + windows_length, df_source.shape[0]-1)
        data_windows = df_source['R_temp_sub'][windows_s:windows_e].values
        data_index = df_source['R_temp_sub'][i]
        source_lower = np.percentile(data_windows, quantile * 100)

        closest_row = find_closest_match(df_config, tempture_t, source_lower)
        print(closest_row['target_lower'])

        scaled_result = scale_by_quantiles_use_config(data_index, config={'target_lower':closest_row['target_lower'], 'source_lower':source_lower, 'scale_factor':closest_row['scale_factor']})

        df_source.at[i, 'R_gauss'] = scaled_result
        print(closest_row)
        print(scaled_result)

    return df_source



# 示例使用
if __name__ == "__main__":
    path_temp_config = search_files_by_criteria(search_root=r'C:\Users\ZFH\Desktop\电阻率校正-07.31',
                                            name_keywords=['config_tempture'], file_extensions=['xlsx', 'csv'])
    path_temp_config = path_temp_config[0]
    if path_temp_config.endswith('xlsx'):
        df_config = pd.read_excel(path_temp_config, sheet_name=0, engine='openpyxl')
    elif path_temp_config.endswith('csv'):
        df_config = pd.read_csv(path_temp_config, encoding='gbk')
    else:
        print('Not find excel file')
        exit(0)

    file_name = r'原始数据-4.xlsx'
    path_measured_logging = r'C:\Users\ZFH\Desktop\电阻率校正-07.31' + '\\' + file_name
    df_source = pd.read_excel(path_measured_logging, sheet_name=0)
    print(df_source.describe())

    # 1. 定义映射关系（原始列名:新列名）
    column_mapping = {
        'MD':'DEPTH',
        '温度-MWD':'TEMP',
        '电阻率-原始-近钻头':'R_temp',
    }
    df_source.rename(columns=column_mapping, inplace=True)
    print(df_source.columns)
    print(df_source[['DEPTH', 'TEMP', 'R_temp']].describe())
    df_source['R_temp_sub'] = 0.0
    df_source['R_gauss'] = 0.0

    df_source = correct_res_by_tempture(df_source, df_config)
    print(df_source.describe())

    column_mapping_inverted = {value: key for key, value in column_mapping.items()}
    df_source.rename(columns=column_mapping_inverted, inplace=True)
    time_str = datetime.now().strftime("%m-%d-%H-%M")
    df_source.to_csv(path_measured_logging.replace(file_name, f'data_pred_{time_str}.csv'), index=False, float_format='%.4f')