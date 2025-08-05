import csv
import math
from datetime import datetime
import numpy as np
import pandas as pd
from src_data_process.OLS1 import nonlinear_fitting
from src_data_process.data_filter import remove_static_depth_data
from src_data_process.resistivity_correction import scale_gaussian, scale_gaussian_by_config, scale_by_quantiles_old, \
    scale_by_quantiles
from src_file_op.dir_operation import search_files_by_criteria
from Remove_temp_influence import offset_linear, offset_power, correction_by_tempture


def window_process_gauss_scaling(df, PRED_GAUSS_SETTING={}, window_work_length=500,
                                 windows_step=100, windows_view = 1.1):
    windows_num = math.floor((df.shape[0] - window_work_length)/windows_step) + 1
    config_all = []

    for i in range(windows_num):
        window_index = i*windows_step + window_work_length//2
        window_work_start = np.max((0, window_index - window_work_length // 2))
        window_work_end = np.min((df.shape[0] - 1, window_index + window_work_length // 2 + 1))

        window_view_length = int(windows_view * window_work_length)
        window_view_start = np.max((0, window_index - window_view_length // 2))
        window_view_end = np.min((df.shape[0] - 1, window_index + window_view_length // 2 + 1))

        print(f'windows index:{window_index}; work window length:{window_work_length}; index from:{window_work_start} to {window_work_end};'
              f'depth from:{df.DEPTH[window_work_start]} to {df.DEPTH[window_work_end]}, view window length:{window_view_length}, from:{window_view_start} to {window_view_end}'
              )
        df_window = df.iloc[window_view_start:window_view_end].copy()

        if PRED_GAUSS_SETTING:
            df_window['R_gauss'], stats = scale_gaussian_by_config(source_data=df['R_temp_sub'],
                                                                target_data_config=PRED_GAUSS_SETTING,
                                                                return_stats=True)
        else:
            # 使用高斯缩放，对数据范围进行处理
            # df['R_gauss'], config = scale_gaussian(source_data=df['R_temp_sub'], target_data=df['R_real'], return_stats=True)
            df_window['R_gauss'], config = scale_by_quantiles(source_data=df_window['R_temp_sub'], target_data=df_window['R_real'])

        # config.append(df.at[window_index, 'TEMP'])
        # config_all.append(config)
        # dataframe_config = pd.DataFrame(config_all, columns=['target_lower', 'source_lower', 'scale_factor','TEMP'])
        # dataframe_config.to_csv('config_tempture.csv', index=False)

        EFFECTIVE_LENGTH = window_work_end - window_work_start
        df.iloc[window_work_start:window_work_end] = df_window.iloc[
                                                     window_work_start - window_view_start:window_work_start - window_view_start + EFFECTIVE_LENGTH]


    return df



# 只使用数据拟合减去基线，得到 测量数据偏移值
def fit_r_pred_window_gauss_scaling(df, PRED_GAUSS_SETTING={}, offset_function='power', gauss_windows=200):
    # 确保输入包含所需列
    if PRED_GAUSS_SETTING:
        required_cols = ['R_temp', 'TEMP', 'R_gauss']
    else:
        required_cols = ['R_temp', 'TEMP', 'R_real', 'R_gauss']
    assert all(col in df.columns for col in required_cols), "DataFrame缺少必要列"

    # 执行拟合,先尝试 幂函数R_temp' = A * TEMP^B，再尝试 线性函数 R_temp' = A * TEMP + B
    if offset_function.lower() == 'linear':
        try:
            df = offset_linear(df)
        except Exception as e:
            print(f"Method power failed: {str(e)}, now try linear formula")
            df = offset_power(df)
    elif offset_function.lower() == 'power':
        try:
            df = offset_power(df)
        except Exception as e:
            print(f"Method power failed: {str(e)}, now try linear formula")
            df = offset_linear(df)
    elif offset_function.lower() == 'tempture':
        try:
            df = correction_by_tempture(df)
        except Exception as e:
            print(f"Method power failed: {str(e)}, now try linear formula")

    df = window_process_gauss_scaling(df, PRED_GAUSS_SETTING,
                                      window_work_length=gauss_windows, windows_step=10, windows_view=1.2)

    return df



if __name__ == '__main__':
    # path_logging = search_files_by_criteria(search_root=r'C:\Users\ZFH\Desktop\电阻率校正-坨73-斜13井',
    path_logging = search_files_by_criteria(search_root=r'C:\Users\ZFH\Desktop\电阻率校正-07.31\SOF003',
                                            name_keywords=['logging_data'], file_extensions=['xlsx', 'csv'])
    path_logging = path_logging[0]
    if path_logging.endswith('xlsx'):
        df = pd.read_excel(path_logging, sheet_name=0, engine='openpyxl')
    elif path_logging.endswith('csv'):
        df = pd.read_csv(path_logging, encoding='gbk')
    else:
        print('Not find excel file')
        exit(0)
    file_name = path_logging.split('\\')[-1]

    # 1. 定义映射关系（原始列名:新列名）
    column_mapping = {
        '#DEPTH':'DEPTH',
        'TEMP':'TEMP',
        'ResFar-矫正':'R_temp',
        'ILD':'R_real',
    }
    df.rename(columns=column_mapping, inplace=True)
    print(df.columns)
    print(df[['DEPTH', 'TEMP', 'R_temp', 'R_real']].describe())
    df['R_temp_sub'] = 0.0
    df['R_gauss'] = 0.0

    df = remove_static_depth_data(df, depth_col='DEPTH')

    gauss_windows = 200
    df = fit_r_pred_window_gauss_scaling(df, PRED_GAUSS_SETTING={}, offset_function='tempture', gauss_windows=gauss_windows)
    # df = fit_r_pred_window_gauss_scaling(df, PRED_GAUSS_SETTING={}, offset_function='linear', gauss_windows=gauss_windows)

    # # 创建筛选条件
    # condition = (df['TEMP'] < 27.5) | (df['TEMP'] > 47.6)
    #
    # # 将符合条件的res列值设为NaN
    # df.loc[condition, 'R_temp_sub'] = np.nan
    # df.loc[condition, 'R_gauss'] = np.nan

    # df = log_predv_nearto_realv(df, window_length=11, SHIFT_RATIO=0.3, pred_col='R_gauss', real_col='R_real')
    column_mapping_inverted = {value: key for key, value in column_mapping.items()}
    df.rename(columns=column_mapping_inverted, inplace=True)
    time_str = datetime.now().strftime("%m-%d-%H-%M")
    df.to_csv(path_logging.replace(file_name, f'data_corrected_{gauss_windows}_{time_str}.csv'), index=False, float_format='%.4f', quoting=csv.QUOTE_NONE)

    # # 调用可视化接口
    # visualize_well_logs(
    #     data=logging_data,
    #     depth_col='DEPTH',
    #     curve_cols=['R_real', 'TEMP_real', 'R_measured', 'TEMP_measured'],
    #     # type_cols=['Type1', 'Type2', 'Type3', 'Type4'],
    #     type_cols=[],
    # )



