import csv
from datetime import datetime
import numpy as np
import pandas as pd
from src_data_process.OLS1 import nonlinear_fitting
from src_data_process.data_filter import remove_static_depth_data
from src_data_process.resistivity_correction import scale_gaussian, scale_gaussian_by_config
from src_file_op.dir_operation import search_files_by_criteria
from Remove_temp_influence import correction_by_tempture, offset_linear, offset_power


def fit_r_pred(df, PRED_GAUSS_SETTING={}, offset_function='linear', LOG_USE=False):
    # 确保输入包含所需列
    if PRED_GAUSS_SETTING:
        required_cols = ['R_temp', 'TEMP', 'R_gauss']
    else:
        required_cols = ['R_temp', 'TEMP', 'R_real', 'R_gauss']
    assert all(col in df.columns for col in required_cols), "DataFrame缺少必要列"

    # 执行拟合,先尝试 幂函数R_temp' = A * TEMP^B，再尝试 线性函数 R_temp' = A * TEMP + B
    if offset_function.lower() == 'linear':
        try:
            df = offset_linear(df, LOG_USE)
        except Exception as e:
            print(f"Method power failed: {str(e)}, now try linear formula")
            df = offset_power(df, LOG_USE)
    elif offset_function.lower() == 'power':
        try:
            df = offset_power(df, LOG_USE)
        except Exception as e:
            print(f"Method power failed: {str(e)}, now try linear formula")
            df = offset_linear(df, LOG_USE)
    elif offset_function.lower() == 'tempture':
        try:
            df = correction_by_tempture(df)
        except Exception as e:
            print(f"Method power failed: {str(e)}, now try linear formula")

    if PRED_GAUSS_SETTING:
        if LOG_USE:
            pass
            df['R_gauss_log'], stats = scale_gaussian_by_config(source_data=df['R_temp_sub'], target_data_config=PRED_GAUSS_SETTING, return_stats=True)
            df['R_gauss'] = np.power(np.e, df['R_gauss_log'])
        else:
            df['R_gauss'], stats = scale_gaussian_by_config(source_data=df['R_temp_sub'], target_data_config=PRED_GAUSS_SETTING, return_stats=True)
    else:
        if LOG_USE:
            df['R_gauss_log'], stats = scale_gaussian(source_data=df['R_temp_sub'], target_data=df['R_real'], return_stats=True)
            df['R_gauss'] = np.power(np.e, df['R_gauss_log'])
        else:
            df['R_gauss'], stats  = scale_gaussian(source_data=df['R_temp_sub'], target_data=df['R_real'], return_stats=True)

    return df


# 根据临井测井资料进行直接的偏移，这个最邪修，放弃
def log_predv_nearto_realv(df, window_length, SHIFT_RATIO=0.5, pred_col='R_gauss', real_col='R_real'):
    df_copy = df.copy()

    for i in range(df.shape[0]):
        index_start = max(i - window_length, 0)
        index_end = min(i + window_length + 1, df.shape[0]-1)
        # print(index_start, index_end)
        df_copy.loc[i, pred_col] = df.loc[i, pred_col] + (np.mean(df.loc[index_start:index_end, real_col])
                                                          - df.loc[i, pred_col]) * SHIFT_RATIO

    return df_copy



if __name__ == '__main__':
    path_logging = search_files_by_criteria(search_root=r'C:\Users\Administrator\Desktop\25.06.29\UPDATE-3',
                                            name_keywords=['data_all_logging'], file_extensions=['csv'])
    print(path_logging)
    path_logging = path_logging[0]
    df = pd.read_csv(path_logging, encoding='gbk')
    print(df.describe())
    # exit(0)

    # 1. 定义映射关系（原始列名:新列名）
    column_mapping = {
        '#DEPTH':'DEPTH',
        'TEMP':'TEMP',
        'GVK':'R_temp',
        'MSFL':'R_real',
    }
    df.rename(columns=column_mapping, inplace=True)
    print(df.columns)
    print(df[['DEPTH', 'TEMP', 'R_temp']].describe())
    df['R_temp_sub'] = 0
    df['R_gauss'] = 0

    df = remove_static_depth_data(df)

    window_work_length = 500
    # window_work_length = df.shape[0]
    windows_step = 100
    windows_num = (df.shape[0] - window_work_length)//windows_step + 2
    windows_view = 1.1
    for i in range(windows_num):
        window_index = i*windows_step + window_work_length//2
        window_work_start = np.max((0, window_index - window_work_length // 2))
        window_work_end = np.min((df.shape[0] - 1, window_index + window_work_length // 2 + 1))

        window_view_length = int(windows_view * window_work_length)
        window_view_start = np.max((0, window_index - window_view_length // 2))
        window_view_end = np.min((df.shape[0] - 1, window_index + window_view_length // 2 + 1))


        print(f'windows index:{window_index}, '
              f'work window length:{window_work_length},'
              f'start:{window_work_start}, end:{window_work_end},'
              f'depth:{df.DEPTH[window_work_start]} to {df.DEPTH[window_work_end]}'
              # f'view window length:{window_view_length}, start:{window_view_start}, end:{window_view_end}'
              )

        # df_window = df.iloc[window_work_start:window_work_end].copy()
        df_window = df.iloc[window_view_start:window_view_end].copy()

        df_window = fit_r_pred(df_window, offset_function='power', LOG_USE=False)
        # df_window = fit_r_pred(df_window, PRED_GAUSS_SETTING={'μ_target': 2.5, 'σ_target': 0.4})

        # df.iloc[window_work_start:window_work_end] = df_window
        EFFECTIVE_LENGTH = window_work_end - window_work_start
        df.iloc[window_work_start:window_work_end] = df_window.iloc[window_work_start-window_view_start:window_work_start-window_view_start+EFFECTIVE_LENGTH]

    df = log_predv_nearto_realv(df, window_length=5, SHIFT_RATIO=0.3, pred_col='R_gauss', real_col='R_real')

    column_mapping_inverted = {value: key for key, value in column_mapping.items()}
    df.rename(columns=column_mapping_inverted, inplace=True)
    time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    df.to_csv(path_logging.replace('data_all_logging.csv', f'data_pred_{time_str}.csv'), index=False, float_format='%.4f', quoting=csv.QUOTE_NONE)

    # # 调用可视化接口
    # visualize_well_logs(
    #     data=logging_data,
    #     depth_col='DEPTH',
    #     curve_cols=['R_real', 'TEMP_real', 'R_measured', 'TEMP_measured'],
    #     # type_cols=['Type1', 'Type2', 'Type3', 'Type4']
    #     type_cols=[]
    # )



