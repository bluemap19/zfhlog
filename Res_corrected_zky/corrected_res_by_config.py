import math
import numpy as np
import pandas as pd
from src_data_process.resistivity_correction import scale_gaussian_by_config, \
    scale_by_quantiles, scale_by_quantiles_use_config
from Remove_temp_influence import offset_linear, offset_power, correction_by_tempture

# 温度去势电阻率的量级缩放，通过临井电阻率数据或者是配置信息进行电阻率的量级校正

def res_window_scaling(df, PRED_GAUSS_SETTING={}, window_work_length=500,
                       windows_step=100, windows_view = 1.1):
    windows_num = math.floor((df.shape[0] - window_work_length)/windows_step) + 1

    print(f'work window length:{window_work_length}; view window length:{int(window_work_length*windows_view)}, windows view ratio:{windows_view}, windows number:{windows_num}' )
    config_all_save = []
    for i in range(windows_num):
        window_index = i*windows_step + window_work_length//2
        window_work_start = np.max((0, window_index - window_work_length // 2))
        window_work_end = np.min((df.shape[0] - 1, window_index + window_work_length // 2 + 1))

        window_view_length = int(windows_view * window_work_length)
        window_view_start = np.max((0, window_index - window_view_length // 2))
        window_view_end = np.min((df.shape[0] - 1, window_index + window_view_length // 2 + 1))

        df_window = df.iloc[window_view_start:window_view_end].copy()

        if PRED_GAUSS_SETTING:
            df_window['R_gauss'], stats = scale_gaussian_by_config(source_data=df['R_temp_sub'],
                                                                target_data_config=PRED_GAUSS_SETTING,
                                                                return_stats=True)
        else:
            # 使用高斯缩放，对数据范围进行处理
            # df['R_gauss'], config = scale_gaussian(source_data=df['R_temp_sub'], target_data=df['R_real'], return_stats=True)
            df_window['R_gauss'], config = scale_by_quantiles(source_data=df_window['R_temp_sub'], target_data=df_window['R_real'])

        # if i%20 == 0:
        #     if PRED_GAUSS_SETTING:
        #         print(f'windows index:{window_index}; work window length:{window_work_length}; index from:{window_work_start} to {window_work_end};'
        #       f'depth from:{df.DEPTH[window_work_start]} to {df.DEPTH[window_work_end]}, view window length:{window_view_length}, from:{window_view_start} to {window_view_end}'
        #       )
        #     else:
        #         print(f'windows index:{window_index}; work window length:{window_work_length}; index from:{window_work_start} to {window_work_end};'
        #             f'depth from:{df.DEPTH[window_work_start]} to {df.DEPTH[window_work_end]}, view window length:{window_view_length}, from:{window_view_start} to {window_view_end}'
        #               f', target_lower:{config[0]}, source_lower:{config[1]}, scale_factor:{config[2]}'
        #       )

        # config.append(df.at[window_index, 'TEMP'])
        if window_index < len(df):
            config.append(df.iloc[window_index]['TEMP'])
        config_all_save.append(config)
        dataframe_config = pd.DataFrame(config_all_save, columns=['target_lower', 'source_lower', 'scale_factor','TEMP'])
        dataframe_config.to_csv('config_tempture.csv', index=False)

        EFFECTIVE_LENGTH = window_work_end - window_work_start
        df.iloc[window_work_start:window_work_end] = df_window.iloc[
                                                     window_work_start - window_view_start:window_work_start - window_view_start + EFFECTIVE_LENGTH]


    return df



# 整体上使用数据拟合减去基线，得到 测量数据偏移值，温度去势，然后逐窗口进行量级缩放
def fit_r_pred_window_gauss_scaling(df, PRED_GAUSS_SETTING={}, offset_function='power', window_work_length=200, windows_step=10):
    # 确保输入包含所需列
    if PRED_GAUSS_SETTING:
        required_cols = ['R_measured', 'TEMP', 'R_gauss']
    else:
        required_cols = ['R_measured', 'TEMP', 'R_real', 'R_gauss']
    assert all(col in df.columns for col in required_cols), "DataFrame缺少必要列"

    # 执行拟合,先尝试 幂函数R_measured' = A * TEMP^B，再尝试 线性函数 R_measured' = A * TEMP + B
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

    df = res_window_scaling(df, PRED_GAUSS_SETTING,
                            window_work_length=window_work_length, windows_step=windows_step, windows_view=1.2)

    return df



# 根据温度，寻找最佳的 量级缩放 配置信息
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


# 根据温度去势，然后再进行电阻率的量级缩放
def correct_res_by_tempture(df_source, df_config):
    df_source = correction_by_tempture(df_source)
    print(df_source.describe())

    quantile = 0.2
    windows_length = 200
    for i in range(df_source.shape[0]):
        tempture_t = df_source.iloc[i]['TEMP']

        windows_s = max(0, i-windows_length)
        windows_e = min(i + windows_length, df_source.shape[0]-1)
        data_windows = df_source.iloc[windows_s:windows_e]['R_temp_sub'].values
        data_index = df_source.iloc[i]['R_temp_sub']
        source_lower = np.percentile(data_windows, quantile * 100)

        closest_row = find_closest_match(df_config, tempture_t, source_lower)
        print(closest_row)

        scaled_result = scale_by_quantiles_use_config(data_index, config={'target_lower':closest_row['target_lower'], 'source_lower':source_lower, 'scale_factor':closest_row['scale_factor']})

        df_source.iloc[i]['R_gauss'] = scaled_result
        print(closest_row)
        print(scaled_result)

    return df_source