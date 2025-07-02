import csv
from datetime import datetime
import numpy as np
import pandas as pd
from src_data_process.OLS1 import nonlinear_fitting
from src_data_process.data_filter import remove_static_depth_data
from src_data_process.data_gauss_correction import scale_gaussian, scale_gaussian_by_config
from src_file_op.dir_operation import search_files_by_criteria


# 温度影响偏移指数
def temp_influence_power_formula(df, A, B):
    return (A*np.power(df['TEMP'], B) - df['R_temp'])/df['R_temp']
    # return (A*np.power(df['TEMP'], B) - df['R_temp'])/(df['R_temp']*df['TEMP'])
    # return (A*np.power(df['TEMP'], B) - df['R_temp'])
def temp_influence_log_power_formula(df, A, B):
    return (A*np.power(df['TEMP'], B) - np.log(df['R_temp']))/np.log(df['R_temp'])


# 温度影响偏移指数
def temp_influence_linear_formula(df, A, B):
    return (A*df['TEMP'] + B - df['R_temp'])/df['R_temp']
def temp_influence_log_linear_formula(df, A, B):
    return (A*df['TEMP'] + B - np.log(df['R_temp']))/np.log(df['R_temp'])

# 对目标数据进行基线的偏移，使用的是 线性函数
def offset_linear(df, LOG_USE=False):
    """ 这两个函数的区别，就是一个是线性函数，一个是幂函数
    该函数主要用来处理 1.温度-电阻率线性函数拟合 2.实测电阻率 减去 温度预测电阻率 3.计算得到电阻率的波动趋势 ，这个当作 原始的电阻率分布
    :param df:目标数据体dataframe，必须要有的数据列为：（减去温度-电阻率原本趋势后的差异电阻率残差数据 R_temp_sub）、
                                温度数据 TEMP、 该温度下对应的电阻率数据 R_temp
    :param LOG_USE: 是否使用log(R_temp, e), log(R_temp_pred, e)事先对电阻率数据进行预处理，这个可以有效预防跨越多个数量级的电阻率数据
    :return: df：依旧是目标数据体的dataframe
    """
    if LOG_USE:
        # 进行数据的拟合，这里使用的公式为，temp_influence_log_linear_formula，温度-log(电阻率) 线性拟合公式
        fit_result = nonlinear_fitting(df, temp_influence_log_linear_formula, initial_guess=(200, -5000),
                                   bounds=([-np.inf, -np.inf], [np.inf, np.inf]))
        A, B = fit_result.x
        # 通过拟合的结果，计算 电阻率的 残差，即 电阻率的 差异趋势
        df['R_temp_sub'] = np.log(df['R_temp']) - A * df['TEMP'] - B
    else:
        # 进行数据的拟合，这里使用的公式为，temp_influence_linear_formula，温度-电阻率 线性拟合处理后的公式
        fit_result = nonlinear_fitting(df, temp_influence_linear_formula, initial_guess=(200, -5000),
                                   bounds=([10, -np.inf], [450, np.inf]))
        A, B = fit_result.x
        # 通过拟合的结果，计算 log(电阻率)的 残差，即 log(电阻率)的 差异趋势
        df['R_temp_sub'] = df['R_temp'] - A * df['TEMP'] - B
    print(f"Linear formular success as: A = {A:.4f}, B = {B:.4f}, 残差平方和: {fit_result.cost:.4f}")
    return df


# 对目标数据进行基线偏移，使用的是 指数函数
def offset_power(df, LOG_USE):
    """ 这两个函数的区别，就是一个是线性函数，一个是幂函数
    该函数主要用来处理 1.温度-电阻率幂函数函数拟合 2.实测电阻率 减去 温度预测电阻率 3.计算得到电阻率的波动趋势 ，这个当作 原始的电阻率分布
    :param df:目标数据体dataframe，必须要有的数据列为：（减去温度预测电阻率原本趋势后的差异电阻率残差数据 R_temp_sub）、
                                温度数据 TEMP、 该温度下对应的电阻率数据 R_temp
    :param LOG_USE: 是否使用log(R_temp, e), log(R_temp_pred, e)事先对电阻率数据进行预处理，这个可以有效预防跨越多个数量级的电阻率数据
    :return: df：依旧是目标数据体的dataframe
    """
    if LOG_USE:
        # A:[0.2, 5], B:[1.5, 3]
        fit_result = nonlinear_fitting(df, temp_influence_log_power_formula, initial_guess=(0.5, 2.5), bounds=([-np.inf, -np.inf], [np.inf, np.inf]))
        A, B = fit_result.x
        df['R_temp_sub'] = np.log(df['R_temp']) / (A * np.power(df['TEMP'], B))
    else:
        # A:[0.02, 5], B:[1.5, 3]
        fit_result = nonlinear_fitting(df, temp_influence_power_formula, initial_guess=(0.5, 2.5), bounds=([0.02, 1.5], [5, 4]))
        A, B = fit_result.x
        df['R_temp_sub'] = df['R_temp'] / (A * np.power(df['TEMP'], B))
    print(f"Power formular success as: A = {A:.4f}, B = {B:.4f}, 残差平方和: {fit_result.cost:.4f}")
    return df


# # 先使用数据拟合减去基线，再使用数据高斯分布进行 测量数据偏移值的缩放
# def fit_r_pred_gauss_scaling(df, PRED_GAUSS_SETTING={}, offset_function='linear', LOG_USE=False):
#     # 确保输入包含所需列
#     if PRED_GAUSS_SETTING:
#         required_cols = ['R_temp', 'TEMP', 'R_gauss']
#     else:
#         required_cols = ['R_temp', 'TEMP', 'R_real', 'R_gauss']
#     assert all(col in df.columns for col in required_cols), "DataFrame缺少必要列"
#
#     # 执行拟合,先尝试 幂函数R_temp' = A * TEMP^B，再尝试 线性函数 R_temp' = A * TEMP + B
#     if offset_function.lower() == 'linear':
#         try:
#             df = offset_linear(df, LOG_USE)
#         except Exception as e:
#             print(f"Method power failed: {str(e)}, now try linear formula")
#             df = offset_power(df, LOG_USE)
#     elif offset_function.lower() == 'power':
#         try:
#             df = offset_power(df, LOG_USE)
#         except Exception as e:
#             print(f"Method power failed: {str(e)}, now try linear formula")
#             df = offset_linear(df, LOG_USE)
#
#
#     # print(f"状态: {fit_result.message}")
#     # plot_dataframe(df, 'R_temp_sub', 'R_real', title=None, X_ticks=None, Y_ticks=None, figure_type='scatter')
#
#     if PRED_GAUSS_SETTING:
#         if LOG_USE:
#             df['R_gauss_log'], stats = scale_gaussian_by_config(source_data=df['R_temp_sub'], target_data_config=PRED_GAUSS_SETTING, return_stats=True)
#             df['R_gauss'] = np.power(np.e, df['R_gauss_log'])
#         else:
#             df['R_gauss'], stats = scale_gaussian_by_config(source_data=df['R_temp_sub'], target_data_config=PRED_GAUSS_SETTING, return_stats=True)
#     else:
#         if LOG_USE:
#             df['R_gauss_log'], stats = scale_gaussian(source_data=df['R_temp_sub'], target_data=df['R_real'], return_stats=True)
#             df['R_gauss'] = np.power(np.e, df['R_gauss_log'])
#         else:
#             df['R_gauss'], stats  = scale_gaussian(source_data=df['R_temp_sub'], target_data=df['R_real'], return_stats=True)
#     return df



def window_process_gauss_scaling(df, PRED_GAUSS_SETTING={}, LOG_USE=False,
                                 window_work_length=500, windows_step=100, windows_view = 1.1):
    windows_num = (df.shape[0] - window_work_length)//windows_step + 2

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

        df_window = df.iloc[window_view_start:window_view_end].copy()

        if PRED_GAUSS_SETTING:
            if LOG_USE:
                df_window['R_gauss_log'], stats = scale_gaussian_by_config(source_data=df['R_temp_sub'],
                                                                    target_data_config=PRED_GAUSS_SETTING,
                                                                    return_stats=True)
                df_window['R_gauss'] = np.power(np.e, df['R_gauss_log'])
            else:
                df_window['R_gauss'], stats = scale_gaussian_by_config(source_data=df['R_temp_sub'],
                                                                target_data_config=PRED_GAUSS_SETTING,
                                                                return_stats=True)
        else:
            if LOG_USE:
                df_window['R_gauss_log'], stats = scale_gaussian(source_data=df['R_temp_sub'], target_data=np.log(df['R_real']))
                df_window['R_gauss'] = np.power(np.e, df['R_gauss_log'])
            else:
                df_window['R_gauss'] = scale_gaussian(source_data=df['R_temp_sub'], target_data=df['R_real'])

        print(df.shape)
        EFFECTIVE_LENGTH = window_work_end - window_work_start
        df.iloc[window_work_start:window_work_end] = df_window.iloc[
                                                     window_work_start - window_view_start:window_work_start - window_view_start + EFFECTIVE_LENGTH]

        # # df_window = df.iloc[window_work_start:window_work_end].copy()
        # df_window = df.iloc[window_view_start:window_view_end].copy()


        # df_window = fit_r_pred_gauss_scaling(df_window, offset_function='power', LOG_USE=False)
        # # df_window = fit_r_pred_gauss_scaling(df_window, PRED_GAUSS_SETTING={'μ_target': 2.5, 'σ_target': 0.4})
        #
        # # df.iloc[window_work_start:window_work_end] = df_window
        # EFFECTIVE_LENGTH = window_work_end - window_work_start
        # df.iloc[window_work_start:window_work_end] = df_window.iloc[window_work_start-window_view_start:window_work_start-window_view_start+EFFECTIVE_LENGTH]

    return df



# 只使用数据拟合减去基线，得到 测量数据偏移值
def fit_r_pred_window_gauss_scaling(df, PRED_GAUSS_SETTING={}, offset_function='power', LOG_USE=False):
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

    # print(df.describe())
    # exit(0)

    df = window_process_gauss_scaling(df, PRED_GAUSS_SETTING, LOG_USE=LOG_USE,
                                      window_work_length=5000, windows_step=100, windows_view=1.2)

    # print(df.describe())
    return df


# def log_predv_nearto_realv(df, window_length=5, SHIFT_RATIO=0.5, pred_col='R_gauss', real_col='R_real'):
#     df_copy = df.copy()
#
#     for i in range(df.shape[0]):
#         index_start = max(i - window_length, 0)
#         index_end = min(i + window_length + 1, df.shape[0]-1)
#         # print(index_start, index_end)
#         df_copy.loc[i, pred_col] = df.loc[i, pred_col] + (np.mean(df.loc[index_start:index_end, real_col])
#                                                           - df.loc[i, pred_col]) * SHIFT_RATIO
#
#     return df_copy



if __name__ == '__main__':
    path_logging = search_files_by_criteria(search_root=r'C:\Users\Administrator\Desktop\25.06.29\UPDATE-3',
                                            name_keywords=['data_all_logging'], file_extensions=['csv'])
    path_logging = path_logging[0]
    df = pd.read_csv(path_logging, encoding='gbk')
    # print(df.describe())
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
    df['R_gauss_log'] = 0
    df['R_gauss'] = 0

    df = remove_static_depth_data(df, depth_col='DEPTH')

    df = fit_r_pred_window_gauss_scaling(df, PRED_GAUSS_SETTING={}, offset_function='power', LOG_USE=False)

    # df = log_predv_nearto_realv(df, window_length=11, SHIFT_RATIO=0.3, pred_col='R_gauss', real_col='R_real')

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



