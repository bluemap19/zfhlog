import csv
from datetime import datetime
import numpy as np
import pandas as pd
from src_data_process.OLS1 import nonlinear_fitting
from src_data_process.data_filter import remove_static_depth_data
from src_data_process.resistivity_correction import scale_gaussian_by_config, scale_by_quantiles_old, scale_gaussian
from src_file_op.dir_operation import search_files_by_criteria
from scipy import stats

# 温度影响偏移 指数函数
def temp_influence_power_formula(df, A, B):
    return (A*np.power(df['TEMP'], B) - df['R_temp'])/df['R_temp']


# 温度影响偏移 线性函数
def temp_influence_linear_formula(df, A, B):
    return (A*df['TEMP'] + B - df['R_temp'])/df['R_temp']
    # return (A*df['TEMP'] + B - df['R_temp'])/(df['R_temp']*df['TEMP'])
    # return (A*np.power(df['TEMP'], B) - df['R_temp'])

# 偏移量剔除
def offset_linear(df):
    fit_result = nonlinear_fitting(df, temp_influence_linear_formula, initial_guess=(200, -5000),
                                   bounds=([0.1, -np.inf], [350, np.inf]))
    A, B = fit_result.x
    df['R_temp_sub'] = df['R_temp'] - A * df['TEMP'] - B
    print(f"Linear formular success as: A = {A:.4f}, B = {B:.4f}, 残差平方和: {fit_result.cost:.4f}")
    return df

# 偏移量剔除
def offset_power(df):
    # A:[0.2, 5], B:[1.5, 3]
    fit_result = nonlinear_fitting(df, temp_influence_power_formula, initial_guess=(0.5, 2.5), bounds=([0.2, 0.003], [200, 30]))
    # fit_result = nonlinear_fitting(df, temp_influence_power_formula, initial_guess=(0.5, 2.5), bounds=([0.2, 1.5], [5, 3]))
    A, B = fit_result.x
    df['R_temp_sub'] = df['R_temp'] / (A * np.power(df['TEMP'], B))
    print(f"Power formular success as: A = {A:.4f}, B = {B:.4f}, 残差平方和: {fit_result.cost:.4f}")
    return df


def calculate_r2(df, col_names):
    """
    计算DataFrame中两列数据的相关性系数R²

    参数:
    df (pd.DataFrame): 包含目标列的DataFrame
    col_names (list): 包含两个列名的列表，例如['col1', 'col2']

    返回:
    r2 (float): 两列数据的R²值
    """
    # 验证输入参数
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df必须是pandas DataFrame")

    if not isinstance(col_names, list) or len(col_names) != 2:
        raise ValueError("col_names必须是包含两个列名的列表")

    # 检查列是否存在
    for col in col_names:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' 不在DataFrame中")

    # 提取目标列数据
    x = df[col_names[0]].astype(float)
    y = df[col_names[1]].astype(float)

    # 处理缺失值（删除有NaN的行）
    mask = ~(x.isna() | y.isna())
    x_clean = x[mask]
    y_clean = y[mask]

    # 检查是否有足够的数据点
    n = len(x_clean)
    if n < 3:
        raise ValueError(f"有效数据点不足({n})，无法计算相关性")

    # 计算皮尔逊相关系数
    r, p_value = stats.pearsonr(x_clean, y_clean)

    # 计算R²值
    r2 = r ** 2

    return r2


def fit_r_pred(df, PRED_GAUSS_SETTING={}, offset_function='linear'):
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
            print('R_real middle:{:.4f},R_sub middle{:.4f}; R_real std {:.4f}, R_sub std{:.4f}'.format(np.median(df['R_real']), np.median(df['R_temp_sub']), np.std(df['R_real']), np.std(df['R_temp_sub'])))
        except Exception as e:
            print(f"Method power failed: {str(e)}, now try linear formula")
            df = offset_linear(df)

    # print(f"状态: {fit_result.message}")
    # plot_dataframe(df, 'R_temp_sub', 'R_real', title=None, X_ticks=None, Y_ticks=None, figure_type='scatter')

    if PRED_GAUSS_SETTING:
        df['R_gauss'], stats = scale_gaussian_by_config(source_data=df['R_temp_sub'], target_data_config=PRED_GAUSS_SETTING, return_stats=True)
    else:
        df['R_gauss']  = scale_gaussian(source_data=df['R_temp_sub'], target_data=df['R_real'])
        # df['R_gauss'] = scale_by_quantiles_old(source_data=df['R_temp_sub'], target_data=df['R_real'], quantile=0.05)

    return df




if __name__ == '__main__':
    # path_logging = search_files_by_criteria(search_root=r'C:\Users\Administrator\Desktop\25.06.29\UPDATE-13',
    path_logging = search_files_by_criteria(search_root=r'C:\Users\Administrator\Desktop\坨128-侧48-实测',
                                            name_keywords=['data_all_logging'], file_extensions=['csv'])
    path_logging = path_logging[0]
    df = pd.read_csv(path_logging, encoding='gbk')
    print(df.describe())
    print(list(df.columns))

    # 1. 定义映射关系（原始列名:新列名）
    column_mapping = {
        '#DEPTH':'DEPTH',
        'TEMP':'TEMP',
        'ResFar':'R_temp',
        'RT':'R_real',
    }
    column_mapping_inverted = {value: key for key, value in column_mapping.items()}
    df.rename(columns=column_mapping, inplace=True)
    print(df[['DEPTH', 'TEMP', 'R_temp']].describe())
    df['R_temp_sub'] = 0
    df['R_gauss'] = 0

    print(f'df remove depth error before:{df.shape}')
    df = remove_static_depth_data(df)
    print(f'df remove depth error after:{df.shape}')

    window_work_length = 400
    windows_step = 100
    windows_num = (df.shape[0] - window_work_length)//windows_step + 2
    windows_num = max(windows_num, 1)        # 防止窗长为<0
    windows_view = 1.1
    for i in range(windows_num):
        print(i)
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

        df_window = fit_r_pred(df_window, offset_function='power')
        # df_window = fit_r_pred(df_window, PRED_GAUSS_SETTING={'μ_target': 2.5, 'σ_target': 0.4})

        # df.iloc[window_work_start:window_work_end] = df_window
        EFFECTIVE_LENGTH = window_work_end - window_work_start
        df.iloc[window_work_start:window_work_end] = df_window.iloc[window_work_start-window_view_start:window_work_start-window_view_start+EFFECTIVE_LENGTH]

    df['DEPTH'] = df['DEPTH']-0.5
    df.rename(columns=column_mapping_inverted, inplace=True)
    time_str = datetime.now().strftime("%m-%d-%H-%M")
    df.to_csv(path_logging.replace('data_all_logging', f'data_pred_{time_str}_{window_work_length}_{windows_step}'),
              index=False, float_format='%.4f', quoting=csv.QUOTE_NONE)

    # # 调用可视化接口
    # visualize_well_logs(
    #     data=logging_data,
    #     depth_col='DEPTH',
    #     curve_cols=['R_real', 'TEMP_real', 'R_measured', 'TEMP_measured'],
    #     # type_cols=['Type1', 'Type2', 'Type3', 'Type4']
    #     type_cols=[]
    # )
