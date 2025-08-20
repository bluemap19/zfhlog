import numpy as np
import pandas as pd
from src_data_process.OLS1 import nonlinear_fitting


######################################################################################################################################
# 温度影响偏移线性函数
def temp_influence_linear_formula(df, A, B):
    return (A*df['TEMP'] + B - df['R_measured'])/df['R_measured']

def temp_influence_log_linear_formula(df, A, B):
    return (A*df['TEMP'] + B - np.log(df['R_measured']))/np.log(df['R_measured'])

# # 对目标数据进行基线的偏移，使用的是 线性函数
# def offset_linear(df):
#     """ 这两个函数的区别，就是一个是线性函数，一个是幂函数
#     该函数主要用来处理 1.温度-电阻率线性函数拟合 2.实测电阻率 减去 温度预测电阻率 3.计算得到电阻率的波动趋势 ，这个当作 原始的电阻率分布
#     :param df:目标数据体dataframe，必须要有的数据列为：（减去温度-电阻率原本趋势后的差异电阻率残差数据 R_temp_sub）、
#                                 温度数据 TEMP、 该温度下对应的电阻率数据 R_measured
#     :return: df：依旧是目标数据体的dataframe
#     """
#     # 进行数据的拟合，这里使用的公式为，temp_influence_linear_formula，温度-电阻率 线性拟合处理后的公式
#     fit_result = nonlinear_fitting(df, temp_influence_linear_formula, initial_guess=(200, -5000),
#                                bounds=([10, -np.inf], [450, np.inf]))
#     A, B = fit_result.x
#     # 通过拟合的结果，计算 log(电阻率)的 残差，即 log(电阻率)的 差异趋势
#     df['R_temp_sub'] = df['R_measured'] - A * df['TEMP'] - B
#     print(f"Linear formular success as: A = {A:.4f}, B = {B:.4f}, 残差平方和: {fit_result.cost:.4f}")
#     return df
def offset_linear(df, LOG_USE=False):
    if LOG_USE:
        fit_result = nonlinear_fitting(df, temp_influence_log_linear_formula, initial_guess=(200, -5000),
                                   bounds=([-np.inf, -np.inf], [np.inf, np.inf]))
        A, B = fit_result.x
        df['R_temp_sub'] = np.log(df['R_measured']) - A * df['TEMP'] - B
    else:
        fit_result = nonlinear_fitting(df, temp_influence_linear_formula, initial_guess=(200, -5000),
                                   bounds=([10, -np.inf], [450, np.inf]))
        A, B = fit_result.x
        df['R_temp_sub'] = df['R_measured'] - A * df['TEMP'] - B
    print(f"Linear formular success as: A = {A:.4f}, B = {B:.4f}, 残差平方和: {fit_result.cost:.4f}")
    return df
######################################################################################################################################


# ************************************************************************************************************************************

# 温度影响偏移指数函数
def temp_influence_power_formula(df, A, B):
    return (A*np.power(df['TEMP'], B) - df['R_measured'])/df['R_measured']

def temp_influence_log_power_formula(df, A, B):
    return (A*np.power(df['TEMP'], B) - np.log(df['R_measured']))/np.log(df['R_measured'])

# 对目标数据进行基线偏移，使用的是 指数函数
# def offset_power(df):
#     """ 这两个函数的区别，就是一个是线性函数，一个是幂函数
#     该函数主要用来处理 1.温度-电阻率幂函数函数拟合 2.实测电阻率 减去 温度预测电阻率 3.计算得到电阻率的波动趋势 ，这个当作 原始的电阻率分布
#     :param df:目标数据体dataframe，必须要有的数据列为：（减去温度预测电阻率原本趋势后的差异电阻率残差数据 R_temp_sub）、
#                                 温度数据 TEMP、 该温度下对应的电阻率数据 R_measured
#     :return: df：依旧是目标数据体的dataframe
#     """
#     # 进行数据的拟合，这里使用的公式为，temp_influence_power_formula，温度-电阻率 幂函数拟合处理后的公式
#     # A:[0.02, 5], B:[1.5, 3]
#     fit_result = nonlinear_fitting(df, temp_influence_power_formula, initial_guess=(0.5, 2.5), bounds=([0.02, 0.5], [500, 4]))
#     A, B = fit_result.x
#     df['R_temp_sub'] = df['R_measured'] / (A * np.power(df['TEMP'], B))
#     print(f"Power formular success as: A = {A:.4f}, B = {B:.4f}, 残差平方和: {fit_result.cost:.4f}")
#     return df
def offset_power(df, LOG_USE=False):
    if LOG_USE:
        fit_result = nonlinear_fitting(df, temp_influence_log_power_formula, initial_guess=(0.5, 2.5), bounds=([-np.inf, -np.inf], [np.inf, np.inf]))
        A, B = fit_result.x
        df['R_temp_sub'] = np.log(df['R_measured']) - A * np.power(df['TEMP'], B)
    else:
        fit_result = nonlinear_fitting(df, temp_influence_power_formula, initial_guess=(0.5, 2.5), bounds=([0.02, 0.5], [500, 4]))
        A, B = fit_result.x
        df['R_temp_sub'] = df['R_measured'] - A * np.power(df['TEMP'], B)
    print(f"Power formular success as: A = {A:.4f}, B = {B:.4f}, 残差平方和: {fit_result.cost:.4f}")
    return df
# ************************************************************************************************************************************


######################################################################################################################################
TEMPTURE_CORRECTION_MAPPING = pd.read_excel(r'F:\电阻率校正\config_temperature_far_resistivity_correction.xlsx', sheet_name=0)

## 对目标数据进行基线的偏移，使用的是 配置好的温度校正文件
def get_tempture_correction_factor(tempture):
    """
    这个函数不对，这个是暂时的，后面要改
    :param tempture: 温度
    :return: 对应温度下的电阻率值
    """
    # res = 24.81*tempture - 521.51
    tempture_sub = TEMPTURE_CORRECTION_MAPPING['Temperature'] - tempture

    # 找到最接近的温度索引
    abs_diff = np.abs(tempture_sub)
    min_index = abs_diff.idxmin()  # 使用Pandas的idxmin方法

    # 获取校正系数
    correction_factor = TEMPTURE_CORRECTION_MAPPING.at[min_index, 'Corrrection_factor']

    return correction_factor

## 对目标数据进行基线的偏移，使用的是 配置好的温度校正文件
def correction_by_tempture(df, res_base=338.4786):
    if isinstance(df, pd.DataFrame):
        for i in range(0, df.shape[0]):
            # tempture = df.at[i, 'TEMP']     # 45
            # res = df.at[i, 'R_measured']        # 真实测量
            # correction_temp = get_tempture_correction_factor(tempture)
            # df.at[i, 'correction_factor'] = correction_temp
            # df.at[i, 'R_temp_sub'] = res / correction_temp
            tempture = df.iloc[i]['TEMP']  # 使用 iloc 访问位置索引
            res = df.iloc[i]['R_measured']  # 使用 iloc 访问位置索引
            correction_temp = get_tempture_correction_factor(tempture)
            df.iloc[i]['correction_factor'] = correction_temp
            df.iloc[i]['R_temp_sub'] = res / correction_temp

        return df
    else:
        print('Not a dataframe')
        exit(0)

######################################################################################################################################


if __name__ == '__main__':
    print(get_tempture_correction_factor(52))
    print(get_tempture_correction_factor(49))
    print(get_tempture_correction_factor(31))
    print(get_tempture_correction_factor(27))