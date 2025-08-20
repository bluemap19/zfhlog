from datetime import datetime
import pandas as pd
import numpy as np

from Res_corrected_zky.corrected_res_by_config import fit_r_pred_window_gauss_scaling, correct_res_by_tempture
from src_data_process.resistivity_correction import scale_by_quantiles_use_config
from src_file_op.dir_operation import search_files_by_criteria
from Remove_temp_influence import correction_by_tempture
from src_plot.plot_logging import visualize_well_logs



if __name__ == "__main__":
    path_temp_config = search_files_by_criteria(search_root=r'D:\GitHubProj\zfhlog\Res_corrected_zky',
                                            name_keywords=['config_tempture'], file_extensions=['xlsx', 'csv'])
    path_temp_config = path_temp_config[0]
    if path_temp_config.endswith('xlsx'):
        df_config = pd.read_excel(path_temp_config, sheet_name=0, engine='openpyxl')
    elif path_temp_config.endswith('csv'):
        df_config = pd.read_csv(path_temp_config, encoding='gbk')
    else:
        print('Not find excel file')
        exit(0)
    print(df_config.describe())

    path_file_target = search_files_by_criteria(search_root=r'F:\电阻率校正\坨128-侧48',
                                            name_keywords=['logging'], file_extensions=['xlsx', 'csv'])
    if len(path_file_target) != 1:
        print('path_file_target error as :{}'.format(path_file_target))
    if path_temp_config.endswith('xlsx'):
        df_source = pd.read_excel(path_file_target[0], sheet_name=0, engine='openpyxl')
    elif path_temp_config.endswith('csv'):
        df_source = pd.read_csv(path_file_target[0], encoding='gbk')
    else:
        print('Not find excel file')
        exit(0)
    print(df_source.columns)


    # 1. 定义映射关系（原始列名:新列名）
    column_mapping = {
        '#DEPTH':'DEPTH',
        'TEMP':'TEMP',
        'ResFar':'R_measured',
        'RT':'R_real'
    }
    df_source.rename(columns=column_mapping, inplace=True)
    print(df_source.columns)
    df_source['correction_factor'] = 0.0
    df_source['R_temp_sub'] = 0.0
    df_source['R_gauss'] = 0.0

    TEMP_MAX = df_config['TEMP'].max()
    TEMP_MIN = df_config['TEMP'].min()
    df_source = df_source[(df_source['TEMP'] >= TEMP_MIN) & (df_source['TEMP'] < TEMP_MAX)]
    df_source.reset_index(drop=True, inplace=True)
    print(df_source[['DEPTH', 'TEMP', 'R_measured', 'R_real']].describe())
    # 调用可视化接口
    visualize_well_logs(
        data=df_source,
        depth_col='DEPTH',
        curve_cols=['TEMP', 'R_measured', 'R_real'],
        # curve_cols=['温度', '测量电阻率', '温度校正因子', '去势电阻率'],
        type_cols=[],
        figsize=(6, 10)
    )

    df_source_corrected_by_near_well = df_source.copy()
    window_work_length = 200
    df_source_corrected_by_near_well = fit_r_pred_window_gauss_scaling(df_source_corrected_by_near_well, PRED_GAUSS_SETTING={},
                                                                       offset_function='linear', window_work_length=window_work_length, windows_step=10)
    # df_source_corrected_by_near_well = fit_r_pred_window_gauss_scaling(df_source_corrected_by_near_well, PRED_GAUSS_SETTING={},
    #                                                                    offset_function='linear', window_work_length=window_work_length, windows_step=10)

    # 调用可视化接口
    visualize_well_logs(
        data=df_source_corrected_by_near_well,
        depth_col='DEPTH',
        curve_cols=['TEMP', 'R_measured', 'R_temp_sub', 'R_gauss', 'R_real'],
        # curve_cols=['温度', '测量电阻率', '温度校正因子', '去势电阻率'],
        type_cols=[],
        figsize=(12, 10)
    )

    df_source_corrected_by_config_file = df_source.copy()
    df_source_corrected_by_config_file = correct_res_by_tempture(df_source_corrected_by_config_file, df_config)
    # 调用可视化接口
    visualize_well_logs(
        data=df_source_corrected_by_config_file,
        depth_col='DEPTH',
        curve_cols=['TEMP', 'R_measured', 'correction_factor', 'R_temp_sub', 'R_gauss', 'R_real'],
        # curve_cols=['温度', '测量电阻率', '温度校正因子', '去势电阻率'],
        type_cols=[],
        figsize=(12, 10)
    )

    # df_source = correct_res_by_tempture(df_source, df_config)
    # print(df_source.describe())


    # column_mapping_inverted = {value: key for key, value in column_mapping.items()}
    # df_source.rename(columns=column_mapping_inverted, inplace=True)
    # time_str = datetime.now().strftime("%m-%d-%H-%M")
    # df_source.to_csv(path_measured_logging.replace(file_name, f'data_pred_{time_str}.csv'), index=False, float_format='%.4f')