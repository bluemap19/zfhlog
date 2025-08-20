import pandas as pd
from Res_corrected_zky.corrected_res_by_config import fit_r_pred_window_gauss_scaling
from src_file_op.dir_operation import search_files_by_criteria
from src_data_process.data_filter import remove_static_depth_data
from src_plot.plot_logging import visualize_well_logs



if __name__ == '__main__':
    # path_logging = search_files_by_criteria(search_root=r'C:\Users\ZFH\Desktop\电阻率校正-坨73-斜13井',
    path_logging = search_files_by_criteria(search_root=r'F:\电阻率校正\SOF003',
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
        'ResFar-矫正':'R_measured',
        'ILD':'R_real',
    }
    df.rename(columns=column_mapping, inplace=True)
    print(df.columns)
    print(df[['DEPTH', 'TEMP', 'R_measured', 'R_real']].describe())
    df['correction_factor'] = 0.0
    df['R_temp_sub'] = 0.0
    df['R_gauss'] = 0.0

    df = remove_static_depth_data(df, depth_col='DEPTH')

    window_work_length = 200
    df = fit_r_pred_window_gauss_scaling(df, PRED_GAUSS_SETTING={}, offset_function='tempture', window_work_length=window_work_length, windows_step=10)
    # df = fit_r_pred_window_gauss_scaling(df, PRED_GAUSS_SETTING={}, offset_function='power', window_work_length=window_work_length, windows_step=10)
    # df = fit_r_pred_window_gauss_scaling(df, PRED_GAUSS_SETTING={}, offset_function='linear', window_work_length=window_work_length, windows_step=10)

    print(df.describe())

    # # df = log_predv_nearto_realv(df, window_length=11, SHIFT_RATIO=0.3, pred_col='R_gauss', real_col='R_real')
    # column_mapping_inverted = {value: key for key, value in column_mapping.items()}
    # df_save = df.copy()
    # df_save.rename(columns=column_mapping_inverted, inplace=True)
    # time_str = datetime.now().strftime("%m-%d-%H-%M")
    # df_save.to_csv(path_logging.replace(file_name, f'data_corrected_{window_work_length}_{time_str}.csv'), index=False, float_format='%.4f', quoting=csv.QUOTE_NONE)


    # 1. 定义映射关系（原始列名:新列名）
    column_mapping = {
        'TEMP':'温度',
        'R_measured':'测量电阻率',
        'R_real':'临井电阻率',
        'correction_factor':'温度校正因子',
        'R_temp_sub':'去势电阻率',
        'R_gauss':'随钻校正电阻率',
    }
    df_plot = df.copy()
    df_plot.rename(columns=column_mapping, inplace=True)
    print(df_plot.columns)

    # 调用可视化接口
    visualize_well_logs(
        data=df_plot.iloc[30:, :],
        depth_col='DEPTH',
        curve_cols=['温度', '测量电阻率', '温度校正因子', '去势电阻率', '随钻校正电阻率', '临井电阻率'],
        # curve_cols=['温度', '测量电阻率', '温度校正因子', '去势电阻率'],
        type_cols=[],
        figsize=(8, 10)
    )



