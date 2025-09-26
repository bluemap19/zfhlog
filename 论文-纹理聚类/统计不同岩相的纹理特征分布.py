import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src_data_process.data_correction_analysis import feature_influence_analysis
from src_data_process.data_dilute import dilute_dataframe
from src_data_process.data_distribution_statistics_overview import data_overview
from src_data_process.data_unsupervised import ClusteringPipeline, evaluate_clustering, \
    evaluate_clustering_performance_with_label
from src_file_op.dir_operation import search_files_by_criteria
from src_plot.Plot_boxplots import plot_boxes
from src_plot.plot_3D_scatter import interactive_3d_pca
from src_plot.plot_correlation import plot_correlation_analyze
from src_plot.plot_logging import visualize_well_logs
from src_plot.plot_matrxi_scatter import plot_matrxi_scatter
from src_table.table_process import table_2_to_3
from src_well_data.DATA_WELL import DATA_WELL
from src_well_project.LOGGING_PROJECT import LOGGING_PROJECT


if __name__ == '__main__':
    # WELL_TEST = DATA_WELL(path_folder=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\SIMU4')
    WELL_CLASS = DATA_WELL(path_folder=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\SIMU4')
    print(WELL_CLASS.get_logging_path_list())
    print(WELL_CLASS.get_FMI_path_list())

    path_fmi_texture = WELL_CLASS.search_data_path_by_charters(target_path_feature=['simu', 'texture'], target_file_type='logging')
    path_table = WELL_CLASS.search_data_path_by_charters(target_path_feature=['LITHO', 'TYPE'], target_file_type='table')
    print(path_fmi_texture)
    print(path_table)

    # Texture_DF = WELL_CLASS.get_logging_data(well_key=path_fmi_texture, Norm=False)
    # print(Texture_DF.describe())
    # COLS_TEXTURE = list(Texture_DF.columns)
    # print(COLS_TEXTURE)
    COLS_Target = [
     'CON_MEAN_DYNA', 'DIS_MEAN_DYNA', 'HOM_MEAN_DYNA',
     'ENG_MEAN_DYNA', 'COR_MEAN_DYNA', 'ASM_MEAN_DYNA', 'ENT_MEAN_DYNA',
     'CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA', 'ENG_SUB_DYNA',
     'COR_SUB_DYNA', 'ASM_SUB_DYNA', 'ENT_SUB_DYNA', 'CON_X_DYNA',
     'DIS_X_DYNA', 'HOM_X_DYNA', 'ENG_X_DYNA', 'COR_X_DYNA', 'ASM_X_DYNA',
     'ENT_X_DYNA', 'CON_Y_DYNA', 'DIS_Y_DYNA', 'HOM_Y_DYNA', 'ENG_Y_DYNA',
     'COR_Y_DYNA', 'ASM_Y_DYNA', 'ENT_Y_DYNA', 'CON_MEAN_STAT',
     'DIS_MEAN_STAT', 'HOM_MEAN_STAT', 'ENG_MEAN_STAT', 'COR_MEAN_STAT',
     'ASM_MEAN_STAT', 'ENT_MEAN_STAT', 'CON_SUB_STAT', 'DIS_SUB_STAT',
     'HOM_SUB_STAT', 'ENG_SUB_STAT', 'COR_SUB_STAT', 'ASM_SUB_STAT',
     'ENT_SUB_STAT', 'CON_X_STAT', 'DIS_X_STAT', 'HOM_X_STAT', 'ENG_X_STAT',
     'COR_X_STAT', 'ASM_X_STAT', 'ENT_X_STAT', 'CON_Y_STAT', 'DIS_Y_STAT',
     'HOM_Y_STAT', 'ENG_Y_STAT', 'COR_Y_STAT', 'ASM_Y_STAT', 'ENT_Y_STAT']
    TARGET_NAME = ['Type_Litho']

    table_3 = WELL_CLASS.get_type_3()
    replace_dict = WELL_CLASS.get_table_replace_dict()
    replace_dict_inversed = {value:key for key, value in replace_dict.items()}
    print(replace_dict, replace_dict_inversed)

    Texture_DF = WELL_CLASS.combine_logging_table(well_key=path_fmi_texture,
                                    table_key=path_table, replace_dict=replace_dict, new_col='Type_Litho', Norm=False)
    print(Texture_DF.describe())

    ############### 表格化的数据统计，并进行保存
    replace_dict_inversed = {0: 'E', 1: 'D', 2: 'C', 3: 'B', 4: 'A'}
    result = data_overview(
        df=Texture_DF[COLS_Target+TARGET_NAME],
        input_names=COLS_Target,
        target_col=TARGET_NAME[0],
        target_col_dict=replace_dict_inversed
    )
    print(result)
    path_texture_static_saved = path_fmi_texture.replace('.csv', '_static.xlsx')
    result.to_excel(path_texture_static_saved, index=True)

    # pearson_result, pearson_sorted, rf_result, rf_sorted = feature_influence_analysis(
    #     df_input=Texture_DF,
    #     input_cols=COLS_Target,
    #     target_col=TARGET_NAME[0],
    #     replace_dict=replace_dict_inversed
    # )
    # print("\n皮尔逊相关系数结果:", pearson_result)
    # print("\n按皮尔逊系数排序的属性:", pearson_sorted)
    # print("\n随机森林特征重要性结果:", rf_result)
    # print("\n按随机森林特征重要性排序的属性:", rf_sorted[:8])
    # COL_NAMES_CHOICED = rf_sorted[:6]

    # plot_correlation_analyze(Texture_DF, COL_NAMES_CHOICED, method='pearson', figsize=(14, 14),
    #                          return_matrix=False)

    # plot_boxes(df=Texture_DF[COL_NAMES_CHOICED + TARGET_NAME], input_names=COL_NAMES_CHOICED, target_col=TARGET_NAME[0],
    #            target_col_dict=replace_dict_inversed, figsize=(24, 5))

    # # replace_dict_inversed = {0: '块状构造泥岩', 1: '富有机质富凝灰质页岩', 2: '富有机质粉砂级长英质页岩', 3: '沉凝灰岩', 4: '薄夹层砂岩'}
    # replace_dict_inversed = {0: 'E', 1: 'D', 2: 'C', 3: 'B', 4: 'A'}
    # data_dilute = dilute_dataframe(Texture_DF[COL_NAMES_CHOICED+TARGET_NAME], ratio=30, group_by=TARGET_NAME[0])
    # print(f"按组抽稀30%后形状: {data_dilute.shape}")
    # # 传入的dataframe，数据不能是整体，必须经过精心挑选，才能
    # plot_matrxi_scatter(df=data_dilute, input_names=COL_NAMES_CHOICED, target_col=TARGET_NAME[0],
    #                     target_col_dict=replace_dict_inversed, figsize=(12, 11))

    # # 表格化的数据统计，并进行保存
    # result = data_overview(
    #     df=Texture_DF[COL_NAMES_CHOICED+TARGET_NAME],
    #     input_names=COL_NAMES_CHOICED,
    #     target_col=TARGET_NAME[0],
    #     target_col_dict=replace_dict_inversed
    # )
    # print(result)
    # path_texture_static_saved = path_fmi_texture.replace('.csv', '_static.xlsx')
    # result.to_excel(path_texture_static_saved, index=True)

    # # 调用可视化接口
    # visualize_well_logs(
    #     data=Texture_DF,
    #     depth_col='DEPTH',
    #     curve_cols=COL_NAMES_CHOICED,
    #     type_cols=TARGET_NAME,
    #     figsize=(22, 10)
    # )




