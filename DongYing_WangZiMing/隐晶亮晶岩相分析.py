import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src_data_process.data_correction_analysis import feature_influence_analysis
from src_data_process.data_unsupervised import ClusteringPipeline, evaluate_clustering, \
    evaluate_clustering_performance_with_label
from src_file_op.dir_operation import search_files_by_criteria
from src_plot.Plot_boxplots import plot_boxes
from src_plot.plot_3D_scatter import interactive_3d_pca
from src_plot.plot_correlation import plot_correlation_analyze
from src_plot.plot_matrxi_scatter import plot_matrxi_scatter
from src_table.table_process import table_2_to_3
from src_well_project.LOGGING_PROJECT import LOGGING_PROJECT


if __name__ == '__main__':
    LG = LOGGING_PROJECT(project_path=r'F:\桌面\算法测试-长庆数据收集\logging_CSV')
    table = LG.get_table_3_all_data(['王子铭-东营-1-15'])
    print(table)

    path_logging_target = LG.search_target_file_path(well_name='王子铭-东营-1-15', target_path_feature=['logging'], target_file_type='logging')
    path_table_target = LG.search_target_file_path(well_name='王子铭-东营-1-15', target_path_feature=['LITHO_TYPE'], target_file_type='table')
    print('get data from path:{},\nget table from:{}'.format(path_logging_target, path_table_target))

    replace_dict = LG.get_all_table_replace_dict(well_names=['王子铭-东营-1-15'], Type_col_name='晶相')
    replace_dict_inversed = {value:key for key, value in replace_dict.items()}
    print(replace_dict, '--->', replace_dict_inversed)

    # COL_NAMES_ALL = [
    #     'STAT_CON', 'STAT_DIS', 'STAT_HOM', 'STAT_ENG', 'STAT_COR', 'STAT_ASM', 'STAT_ENT', 'STAT_XY_CON',
    #     'STAT_XY_DIS', 'STAT_XY_HOM', 'STAT_XY_ENG', 'STAT_XY_COR', 'STAT_XY_ASM', 'STAT_XY_ENT',
    #     'DYNA_CON', 'DYNA_DIS', 'DYNA_HOM', 'DYNA_ENG', 'DYNA_COR', 'DYNA_ASM', 'DYNA_ENT', 'DYNA_XY_CON',
    #     'DYNA_XY_DIS', 'DYNA_XY_HOM', 'DYNA_XY_ENG', 'DYNA_XY_COR', 'DYNA_XY_ASM', 'DYNA_XY_ENT'
    # ]
    COL_NAMES = [
        # AZIM BS CAL CON1 DAZZ DEN DEV GLD GR HOAZ HOFS POR PORF PORW QT R4 RD RS RT RXO SH SP SW SXO TVDD VAC VCAL VCNL VDEN VDEV VGR VQT VR4 VRD VRS VSP XE YN
        'AC', 'CNL', 'DEN', 'GR', 'RS', 'POR',
    ]



    TARGET_NAME = ['Type']
    data_combined_all = LG.combined_all_logging_with_type(well_names=['王子铭-东营-1-15'],
                                                   # file_path_logging={'王子铭-东营-1-15': path_logging_target},
                                                   # file_path_table={'王子铭-东营-1-15': path_table_target},
                                                   replace_dict=replace_dict, type_new_col=TARGET_NAME[0],
                                                   curve_names_logging=COL_NAMES,
                                                   Norm=False)
    print(data_combined_all.describe())
    print(data_combined_all.columns)            # '#DEPTH', 'AC', 'CNL', 'DEN', 'GR', 'RD', 'RS', 'RXO', 'POR', 'Type',
    data_combined_all[COL_NAMES] = data_combined_all[COL_NAMES].astype(float)
    data_combined_all[TARGET_NAME] = data_combined_all[TARGET_NAME].astype(int)

    # plot_correlation_analyze(data_combined_all, COL_NAMES, method='pearson', figsize=(14, 14),
    #                          return_matrix=False)
    #
    # plot_matrxi_scatter(df=data_combined_all, input_names=COL_NAMES, target_col=TARGET_NAME[0],
    #                     target_col_dict=replace_dict_inversed, figsize=(12, 11))
    #
    # plot_boxes(df=data_combined_all, input_names=COL_NAMES, target_col=TARGET_NAME[0],
    #            target_col_dict=replace_dict_inversed, figsize=(24, 5))

    pearson_result, pearson_sorted, rf_result, rf_sorted = feature_influence_analysis(
        df=data_combined_all,
        input_cols=COL_NAMES,
        target_col=TARGET_NAME[0],
        replace_dict=replace_dict_inversed
    )
    print("\n皮尔逊相关系数结果:", pearson_result)
    print("\n按皮尔逊系数排序的属性:", pearson_sorted)
    print("\n随机森林特征重要性结果:", rf_result)
    print("\n按随机森林特征重要性排序的属性:", rf_sorted)

    # # plot_correlation_analyze 进行Peason相关性系数计算，必须全部使用float格式的数据，这里把类别转换成float类型数据
    # # 调用接口进行分析
    # corr_matrix = plot_correlation_analyze(
    #     df=data_combined_all,
    #     col_names=COL_NAMES+TARGET_NAME,
    #     method='pearson',
    #     figsize=(14, 14)
    # )
    # # 查看相关系数矩阵
    # print("\n相关系数矩阵:")
    # print(corr_matrix)
