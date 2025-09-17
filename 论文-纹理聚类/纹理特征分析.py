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
from src_well_project.LOGGING_PROJECT import LOGGING_PROJECT


if __name__ == '__main__':
    LG = LOGGING_PROJECT(project_path=r'F:\桌面\算法测试-长庆数据收集\logging_CSV')
    table = LG.get_table_3_all_data(['SIMU4'])
    print(table)

    path_logging_target = LG.search_target_file_path(well_name='SIMU4', target_path_feature=['logging'], target_file_type='logging')
    path_table_target = LG.search_target_file_path(well_name='SIMU4', target_path_feature=['LITHO_TYPE'], target_file_type='table')
    print('get data from path:{},\nget table from:{}'.format(path_logging_target, path_table_target))

    replace_dict = LG.get_all_table_replace_dict(well_names=['SIMU4'], Type_col_name='LITHO')
    replace_dict_inversed = {value:key for key, value in replace_dict.items()}
    print(replace_dict, '--->', replace_dict_inversed)


    # data_combined_all = LG.combined_all_logging_with_type(well_names=['SIMU4'],
    #                                                # file_path_logging={'SIMU4': path_logging_target},
    #                                                # file_path_table={'SIMU4': path_table_target},
    #                                                replace_dict=replace_dict,
    #                                                Norm=False)
    # print(data_combined_all.columns)
    # #    'DEPTH', 'DYNA_CON', 'DYNA_DIS', 'DYNA_HOM', 'DYNA_ENG', 'DYNA_COR',
    # #    'DYNA_ASM', 'DYNA_ENT', 'STAT_CON', 'STAT_DIS', 'STAT_HOM', 'STAT_ENG',
    # #    'STAT_COR', 'STAT_ASM', 'STAT_ENT', 'DYNA_XY_CON', 'DYNA_XY_DIS',
    # #    'DYNA_XY_HOM', 'DYNA_XY_ENG', 'DYNA_XY_COR', 'DYNA_XY_ASM',
    # #    'DYNA_XY_ENT', 'STAT_XY_CON', 'STAT_XY_DIS', 'STAT_XY_HOM',
    # #    'STAT_XY_ENG', 'STAT_XY_COR', 'STAT_XY_ASM', 'STAT_XY_ENT', 'Type',
    # #    'Well_Name'


    COL_NAMES_ALL = [
        'STAT_CON', 'STAT_DIS', 'STAT_HOM', 'STAT_ENG', 'STAT_COR', 'STAT_ASM', 'STAT_ENT', 'STAT_XY_CON',
        'STAT_XY_DIS', 'STAT_XY_HOM', 'STAT_XY_ENG', 'STAT_XY_COR', 'STAT_XY_ASM', 'STAT_XY_ENT',
        'DYNA_CON', 'DYNA_DIS', 'DYNA_HOM', 'DYNA_ENG', 'DYNA_COR', 'DYNA_ASM', 'DYNA_ENT', 'DYNA_XY_CON',
        'DYNA_XY_DIS', 'DYNA_XY_HOM', 'DYNA_XY_ENG', 'DYNA_XY_COR', 'DYNA_XY_ASM', 'DYNA_XY_ENT'
    ]
    TARGET_NAME = ['Type']
    data_combined_all = LG.combined_all_logging_with_type(well_names=['SIMU4'],
                                                   # file_path_logging={'SIMU4': path_logging_target},
                                                   # file_path_table={'SIMU4': path_table_target},
                                                   replace_dict=replace_dict, type_new_col=TARGET_NAME[0],
                                                   curve_names_logging=COL_NAMES_ALL,
                                                   Norm=False)
    print(data_combined_all.describe())
    print(data_combined_all.columns)
    data_combined_all[COL_NAMES_ALL] = data_combined_all[COL_NAMES_ALL].astype(float)
    data_combined_all[TARGET_NAME] = data_combined_all[TARGET_NAME].astype(int)

    # print(pd.unique(data_combined_all['Type']))
    # exit(0)

    pearson_result, pearson_sorted, rf_result, rf_sorted = feature_influence_analysis(
        df=data_combined_all,
        input_cols=COL_NAMES_ALL,
        target_col=TARGET_NAME[0],
        replace_dict=replace_dict_inversed
    )
    print("\n皮尔逊相关系数结果:", pearson_result)
    print("\n按皮尔逊系数排序的属性:", pearson_sorted)
    print("\n随机森林特征重要性结果:", rf_result)
    print("\n按随机森林特征重要性排序的属性:", rf_sorted[:8])
    COL_NAMES_CHOICED = rf_sorted[:8]

    # COL_NAMES_CHOICED = ['DYNA_XY_HOM', 'DYNA_XY_COR', 'DYNA_XY_DIS', 'DYNA_COR', 'DYNA_XY_ENT', 'STAT_CON', 'DYNA_CON', 'DYNA_HOM']
    # COL_NAMES_CHOICED = ['DYNA_XY_HOM', 'DYNA_XY_COR', 'DYNA_XY_DIS', 'DYNA_XY_ENT']
    # COL_NAMES_CHOICED = ['DYNA_COR', 'STAT_CON', 'DYNA_CON', 'DYNA_HOM']

    # plot_correlation_analyze(data_combined_all, COL_NAMES_CHOICED, method='pearson', figsize=(14, 14),
    #                          return_matrix=False)

    # replace_dict_inversed = {0: '块状构造泥岩', 1: '富有机质富凝灰质页岩', 2: '富有机质粉砂级长英质页岩', 3: '沉凝灰岩', 4: '薄夹层砂岩'}
    replace_dict_inversed = {0: 'E', 1: 'D', 2: 'C', 3: 'B', 4: 'A'}
    data_dilute = dilute_dataframe(data_combined_all[COL_NAMES_CHOICED+TARGET_NAME], ratio=50, group_by='Type')
    print(f"按组抽稀30%后形状: {data_dilute.shape}")

    # # 传入的dataframe，数据不能是整体，必须经过精心挑选，才能
    # plot_matrxi_scatter(df=data_dilute, input_names=COL_NAMES_CHOICED, target_col=TARGET_NAME[0],
    #                     target_col_dict=replace_dict_inversed, figsize=(12, 11))

    # # 表格化的数据统计，并进行保存
    # result = data_overview(
    #     df=data_combined_all[COL_NAMES_CHOICED+TARGET_NAME],
    #     input_names=COL_NAMES_CHOICED,
    #     target_col=TARGET_NAME[0],
    #     target_col_dict=replace_dict_inversed
    # )
    # print(result)
    # PATH_FOLDER = r'C:\Users\ZFH\Desktop'
    # result.to_excel(PATH_FOLDER+'\\result_statics_隐晶亮晶相_all.xlsx', index=True)

    # # 调用可视化接口
    # visualize_well_logs(
    #     data=data_combined_all,
    #     depth_col='DEPTH',
    #     curve_cols=['GR10', '_RD', '_RS', '_DEN', '_AC', '_CNL', '_QF', '_CLA', '_GRAY'],
    #     # curve_cols=['温度', '测量电阻率', '温度校正因子', '去势电阻率'],
    #     type_cols=['SOM_Cluster'],
    #     figsize=(22, 10)
    # )

    plot_boxes(df=data_dilute[COL_NAMES_CHOICED+TARGET_NAME], input_names=COL_NAMES_CHOICED, target_col=TARGET_NAME[0],
               target_col_dict=replace_dict_inversed, figsize=(24, 5))


