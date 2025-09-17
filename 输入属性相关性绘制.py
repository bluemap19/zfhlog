from src_data_process.data_correlation_analysis_old import random_forest_correlation_analysis
from src_plot.plot_correlation import plot_correlation_analyze
from src_well_project.LOGGING_PROJECT import LOGGING_PROJECT


if __name__ == '__main__':
    LG = LOGGING_PROJECT(project_path=r'C:\Users\ZFH\Desktop\simulated-dyna')
    path_logging = LG.search_target_file_path(well_name='simu1', target_path_feature=['Texture_ALL', '_80_5'],
                                              # path_logging = LG.search_target_file_path(well_name='simu1', target_path_feature=['Texture_ALL', '_100_5'],
                                              target_file_type='logging')
    print(path_logging)

    path_table = LG.search_target_file_path(well_name='simu1', target_path_feature=['LITHO_TYPE_1'],
                                            target_file_type='table')

    table = LG.get_table_3_all_data(['simu1'])
    # print(table)
    print(LG.get_all_table_replace_dict(well_names=['simu1'], file_path={'simu1': path_table}))
    # dict = {'中GR长英黏土质': 0, '中GR长英黏土质（泥岩）': 0, '中低GR长英质': 1, '中低GR长英质（砂岩）': 1,
    #         '富有机质长英质': 2, '富有机质长英质页岩': 2, '富有机质黏土质': 3, '富有机质黏土质页岩': 3,
    #         '高GR富凝灰长英质': 4, '高GR富凝灰长英质（沉凝灰岩）': 4}

    path_logging_target = LG.search_target_file_path(well_name='simu1',
                                                     # target_path_feature=['Texture_ALL', '_200_5'],
                                                     target_path_feature=['Texture_ALL', '_100_5'],
                                                     target_file_type='logging')
    path_table_target = LG.search_target_file_path(well_name='simu1',
                                                   target_path_feature=['litho_type_2'],
                                                   target_file_type='table')
    # print('get data from path:{}, get table from:{}'.format(path_logging_target, path_table_target))

    replace_dict = {'高密度层理': 0, '低密度层理': 1, '孔洞': 2, '高密度低角度层理': 0, '高密度高角度层理': 1, '低密度低角度层理': 2, '低密度高角度层理': 3, '高阻块状': 4, '低阻块状':5}
    COL_NAMES = [
        'STAT_CON', 'STAT_DIS', 'STAT_HOM', 'STAT_ENG', 'STAT_COR', 'STAT_ASM', 'STAT_ENT',
        'DYNA_CON', 'DYNA_DIS', 'DYNA_HOM', 'DYNA_ENG', 'DYNA_COR', 'DYNA_ASM', 'DYNA_ENT',
        'STAT_XY_CON', 'STAT_XY_DIS', 'STAT_XY_HOM', 'STAT_XY_ENG', 'STAT_XY_COR', 'STAT_XY_ASM', 'STAT_XY_ENT',
        'DYNA_XY_CON', 'DYNA_XY_DIS', 'DYNA_XY_HOM', 'DYNA_XY_ENG', 'DYNA_XY_COR', 'DYNA_XY_ASM', 'DYNA_XY_ENT',
    ]
    TARGET_NAME = ['LITHO']

    data_combined_all = LG.combined_all_logging_with_type(well_names=['simu1'],
                                                          file_path_logging={'simu1': path_logging_target},
                                                          file_path_table={'simu1': path_table_target},
                                                          replace_dict=replace_dict, type_new_col=TARGET_NAME[0],
                                                          curve_names_logging=COL_NAMES,
                                                          Norm=True)
    print(data_combined_all.describe())

    # # 这里是整体上看一下ACC在 窗长-随机森林参数 上的分布特征
    replace_dict = {'高密度层理': 0, '低密度层理': 1, '孔洞': 2, '高密度低角度层理': 0, '高密度高角度层理': 1, '低密度低角度层理': 2, '低密度高角度层理': 3, '高阻块状': 4, '低阻块状':5}

    # # 调用接口进行分析
    print(data_combined_all[COL_NAMES].describe())

    plot_correlation_analyze(data_combined_all, COL_NAMES, method='pearson', figsize=(14, 14),
                             return_matrix=False)


    # scores, accuracies, auc_score, importances = random_forest_correlation_analysis(data_combined_all[COL_NAMES+TARGET_NAME], random_seed=44,
    #                                                                                 plot_index=[2, 2],
    #                                                                                 figsize=(16, 5),
    #                                                                                 tree_num=9, Norm=False)
    # print(importances, accuracies, scores)

