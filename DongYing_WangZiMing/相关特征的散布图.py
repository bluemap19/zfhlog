from src_data_process.data_correlation_analysis_old import random_forest_correlation_analysis
from src_data_process.data_distribution_statistics_overview import data_overview
from src_plot.plot_correlation import plot_correlation_analyze
from src_plot.plot_matrxi_scatter import plot_matrxi_scatter
from src_well_project.LOGGING_PROJECT import LOGGING_PROJECT


if __name__ == '__main__':
    path_project = r'C:\Users\ZFH\Desktop\1-15'
    LG = LOGGING_PROJECT(project_path=path_project)
    path_logging_target = LG.search_target_file_path(well_name='well_texture_all', target_path_feature=['texture', '_80_'],
                                              target_file_type='logging')
    print(path_logging_target)

    path_table_target = LG.search_target_file_path(well_name='well_texture_all', target_path_feature=['Litho_Type'],
                                            target_file_type='table')

    table = LG.get_table_3_all_data(['well_texture_all'])
    # print(table)
    print(LG.get_all_table_replace_dict(well_names=['well_texture_all'], file_path={'well_texture_all': path_table_target}))
    replace_dict = {'亮晶': 0, '块状': 1, '层界面': 2, '隐晶': 3}
    COL_NAMES = ('CON\tDIS\tHOM\tENG\tCOR\tASM\tENT\t'
                 'XY_CON\tXY_DIS\tXY_HOM\tXY_ENG\tXY_COR\tXY_ASM\tXY_ENT').split('\t')
    TARGET_NAME = ['Type']

    data_combined_all = LG.combined_all_logging_with_type(well_names=['well_texture_all'],
                                                          file_path_logging={'well_texture_all': path_logging_target},
                                                          file_path_table={'well_texture_all': path_table_target},
                                                          replace_dict=replace_dict, type_new_col=TARGET_NAME[0],
                                                          curve_names_logging=COL_NAMES,
                                                          Norm=True)
    print(data_combined_all.describe())

    # # 这里是整体上看一下ACC在 窗长-随机森林参数 上的分布特征
    replace_dict = {'亮晶': 0, '块状': 1, '层界面': 2, '隐晶': 3}
    COL_NAMES_Target = ['DIS', 'ENT', 'CON', 'COR', 'XY_DIS', 'XY_ASM']
    # # 调用接口进行分析
    print(data_combined_all[COL_NAMES].describe())
    # plot_correlation_analyze(data_combined_all, COL_NAMES_Target, method='pearson', figsize=(14, 14),
    #                          return_matrix=False)

    # plot_matrxi_scatter(data_combined_all, COL_NAMES_Target, 'Type',
    #                     target_col_dict=replace_dict)

    replace_dict_inversed = {value: key for key, value in replace_dict.items()}
    result = data_overview(df=data_combined_all,
                        input_names=COL_NAMES_Target,
                        target_col='Type',
                        target_col_dict=replace_dict_inversed)

    # 查看结果
    print(result.head())

    # 保存结果
    result.to_excel('data_overview.xlsx', index=True)

    # scores, accuracies, auc_score, importances = random_forest_correlation_analysis(data_combined_all[COL_NAMES+TARGET_NAME], random_seed=44,
    #                                                                                 plot_index=[2, 2],
    #                                                                                 figsize=(16, 5),
    #                                                                                 tree_num=9, Norm=False)
    # print(importances, accuracies, scores)


