import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src_data_process.data_balanace import smart_balance_dataset
from src_data_process.data_correlation_analysis_old import data_correction_analyse_by_tree
from src_data_process.data_correlation_analysis_old import random_forest_correlation_analysis
from src_data_process.data_filter import pdnads_data_drop, pandas_data_filtration
from src_data_process.data_supervised import supervised_classification, model_predict
from src_plot.plot_heatmap import plot_clustering_heatmap
from src_plot.plot_matrxi_scatter import plot_matrxi_scatter
from src_well_project.LOGGING_PROJECT import LOGGING_PROJECT


if __name__ == '__main__':
    # path_project = r'C:\Users\ZFH\Documents\simu-result'
    path_project = r'F:\电成像模拟结果'
    LG = LOGGING_PROJECT(project_path=path_project)
    print(LG.WELL_NAMES)

    windows_length = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220]
    tree_num_list = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    # windows_length = [140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]
    # tree_num_list = [8, 9, 10, 11, 12]
    # windows_length = [40, 60, 80]
    # tree_num_list = [10, 12]

    ACC_windows_length_dict = {}  # 所有的正确率结果 字典   正确率np.array([n,]),每一个类别的正确率 + 平均正确率
    AUC_windows_length_dict = {}  # 所有的AUC分数结果 字典     float
    Imp_windows_length_dict = {}  # 所有的Importance结果 字典    np.array([24,]) 24项纹理特征影响因子数组
    # 这三个字点都是字典含字典的格式,两个索引分别是 窗长-树个数

    path_table_target = LG.search_target_file_path(well_name='simu4', target_path_feature=['LITHO_TYPE'], target_file_type='table')
    print(LG.get_table_3_all_data(well_names=['simu4'], file_path={'simu4':path_table_target}))
    print(LG.get_all_table_replace_dict(well_names=['simu4']))

    replace_dict = {'块状构造泥岩': 0, '富有机质富凝灰质页岩': 1, '富有机质粉砂级长英质页岩': 2, '沉凝灰岩': 3, '薄夹层砂岩': 4}
    COL_NAMES = [
                'STAT_CON', 'STAT_DIS', 'STAT_HOM', 'STAT_ENG', 'STAT_COR', 'STAT_ASM', 'STAT_ENT', 'STAT_XY_CON',
                'STAT_XY_DIS', 'STAT_XY_HOM', 'STAT_XY_ENG', 'STAT_XY_COR', 'STAT_XY_ASM', 'STAT_XY_ENT',
                'DYNA_CON', 'DYNA_DIS', 'DYNA_HOM', 'DYNA_ENG', 'DYNA_COR', 'DYNA_ASM', 'DYNA_ENT', 'DYNA_XY_CON',
                'DYNA_XY_DIS', 'DYNA_XY_HOM', 'DYNA_XY_ENG', 'DYNA_XY_COR', 'DYNA_XY_ASM', 'DYNA_XY_ENT'
                ]
    TARGET_NAME = ['Type']

    Type_Y = []
    for i in windows_length:
        ACC_tree_num_dict = {}
        AUC_tree_num_dict = {}
        Imp_tree_num_dict = {}
        path_logging_target = LG.search_target_file_path(well_name='simu4', target_path_feature=['Texture_ALL', f'_{i}_5'], target_file_type='logging')
        path_table_target = LG.search_target_file_path(well_name='simu4', target_path_feature=['LITHO_TYPE'], target_file_type='table')
        print('get data from path:{}, get table from:{}'.format(path_logging_target, path_table_target))

        data_input = LG.combined_all_logging_with_type(well_names=['simu4'],
                                                       file_path_logging={'simu4': path_logging_target},
                                                       file_path_table={'simu4': path_table_target},
                                                       replace_dict=replace_dict, type_new_col=TARGET_NAME[0],
                                                       curve_names_logging=COL_NAMES,
                                                       Norm=True)
        print(data_input.describe())

        for j in tree_num_list:
            # exit(0)

            data_input = data_input[COL_NAMES+TARGET_NAME]

            scores, accuracies, auc_score, importances = random_forest_correlation_analysis(data_input, random_seed=44,
                                                                                            plot_index=[2, 2],
                                                                                            figsize=(16, 5),
                                                                                            tree_num=j, Norm=False)

            # print(accuracies, '\n', auc_score, '\n', importances)

            ACC_tree_num_dict[j] = accuracies
            AUC_tree_num_dict[j] = auc_score
            Imp_tree_num_dict[j] = importances

        ACC_windows_length_dict[i] = ACC_tree_num_dict
        AUC_windows_length_dict[i] = AUC_tree_num_dict
        Imp_windows_length_dict[i] = Imp_tree_num_dict


    # print(ACC_windows_length_dict)
    # print(AUC_windows_length_dict)
    # print(Imp_windows_length_dict)

    # 提取所有类别标签（假设所有内部字典有相同的类别）
    class_labels = sorted(next(iter(next(iter(ACC_windows_length_dict.values())).values())).keys())
    class_columns = [f"类别{int(label)}" for label in class_labels]

    # 创建数据列表
    ACC_LIST = []
    for window_size, rf_dict in ACC_windows_length_dict.items():
        for rf_param, class_dict in rf_dict.items():
            row = [window_size, rf_param]
            for label in class_labels:
                row.append(class_dict[label])
            ACC_LIST.append(row)

    # 创建列名
    columns = ["窗长", "随机森林参数"] + class_columns

    # 转换为NumPy数组
    accuracy_array = np.array(ACC_LIST)

    # 转换为Pandas DataFrame（更易查看）
    ACC_df = pd.DataFrame(accuracy_array, columns=columns)
    print(ACC_df.describe())
    ACC_df.to_excel(
        path_project + "\\ACC_win-{}-{}_tree-{}-{}.xlsx".format(
            windows_length[0], windows_length[-1], tree_num_list[0], tree_num_list[-1]), index=False,
        sheet_name='sheet_temp')

    # 确定属性数量（从第一个数组中获取）
    num_attributes = len(next(iter(next(iter(Imp_windows_length_dict.values())).values())))

    # 创建数据列表
    IMP_LIST = []
    for window_size, rf_dict in Imp_windows_length_dict.items():
        for rf_param, importance_array in rf_dict.items():
            # 创建新行：窗长 + 随机森林参数 + 所有属性重要性值
            new_row = [window_size, rf_param] + list(importance_array)
            IMP_LIST.append(new_row)

    # 创建列名
    columns = ["窗长", "随机森林参数"] + COL_NAMES

    # 转换为NumPy数组
    importance_array = np.array(IMP_LIST)

    # 转换为Pandas DataFrame（更易查看）
    IMP_df = pd.DataFrame(importance_array, columns=columns)
    print(IMP_df.describe())
    IMP_df.to_excel(
        path_project + "\\IMP_win-{}-{}_tree-{}-{}.xlsx".format(
            windows_length[0], windows_length[-1], tree_num_list[0], tree_num_list[-1]), index=False,
        sheet_name='sheet_temp')


