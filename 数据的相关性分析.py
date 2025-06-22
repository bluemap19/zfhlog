import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src_data_process.data_balanace import smart_balance_dataset
from src_data_process.data_correction_analysis import data_correction_analyse_by_tree
from src_data_process.data_correlation_analysis import random_forest_correlation_analysis
from src_data_process.data_filter import pdnads_data_drop, pandas_data_filtration
from src_data_process.data_supervised import supervised_classification, model_predict
from src_plot.plot_acc_heatmap import plot_clustering_heatmap
from src_plot.plot_matrxi_scatter import plot_matrxi_scatter
from src_well_project.LOGGING_PROJECT import LOGGING_PROJECT


if __name__ == '__main__':
    LG = LOGGING_PROJECT(project_path=r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV')
    print(LG.WELL_NAMES)

    windows_length = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                    210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400,
                    410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600]
    tree_num_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # windows_length = [140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]
    # tree_num_list = [8, 9, 10, 11, 12]
    # windows_length = [240, 260, 280]
    # tree_num_list = [10, 12]

    ACC_windows_length_dict = {}  # 所有的正确率结果 字典   正确率np.array([n,]),每一个类别的正确率 + 平均正确率
    AUC_windows_length_dict = {}  # 所有的AUC分数结果 字典     float
    Imp_windows_length_dict = {}  # 所有的Importance结果 字典    np.array([24,]) 24项纹理特征影响因子数组
    # 这三个字点都是字典含字典的格式,两个索引分别是 窗长-树个数

    replace_dict = {'中GR长英黏土质': 0, '中低GR长英质': 1, '富有机质长英质页岩': 2, '富有机质黏土质页岩': 3, '高GR富凝灰长英质': 4}
    COL_NAMES = [
                # 'STAT_CON', 'STAT_DIS', 'STAT_HOM', 'STAT_ENG', 'STAT_COR', 'STAT_ASM', 'STAT_ENT', 'STAT_XY_CON',
                # 'STAT_XY_DIS', 'STAT_XY_HOM', 'STAT_XY_ENG', 'STAT_XY_COR', 'STAT_XY_ASM', 'STAT_XY_ENT',
                'DYNA_CON', 'DYNA_DIS', 'DYNA_HOM', 'DYNA_ENG', 'DYNA_COR', 'DYNA_ASM', 'DYNA_ENT', 'DYNA_XY_CON',
                'DYNA_XY_DIS', 'DYNA_XY_HOM', 'DYNA_XY_ENG', 'DYNA_XY_COR', 'DYNA_XY_ASM', 'DYNA_XY_ENT'
                ]
    TARGET_NAME = ['Type_litho']

    Type_Y = []
    for i in windows_length:
        ACC_tree_num_dict = {}
        AUC_tree_num_dict = {}
        Imp_tree_num_dict = {}
        for j in tree_num_list:
            path_logging_target = LG.search_target_file_path(well_name='城96', target_path_feature=['Texture_ALL', f'_{i}_5'], target_file_type='logging')
            path_table_target = LG.search_target_file_path(well_name='城96', target_path_feature=['litho_type'], target_file_type='table')
            # print('get data from path:{}, get table from:{}'.format(path_logging_target, path_table_target))

            data_input = LG.combined_all_logging_with_type(well_names=['城96'],
                                                           file_path_logging={'城96': path_logging_target},
                                                           file_path_table={'城96': path_table_target},
                                                           replace_dict=replace_dict, type_new_col=TARGET_NAME[0],
                                                           curve_names_logging=COL_NAMES,
                                                           Norm=True)
            # print(data_input.describe())
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
        r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV' + "\\ACC_win-{}-{}_tree-{}-{}.xlsx".format(
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
        r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV' + "\\IMP_win-{}-{}_tree-{}-{}.xlsx".format(
            windows_length[0], windows_length[-1], tree_num_list[0], tree_num_list[-1]), index=False,
        sheet_name='sheet_temp')
    # DF_ACC_ALL = pd.DataFrame(np.array(list(ACC_windows_length_dict.values())))
    # DF_ACC_ALL.to_excel(
    #     r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV' + "\\ACC_win-{}-{}_tree-{}-{}.xlsx".format(
    #         windows_length[0], windows_length[-1], tree_num_list[0], tree_num_list[-1]), index=False,
    #     sheet_name='sheet_temp')
    #
    # DF_IMP_ALL = pd.DataFrame(np.array(list(Imp_windows_length_dict.values())))
    # DF_IMP_ALL.to_excel(
    #     r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV' + "\\IMP_win-{}-{}_tree-{}-{}.xlsx".format(
    #         windows_length[0], windows_length[-1], tree_num_list[0], tree_num_list[-1]), index=False,
    #     sheet_name='sheet_temp')

    # path_logging_target = LG.search_target_file_path(well_name='城96', target_path_feature=['Texture_ALL', '_50_5'], target_file_type='logging')
    # print(path_logging_target)
    #
    # path_table_target = LG.search_target_file_path(well_name='城96', target_path_feature=['litho_type'], target_file_type='table')
    # print(path_table_target)
    #
    # # LG.get_table_3_all_data(['城96'])
    # # print(LG.get_all_table_replace_dict(well_names=['城96']))
    # # replace_dict = {'中GR长英黏土质': 0, '中GR长英黏土质（泥岩）': 0, '中低GR长英质': 1, '中低GR长英质（砂岩）': 1,
    # #         '富有机质长英质': 2, '富有机质长英质页岩': 2, '富有机质黏土质': 3, '富有机质黏土质页岩': 3, '高GR富凝灰长英质': 4, '高GR富凝灰长英质（沉凝灰岩）': 4}
    #
    #
    # data_input = LG.combined_all_logging_with_type(well_names=['城96'], file_path_logging={'城96':path_logging_target},
    #                                                file_path_table={'城96':path_table_target},
    #                                                replace_dict=replace_dict, type_col_name='Type_litho', Norm=True)
    # # print(data_input.describe())
    # # print(data_input.columns)
    # data_input = data_input[['STAT_CON', 'STAT_HOM', 'STAT_ENT', 'STAT_XY_HOM', 'Type_litho']]
    # # data_input = data_input[['STAT_CON', 'STAT_DIS', 'STAT_HOM', 'STAT_ENG', 'STAT_COR', 'STAT_ASM', 'STAT_ENT', 'STAT_XY_CON',
    # #                 'STAT_XY_DIS', 'STAT_XY_HOM', 'STAT_XY_ENG', 'STAT_XY_COR', 'STAT_XY_ASM', 'STAT_XY_ENT', 'Type_litho']]
    # # data_input = data_input[['DYNA_CON', 'DYNA_DIS', 'DYNA_HOM', 'DYNA_ENG', 'DYNA_COR', 'DYNA_ASM', 'DYNA_ENT', 'DYNA_XY_CON',
    # #                 'DYNA_XY_DIS', 'DYNA_XY_HOM', 'DYNA_XY_ENG', 'DYNA_XY_COR', 'DYNA_XY_ASM', 'DYNA_XY_ENT', 'Type_litho']]
    #
    # scores, accuracies, auc_score, importances = random_forest_correlation_analysis(data_input, random_seed=44, plot_index=[2, 2], figsize=(16, 5), tree_num=10, Norm=False)
    # # 输出结果
    # print("交叉验证分数:", scores)
    # print("类别准确率:", accuracies)
    # print("AUC总分:", auc_score)
    # print("特征重要性:", importances)



    # data_combined_all = LG.combined_all_logging_with_type(well_names=['城96', '珠80'], file_path_logging={'城96':path_logging_target, '珠80':path_logging_1},
    #                                                       file_path_table={'城96':path_table_target, '珠80':path_table_1}, curve_names_logging=['dy_con', 'dy_ASM', 'ss_cor', 'ss_ASM'],
    #                                                       replace_dict=dict, type_col_name='Type_litho', Norm=True)
    # print(data_combined_all.describe())

    # LG.combined_all_logging_with_type_by_charters(  well_names=['城96', '珠80'], logging_charters=['Texture_ALL', '_50_5'], table_charters=['litho_type'],
    #                                                 curve_names_logging=['DEPTH', 'dy_con', 'dy_ASM', 'ss_cor', 'ss_ASM'], curve_names_type=['DEPTH', '岩相'],
    #                                                 replace_dict=dict, type_col_name='Type_num', Norm=True)

    # # LG = LOGGING_PROJECT(project_path=r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV')
    # # curves_list = ['AC', 'CNL', 'GR', 'DEN', 'SP', 'RT', 'CAL']
    # curves_list = ['CNL', 'GR', 'DEN', 'RT']
    # # curves_list = ['AC', 'CNL', 'GR']


    # LG.get_table_3_all_data()
    # ALL_REPLACE_DICT = LG.get_all_table_replace_dict()
    # print(ALL_REPLACE_DICT)
    # dict = {'中GR长英黏土质': 0, '中GR长英黏土质（泥岩）': 0, '中低GR长英质': 1, '中低GR长英质（砂岩）': 1,
    #                                    '富有机质长英质': 2, '富有机质长英质页岩': 2, '富有机质黏土质': 3, '富有机质黏土质页岩': 3,
    #                                    '高GR富凝灰长英质': 4, '高GR富凝灰长英质（沉凝灰岩）': 4}
    # LG.set_all_table_replace_dict(dict=dict)
    #
    # # '元543', '元552', '悦235', '珠23'
    # # data_all = LG.combined_all_logging_with_type(well_names=['元543', '元552', '悦235', '珠23'],
    # data_all = LG.combined_all_logging_with_type(well_names=['元543', '元552'],
    # # data_all = LG.combined_all_logging_with_type(well_names=[],
    #                                       curve_names_logging=curves_list, Norm=True)
    # print(f'data_all:{data_all.shape}\n', data_all.head())
    # data_all_dropped = pdnads_data_drop(data_all)
    # print(f'data_all_dropped:{data_all_dropped.shape}\n', data_all_dropped.head())
    # index_filter = pandas_data_filtration(data_all_dropped[curves_list])
    # data_all_dropped_filted = data_all_dropped.iloc[index_filter]
    # print(f'data_all_dropped_filted:{data_all_dropped_filted.shape}\n', data_all_dropped_filted.head())
    #
    # dict_rf = {'中GR长英黏土质': 0, '中低GR长英质': 1, '富有机质长英质': 2, '富有机质黏土质': 3, '高GR富凝灰长英质': 4}
    # data_correction_analyse_by_tree(data_all_dropped_filted, curves_list, 'Type', 10,
    #                                 y_replace_dict=dict_rf, title_string='fffffffffuck')
    # data_all_dropped_filted_balanced = smart_balance_dataset(data_all_dropped_filted[curves_list+['Type']],
    #                               target_col='Type', method='smote', Type_dict=dict_rf)
    # # 进行数据的监督聚类分析 聚类分析的结果百分比
    # df_result, classifiers = supervised_classification(data_all_dropped_filted_balanced[curves_list],
    #                                                   data_all_dropped_filted_balanced['Type'],
    #                                                    Type_str=dict_rf)
    # print(df_result)
    # # 使用监督模型进行输入数据的预测
    # model_classify_result = model_predict(classifiers, data_all_dropped_filted[curves_list])
    # # print(model_classify_result.shape, model_classify_result)
    # plot_clustering_heatmap(
    #     df_result,
    #     title="Supervised Classify Result",
    #     condition_formatter=lambda x: f"Model {x}",
    #     font_scale=0.9
    # )
    # plt.show()
    #
    #
    # plot_matrxi_scatter(df=data_all_dropped_filted, input_names=curves_list, target_col='Type', plot_string='分类相关性', target_col_dict=dict_rf)
    # plot_matrxi_scatter(df=data_all_dropped_filted, input_names=curves_list, target_col='Well_Name', plot_string='分井')


