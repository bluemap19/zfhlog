import numpy as np
import pandas as pd

from src_data_process.data_correction_analysis import feature_influence_analysis
from src_data_process.data_dilute import dilute_dataframe
from src_data_process.data_distribution_statistics_overview import data_overview
from src_data_process.data_fisher import fisher_discriminant_analysis, fisher_apply
from src_plot.Plot_boxplots import plot_boxes
from src_plot.plot_matrxi_scatter import plot_matrxi_scatter
from src_table.table_process import table_2_to_3
from src_well_data.DATA_WELL import DATA_WELL





if __name__ == '__main__':

    WELL_TEST = DATA_WELL(path_folder=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\城96', WELL_NAME='城96')
    print(WELL_TEST.get_type_3(table_key='F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\城96__纹理岩相划分_LITHO_TYPE.csv'))
    # table_3 = WELL_TEST.get_type_3(table_key='F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\城96__LITHO_TYPE_Fisher_result.csv')
    # print(table_3)
    # table_3.to_csv('F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96'+'\\result_fisher_3col.csv', index=True)

    replace_dict = WELL_TEST.get_table_replace_dict(table_key='F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\城96__纹理岩相划分_LITHO_TYPE.csv')
    replace_dict_inversed = {value: key for key, value in replace_dict.items()}
    print(replace_dict, '--->', replace_dict_inversed)


    # data_combined = WELL_TEST.combine_logging_table(
    #     well_key='F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\Texture_File\\城96_texture_logging_Texture_ALL_80_5.csv',
    #     curve_names_logging=[],
    #     table_key='F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\城96__纹理岩相划分_LITHO_TYPE.csv',
    #     curve_names_table=[],
    #     replace_dict=replace_dict, new_col='', Norm=False)
    #
    # print(data_combined.describe())
    # print(data_combined.columns)
    #
    # col_all = ['DYNA_CON', 'DYNA_DIS', 'DYNA_HOM', 'DYNA_ENG', 'DYNA_COR',
    #    'DYNA_ASM', 'DYNA_ENT', 'STAT_CON', 'STAT_DIS', 'STAT_HOM', 'STAT_ENG',
    #    'STAT_COR', 'STAT_ASM', 'STAT_ENT', 'DYNA_XY_CON', 'DYNA_XY_DIS',
    #    'DYNA_XY_HOM', 'DYNA_XY_ENG', 'DYNA_XY_COR', 'DYNA_XY_ASM',
    #    'DYNA_XY_ENT', 'STAT_XY_CON', 'STAT_XY_DIS', 'STAT_XY_HOM',
    #    'STAT_XY_ENG', 'STAT_XY_COR', 'STAT_XY_ASM', 'STAT_XY_ENT']
    # col_target = ['Type']
    #
    # # pearson_result, pearson_sorted, rf_result, rf_sorted = feature_influence_analysis(
    # #     df_input=data_combined[col_all+col_target],
    # #     input_cols=col_all,
    # #     target_col=col_target[0],
    # #     replace_dict=replace_dict_inversed
    # # )
    # # print("\n皮尔逊相关系数结果:", pearson_result)
    # # print("\n按皮尔逊系数排序的属性:", pearson_sorted)
    # # print("\n随机森林特征重要性结果:", rf_result)
    # # print("\n按随机森林特征重要性排序的属性:", rf_sorted)
    #
    # col_choiced = ['DYNA_XY_ENG', 'DYNA_XY_HOM', 'DYNA_XY_ENT', 'STAT_XY_HOM', 'STAT_XY_ENT', 'STAT_XY_ENG']
    # # 计算F1和F2
    # data_combined['F1'] = (
    #         -68.417 * data_combined['DYNA_XY_ENG']
    #         - 160.874 * data_combined['DYNA_XY_HOM']
    #         - 27.089 * data_combined['DYNA_XY_ENT']
    #         - 33.603 * data_combined['STAT_XY_HOM']
    #         + 0.824 * data_combined['STAT_XY_ENT']
    #         + 31.946 * data_combined['STAT_XY_ENG']
    #         - 10.559
    # )
    # data_combined['F2'] = (
    #         -7.067 * data_combined['DYNA_XY_ENG']
    #         - 150.926 * data_combined['DYNA_XY_HOM']
    #         - 24.937 * data_combined['DYNA_XY_ENT']
    #         + 15.355 * data_combined['STAT_XY_HOM']
    #         + 11.117 * data_combined['STAT_XY_ENT']
    #         + 35.915 * data_combined['STAT_XY_ENG']
    #         - 6.958
    # )
    # # 使用向量化操作替代循环
    # data_combined['Fisher_Type'] = np.where(
    #     data_combined['F1'] > data_combined['F2'],
    #     0,
    #     1
    # )
    # print(data_combined.describe())
    # Table_fisher_2 = data_combined[['DEPTH', 'Fisher_Type']].values
    # Table_fisher_3 = table_2_to_3(Table_fisher_2)
    # DF_Table_fisher_3 = pd.DataFrame(Table_fisher_3, columns=['DEPTH_START', 'DEPTH_END', 'Fisher_Type'])
    # print(DF_Table_fisher_3.describe())
    # DF_Table_fisher_3.to_csv('F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96'+'\\result_fisher_3cols.csv', index=True)
    #
    #
    # # data_plot = dilute_dataframe(data_combined[col_choiced+col_target], ratio=10, method='random', group_by=col_target[0])
    # # plot_matrxi_scatter(df=data_plot,
    # #                     input_names=col_choiced,
    # #                     target_col=col_target[0],
    # #                     target_col_dict=replace_dict_inversed,
    # #                     figsize=(12, 11))
    # #
    # # plot_boxes(df=data_combined[col_choiced+col_target],
    # #            input_names=col_choiced,
    # #            target_col = col_target[0],
    # #            target_col_dict=replace_dict_inversed,
    # #            figsize=(24, 5))
    # #
    # # # 执行分析
    # # result = data_overview(
    # #     df=data_combined[col_choiced+col_target],
    # #     input_names=col_choiced,
    # #     target_col=col_target[0],
    # #     target_col_dict=replace_dict_inversed
    # # )
    # # print(result)
    # # # result.to_excel('F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96'+'\\result_statics_纹状层状_all.xlsx', index=True)
    # # # # result.to_csv(PATH_FOLDER+'\\result_statics——1.xlsx', index=False)
    #
    # # # 调用接口，接收额外返回值
    # # fisher_coef, class_centers, class_labels, train_acc, test_acc, scaler, coef, intercept = fisher_discriminant_analysis(
    # #     data_all=data_combined[col_choiced+col_target],
    # #     col_input=col_choiced,
    # #     col_target=col_target[0],
    # # )
    # # result = fisher_apply(
    # #     data_combined[['DEPTH']+col_choiced+col_target],
    # #     col_choiced,
    # #     fisher_coef,
    # #     class_centers,
    # #     class_labels,
    # #     scaler,
    # #     coef,
    # #     intercept,
    # # )
    # # result.to_csv('F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96'+'\\result_fisher.csv', index=False)
    #
    # # table_2_to_3()