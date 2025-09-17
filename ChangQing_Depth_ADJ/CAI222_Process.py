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
    # Path_WELL = r'F:\桌面\算法测试-长庆数据收集\logging_CSV\蔡222'
    Path_WELL = r'F:\桌面\算法测试-长庆数据收集\logging_CSV\乔12'
    WELL_TEST = DATA_WELL(path_folder=Path_WELL)
    print(WELL_TEST.file_path_dict)

    data_combined = WELL_TEST.get_logging_data(well_key='F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\乔12\\乔12_texture_logging_Texture_ALL_80_5.csv')
    print(data_combined.describe())

    # 计算F1和F2
    data_combined['F1'] = (
            -68.417 * data_combined['DYNA_XY_ENG']
            - 160.874 * data_combined['DYNA_XY_HOM']
            - 27.089 * data_combined['DYNA_XY_ENT']
            - 33.603 * data_combined['STAT_XY_HOM']
            + 0.824 * data_combined['STAT_XY_ENT']
            + 31.946 * data_combined['STAT_XY_ENG']
            - 10.559
    )
    data_combined['F2'] = (
            -7.067 * data_combined['DYNA_XY_ENG']
            - 150.926 * data_combined['DYNA_XY_HOM']
            - 24.937 * data_combined['DYNA_XY_ENT']
            + 15.355 * data_combined['STAT_XY_HOM']
            + 11.117 * data_combined['STAT_XY_ENT']
            + 35.915 * data_combined['STAT_XY_ENG']
            - 6.958
    )
    # 使用向量化操作替代循环
    data_combined['Fisher_Type'] = np.where(
        data_combined['F1'] > data_combined['F2'],
        0,
        1
    )
    print(data_combined.describe())
    Table_fisher_2 = data_combined[['DEPTH', 'Fisher_Type']].values
    Table_fisher_3 = table_2_to_3(Table_fisher_2)
    DF_Table_fisher_3 = pd.DataFrame(Table_fisher_3, columns=['DEPTH_START', 'DEPTH_END', 'Fisher_Type'])
    print(DF_Table_fisher_3.describe())
    DF_Table_fisher_3.to_csv(Path_WELL+'\\result_fisher_3cols.csv', index=True)
    data_combined.to_csv(Path_WELL+'\\result_fisher_2cols.csv', index=False)


    # data_plot = dilute_dataframe(data_combined[col_choiced+col_target], ratio=10, method='random', group_by=col_target[0])
    # plot_matrxi_scatter(df=data_plot,
    #                     input_names=col_choiced,
    #                     target_col=col_target[0],
    #                     target_col_dict=replace_dict_inversed,
    #                     figsize=(12, 11))
    #
    # plot_boxes(df=data_combined[col_choiced+col_target],
    #            input_names=col_choiced,
    #            target_col = col_target[0],
    #            target_col_dict=replace_dict_inversed,
    #            figsize=(24, 5))
    #
    # # 执行分析
    # result = data_overview(
    #     df=data_combined[col_choiced+col_target],
    #     input_names=col_choiced,
    #     target_col=col_target[0],
    #     target_col_dict=replace_dict_inversed
    # )
    # print(result)
    # # result.to_excel('F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96'+'\\result_statics_纹状层状_all.xlsx', index=True)
    # # # result.to_csv(PATH_FOLDER+'\\result_statics——1.xlsx', index=False)

    # # 调用接口，接收额外返回值
    # fisher_coef, class_centers, class_labels, train_acc, test_acc, scaler, coef, intercept = fisher_discriminant_analysis(
    #     data_all=data_combined[col_choiced+col_target],
    #     col_input=col_choiced,
    #     col_target=col_target[0],
    # )
    # result = fisher_apply(
    #     data_combined[['DEPTH']+col_choiced+col_target],
    #     col_choiced,
    #     fisher_coef,
    #     class_centers,
    #     class_labels,
    #     scaler,
    #     coef,
    #     intercept,
    # )
    # result.to_csv('F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96'+'\\result_fisher.csv', index=False)

    # table_2_to_3()