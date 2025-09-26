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

    # WELL_TEST = DATA_WELL(path_folder=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\SIMU4')
    WELL_TEST = DATA_WELL(path_folder=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\白75')
    print(WELL_TEST.get_logging_path_list())
    print(WELL_TEST.get_FMI_path_list())



    # print(WELL_TEST.get_type_3(table_key='F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\城96__纹理岩相划分_LITHO_TYPE.csv'))
    # table_3 = WELL_TEST.get_type_3(table_key='F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\城96__LITHO_TYPE_Fisher_result.csv')
    # print(table_3)
    # table_3.to_csv('F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96'+'\\result_fisher_3col.csv', index=True)

    # replace_dict = WELL_TEST.get_table_replace_dict(table_key='F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\城96__纹理岩相划分_LITHO_TYPE.csv')
    # replace_dict_inversed = {value: key for key, value in replace_dict.items()}
    # print(replace_dict, '--->', replace_dict_inversed)