import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src_file_op.dir_operation import search_files_by_criteria
from src_plot.plot_heatmap import create_acc_heatmap
from src_plot.plot_matrxi_scatter import plot_matrxi_scatter
from src_well_project.LOGGING_PROJECT import LOGGING_PROJECT


if __name__ == '__main__':
    LG = LOGGING_PROJECT(project_path=r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV')
    # path_logging = LG.search_target_file_path(well_name='城96', target_path_feature=['Texture_ALL', '_240_5'],
    path_logging = LG.search_target_file_path(well_name='城96', target_path_feature=['Texture_ALL', '_100_5'],
                                                target_file_type='logging')
    print(path_logging)

    path_table = LG.search_target_file_path(well_name='城96', target_path_feature=['litho_type'],
                                              target_file_type='table')
    print(path_table)

    table = LG.get_table_3_all_data(['城96'])
    # print(table)
    print(LG.get_all_table_replace_dict(well_names=['城96'], file_path={'城96':path_table}))
    # dict = {'中GR长英黏土质': 0, '中GR长英黏土质（泥岩）': 0, '中低GR长英质': 1, '中低GR长英质（砂岩）': 1,
    #         '富有机质长英质': 2, '富有机质长英质页岩': 2, '富有机质黏土质': 3, '富有机质黏土质页岩': 3,
    #         '高GR富凝灰长英质': 4, '高GR富凝灰长英质（沉凝灰岩）': 4}

    path_logging_target = LG.search_target_file_path(well_name='城96',
                                                     # target_path_feature=['Texture_ALL', '_200_5'],
                                                     target_path_feature=['Texture_ALL', '_100_5'],
                                                     target_file_type='logging')
    path_table_target = LG.search_target_file_path(well_name='城96',
                                                   target_path_feature=['litho_type'],
                                                   target_file_type='table')
    # print('get data from path:{}, get table from:{}'.format(path_logging_target, path_table_target))

    replace_dict = {'中GR长英黏土质': 0, '中低GR长英质': 1, '富有机质长英质页岩': 2, '富有机质黏土质页岩': 3, '高GR富凝灰长英质': 4}
    COL_NAMES = [
        # windows = 200
        # 'STAT_CON', 'STAT_ENT', 'STAT_HOM', 'STAT_XY_CON', 'DYNA_DIS', 'STAT_XY_HOM', 'STAT_DIS', 'DYNA_HOM'

        # windows = 100
        'STAT_ENT', 'STAT_DIS', 'STAT_CON', 'STAT_XY_HOM', 'STAT_HOM', 'STAT_XY_CON', 'DYNA_DIS', 'STAT_ENG',

        # 'STAT_CON', 'STAT_DIS', 'STAT_HOM', 'STAT_ENG', 'STAT_COR', 'STAT_ASM', 'STAT_ENT', 'STAT_XY_CON',
        # 'STAT_XY_DIS', 'STAT_XY_HOM', 'STAT_XY_ENG', 'STAT_XY_COR', 'STAT_XY_ASM', 'STAT_XY_ENT',
        # 'DYNA_CON', 'DYNA_DIS', 'DYNA_HOM', 'DYNA_ENG', 'DYNA_COR', 'DYNA_ASM', 'DYNA_ENT', 'DYNA_XY_CON',
        # 'DYNA_XY_DIS', 'DYNA_XY_HOM', 'DYNA_XY_ENG', 'DYNA_XY_COR', 'DYNA_XY_ASM', 'DYNA_XY_ENT'
    ]
    TARGET_NAME = ['LITHO']

    data_combined_all = LG.combined_all_logging_with_type(well_names=['城96'],
                                                   file_path_logging={'城96': path_logging_target},
                                                   file_path_table={'城96': path_table_target},
                                                   replace_dict=replace_dict, type_new_col=TARGET_NAME[0],
                                                   curve_names_logging=COL_NAMES,
                                                   Norm=True)
    print(data_combined_all.describe())

    # 这里是整体上看一下ACC在 窗长-随机森林参数 上的分布特征

    TARGET_COL_NAMES = [
        # 'STAT_CON', 'STAT_ENT', 'STAT_HOM', 'STAT_XY_CON', 'DYNA_DIS', 'STAT_XY_HOM', 'STAT_DIS', 'DYNA_HOM',

        # windows = 100
        'STAT_ENT', 'STAT_DIS', 'STAT_CON', 'STAT_XY_HOM', 'STAT_HOM', 'STAT_XY_CON',
        # 'DYNA_DIS', 'STAT_ENG',

        # 'STAT_CON', 'STAT_DIS', 'STAT_HOM', 'STAT_ENG', 'STAT_COR', 'STAT_ASM', 'STAT_ENT', 'STAT_XY_CON',
        # 'STAT_XY_DIS', 'STAT_XY_HOM', 'STAT_XY_ENG', 'STAT_XY_COR', 'STAT_XY_ASM', 'STAT_XY_ENT',
        # 'DYNA_CON', 'DYNA_DIS', 'DYNA_HOM', 'DYNA_ENG', 'DYNA_COR', 'DYNA_ASM', 'DYNA_ENT', 'DYNA_XY_CON',
        # 'DYNA_XY_DIS', 'DYNA_XY_HOM', 'DYNA_XY_ENG', 'DYNA_XY_COR', 'DYNA_XY_ASM', 'DYNA_XY_ENT'
    ]
    # target_col_dict = {0:'中GR长英黏土质', 1:'中低GR长英质', 2:'富有机质长英质页岩', 3:'富有机质黏土质页岩', 4:'高GR富凝灰长英质'}
    target_col_dict = {0:'块状构造泥岩', 1:'薄层砂岩', 2:'富有机质粉砂级长英质页岩', 3:'富有机质富凝灰质页岩', 4:'沉凝灰岩'}
    plot_matrxi_scatter(df=data_combined_all, input_names=TARGET_COL_NAMES, target_col=TARGET_NAME[0],
                        plot_string='输入属性分布',
                        target_col_dict = target_col_dict
                        )




