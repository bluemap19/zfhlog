from matplotlib import pyplot as plt
from src_data_process.data_balanace import smart_balance_dataset
from src_data_process.data_correction_analysis import data_correction_analyse_by_tree
from src_data_process.data_filter import pdnads_data_drop, pandas_data_filtration
from src_data_process.data_supervised import supervised_classification, model_predict
from src_plot.plot_acc_heatmap import plot_clustering_heatmap
from src_plot.plot_matrxi_scatter import plot_matrxi_scatter
from src_well_project.LOGGING_PROJECT import LOGGING_PROJECT

LG = LOGGING_PROJECT(project_path=r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV')
print(LG.WELL_NAMES)
# well_texture1 = LG.get_well_data_by_charters(well_name='珠80', target_path_feature=['Texture_ALL', '_50_5'], curve_names=['DEPTH', 'dy_con', 'dy_ASM', 'ss_cor', 'ss_ASM'])
# print(well_texture1.describe())
# well_texture2 = LG.get_well_data_by_charters(well_name='城96', target_path_feature=['Texture_ALL', '_50_5'], curve_names=['DEPTH', 'dy_con', 'dy_ASM', 'ss_cor', 'ss_ASM'])
# print(well_texture2.describe())
path_table_1 = LG.search_target_file_path(well_name='城96', target_path_feature=['litho_type'], target_file_type='table')
print(path_table_1)

table_3 = LG.get_table_3_data(well_name='城96', file_path=path_table_1)
print(table_3)


# path_table_1 = LG.search_target_file_path(well_name='珠80', target_path_feature=['litho_type'], target_file_type='table')
# path_table_2 = LG.search_target_file_path(well_name='城96', target_path_feature=['litho_type'], target_file_type='table')
# print(path_table_1, path_table_2)
#
# LG.get_table_3_all_data(['城96', '珠80'])
# print(LG.get_all_table_replace_dict(well_names=['城96', '珠80']))
# dict = {'中GR长英黏土质': 0, '中GR长英黏土质（泥岩）': 0, '中低GR长英质': 1, '中低GR长英质（砂岩）': 1,
#         '富有机质长英质': 2, '富有机质长英质页岩': 2, '富有机质黏土质': 3, '富有机质黏土质页岩': 3, '高GR富凝灰长英质': 4, '高GR富凝灰长英质（沉凝灰岩）': 4}
#
# data_combined_all = LG.combined_all_logging_with_type(well_names=['城96', '珠80'], file_path_logging={'城96':path_logging_2, '珠80':path_logging_1},
#                                                       file_path_table={'城96':path_table_2, '珠80':path_table_1}, curve_names_logging=['dy_con', 'dy_ASM', 'ss_cor', 'ss_ASM'],
#                                                       replace_dict=dict, type_col_name='Type_litho', Norm=True)
# print(data_combined_all.describe())


