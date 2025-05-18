from src_data_process.data_filter import pdnads_data_drop, pandas_data_filtration
from src_plot.plot_matrxi_scatter import plot_matrxi_scatter
from src_well_project.LOGGING_PROJECT import LOGGING_PROJECT


LG = LOGGING_PROJECT()
# curves_list = ['AC', 'CNL', 'GR', 'DEN', 'SP', 'RT', 'CAL']
curves_list = ['CNL', 'GR', 'DEN', 'RT', 'SP']
# curves_list = ['AC', 'CNL', 'GR']
a = LG.get_well_data(well_name='白75', curve_names=curves_list)
# print(a)

data_vc = LG.get_table_3_all_data()
b = LG.get_all_table_replace_dict()
print(b)
dict = {'中GR长英黏土质': 0, '中GR长英黏土质（泥岩）': 0, '中低GR长英质': 1, '中低GR长英质（砂岩）': 1,
                                   '富有机质长英质': 2, '富有机质长英质页岩': 2, '富有机质黏土质': 3, '富有机质黏土质页岩': 3,
                                   '高GR富凝灰长英质': 4, '高GR富凝灰长英质（沉凝灰岩）': 4}
LG.set_all_table_replace_dict(dict=dict)

# '元543', '元552', '悦235', '珠23'
# c = LG.combined_all_logging_with_type(well_names=['元543', '元552', '悦235', '珠23'],
# c = LG.combined_all_logging_with_type(well_names=['元543', '元552'],
data_all = LG.combined_all_logging_with_type(well_names=[],
                                      curve_names_logging=curves_list, Norm=True)
print(f'C:{data_all.shape}\n', data_all.head())
data_all_dropped = pdnads_data_drop(data_all)
print(f'D:{data_all_dropped.shape}\n', data_all_dropped.head())
index_filter = pandas_data_filtration(data_all_dropped[curves_list])
data_all_dropped_filted = data_all_dropped.iloc[index_filter]
print(f'F:{data_all_dropped_filted.shape}\n', data_all_dropped_filted.head())

dict_rf = {'中GR长英黏土质': 0, '中低GR长英质': 1, '富有机质长英质': 2, '富有机质黏土质': 3, '高GR富凝灰长英质': 4}

# data_correction_analyse_by_tree(f, curves_list, 'Type', 10,
#                                 y_replace_dict=dict_rf, title_string='fffffffffuck')
# data_balanced = smart_balance_dataset(f[curves_list+['Type']], target_col='Type', method='smote', Type_dict=dict_rf)
# # 进行数据的监督聚类分析 聚类分析的结果百分比
# df_result, classifiers = supervised_classification(data_balanced[curves_list], data_balanced['Type'],
#                                                    Type_str=dict_rf)
# print(df_result)
# # 使用监督模型进行输入数据的预测
# model_classify_result = model_predict(classifiers, f[curves_list])
# # print(model_classify_result.shape, model_classify_result)
# plot_clustering_heatmap(
#     df_result,
#     title="Supervised Classify Result",
#     condition_formatter=lambda x: f"Model {x}",
#     font_scale=0.9
# )
# plt.show()

plot_matrxi_scatter(df=data_all_dropped_filted, input_names=curves_list, target_col='Type', plot_string='分类相关性', target_col_dict=dict_rf)
plot_matrxi_scatter(df=data_all_dropped_filted, input_names=curves_list, target_col='Well_Name', plot_string='分井')


