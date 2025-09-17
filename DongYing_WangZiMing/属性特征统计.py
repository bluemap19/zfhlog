# PATH_FOLDER = r'C:\Users\ZFH\Desktop\1-15'
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from minisom import MiniSom
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src_data_process.data_distribution_statistics_overview import data_overview
from src_data_process.data_unsupervised import ClusteringPipeline, evaluate_clustering, \
    evaluate_clustering_performance_with_label
from src_file_op.dir_operation import search_files_by_criteria
from src_plot.Plot_boxplots import plot_boxes
from src_plot.plot_logging import visualize_well_logs
from src_plot.plot_matrxi_scatter import plot_matrxi_scatter


# # 设置中文支持
# def setup_chinese_support():
#     """配置Matplotlib支持中文显示"""
#     # 检查操作系统
#     if os.name == 'nt':  # Windows系统
#         plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
#     else:  # Linux/Mac系统
#         plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Heiti TC', 'STHeiti', 'SimHei']
#
#     # 解决负号显示问题
#     plt.rcParams['axes.unicode_minus'] = False
#
#     # 设置字体大小
#     plt.rcParams['font.size'] = 12
#     plt.rcParams['axes.titlesize'] = 14
#     plt.rcParams['axes.labelsize'] = 12
#     plt.rcParams['xtick.labelsize'] = 10
#     plt.rcParams['ytick.labelsize'] = 10
#     plt.rcParams['legend.fontsize'] = 10
#
#
# # 调用中文支持设置
# setup_chinese_support()

# def data_filter(df, attributes, Normed=False):
#     """
#     数据预处理函数 - 缺失值处理优化版
#     最大最小值统计
#
#     参数:
#     df: 原始数据DataFrame
#     attributes: 属性列表
#     Normed: 是否进行归一化
#
#     返回:
#     scaled_data: 标准化后的数据
#     valid_attributes: 有效属性列表
#     """
#     # 1. 选择有效属性列
#     valid_attributes = [attr for attr in attributes if attr and attr in df.columns]
#     print(f"有效属性列: {valid_attributes}")
#
#     # 2. 提取数据
#     data = df[valid_attributes].copy()
#
#     # 3. 处理缺失值
#     # 定义缺失值范围
#     missing_threshold_low = -99
#     missing_threshold_high = 9999
#
#     # 创建缺失值掩码，并把不合格数值标记为 NaN
#     missing_mask = (
#         data.isna() |
#         (data < missing_threshold_low) |
#         (data > missing_threshold_high)
#     )
#     data = data.mask(missing_mask, np.nan)
#
#     # 统计每行缺失值数量
#     row_missing_count = missing_mask.sum(axis=1)
#
#     # 删除包含缺失值的行 (超过 1% 缺失才删)
#     max_missing_per_row = len(valid_attributes) * 0.001
#     rows_to_keep = row_missing_count <= max_missing_per_row
#
#     # 在应用筛选前记录保留的索引
#     retained_indices = df.index[rows_to_keep]
#     # 应用筛选
#     data = data[rows_to_keep].copy()
#
#     # 重置索引
#     data.reset_index(drop=True, inplace=True)
#
#     # 打印删除的行数
#     n_deleted = len(df) - len(data)
#     print(f"删除包含过多缺失值的行数: {n_deleted}/{len(df)} ({n_deleted / len(df) * 100:.2f}%)")
#
#     # 4. 处理剩余缺失值：使用列中位数填充
#     for col in data.columns:
#         if data[col].isna().any():
#             median_val = data[col].median()
#             if np.isnan(median_val):  # 如果整列都是 NaN
#                 median_val = 0
#             data[col].fillna(median_val, inplace=True)
#             print(f"列 '{col}' 缺失值已填充 (中位数: {median_val:.4f})")
#
#     # 5. 保证最终结果中没有 -9999 或 NaN
#     data.replace([-9999, -9999.0], np.nan, inplace=True)
#     data.fillna(0, inplace=True)  # 再兜底一次，填充 0
#
#     # 6. 最终检查环节
#     has_neg9999 = (data <= -999).any().any()
#     has_nan = data.isna().any().any()
#
#     if not has_neg9999 and not has_nan:
#         print("✅ 数据清洗完成：不存在 -9999 或 NaN")
#     else:
#         print("⚠️ 数据清洗未完全：")
#         if has_neg9999:
#             print("   - 仍然存在 -9999")
#         if has_nan:
#             print("   - 仍然存在 NaN")
#
#     if Normed:
#         # 数据标准化
#         scaler = StandardScaler()
#         data = scaler.fit_transform(data)
#
#     return data, valid_attributes, retained_indices
def data_filter(df, attributes, Normed=False):
    """
    精简版数据预处理函数 - 更合理的缺失值处理

    参数:
    df: 原始数据DataFrame
    attributes: 属性列表
    Normed: 是否进行归一化

    返回:
    scaled_data: 标准化后的数据
    valid_attributes: 有效属性列表
    retained_indices: 保留的行索引
    """
    # 1. 选择有效属性列
    valid_attributes = [attr for attr in attributes if attr and attr in df.columns]
    print(f"有效属性列: {len(valid_attributes)}个")

    # 2. 提取数据
    data = df[valid_attributes].copy()

    # 3. 设置合理的缺失值范围（宽松范围）
    low_threshold = -999  # 非常低的阈值
    high_threshold = 9999  # 非常高的阈值

    # 4. 标记缺失值（宽松策略）
    missing_mask = (
            data.isna() |
            (data < low_threshold) |
            (data > high_threshold)
    )

    # 5. 计算每行缺失值数量
    row_missing_count = missing_mask.sum(axis=1)

    # 6. 设置合理的缺失值容忍度（允许最多20%的缺失值）
    max_missing_per_row = len(valid_attributes) * 0.20

    # 7. 保留行索引
    retained_indices = df.index[row_missing_count <= max_missing_per_row]

    # 8. 应用筛选
    data = data.loc[retained_indices].copy()

    # 9. 填充缺失值（使用中位数）
    for col in data.columns:
        median_val = data[col].median()
        data[col].fillna(median_val, inplace=True)

    # 10. 打印统计信息
    n_deleted = len(df) - len(data)
    print(f"删除行数: {n_deleted}/{len(df)} ({n_deleted / len(df) * 100:.2f}%)")

    # 11. 标准化处理
    if Normed:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    return data, valid_attributes, retained_indices




if __name__ == '__main__':
    PATH_FOLDER = r'C:\Users\ZFH\Desktop\FY1-4HF'
    # PATH_FOLDER = r'C:\Users\ZFH\Desktop\6b'

    path_list_target = search_files_by_criteria(PATH_FOLDER, name_keywords=['logging_5'],
                                                file_extensions=['.xlsx'])

    print(path_list_target)
    DF_O = pd.read_excel(path_list_target[0], sheet_name=0, engine='openpyxl')
    print(DF_O.describe())

    print(DF_O.columns)     # logging_1: '井号', '深度', 'TVD', 'GR10', '_CAL', '_SPDH', '_RD', '_RS', '_DEN', '_AC', '_CNL', '_TOC', '_POR', '_P10', '_P25', '_P75', '_P90', '_P50', '_TOC实测', '_实测S2', '_实测S1', '_GRAY', '_QF', '_CLA', '_岩相1', '_岩相7'
                            # logging_2: '井号', '深度', 'TVD', 'GR10', '_CAL', '_SPDH', '_RD', '_RS', '_DEN', '_AC', '_CNL', '_TOC', '_POR', '_P10', '_P25', '_P75', '_P90', '_P50', '_GRAY1', '_QF1', '_CLA1', '_S1实验', '_TOC实验', '_GRAY', '_QF', '_CLA', '_岩相7'

    # ATTRIBUTE_INPUT = ['GR10', '_RD', '_RS', '_DEN', '_AC', '_CNL', '_TOC', '_P10', '_P50', '_P90', '_QF', '_CLA', '_GRAY']
    # print(DF_O['矿物组分相'].value_counts())
    # print(DF_O['结构相'].value_counts())
    print(DF_O['晶相'].value_counts())
    # replace_dict = {'粘土质灰页岩':0, '富长英质混合页岩':1, '富黏土混合页岩':2, '富灰质混合页岩':3, '泥岩':4, '致密灰岩':5}
    # replace_dict = {'层状':0, '纹层状':1}
    replace_dict = {'隐晶':0, '亮晶':1}
    replace_dict_inversed = {value:key for key, value in replace_dict.items()}
    # DF_O['Type'] = DF_O['矿物组分相'].map(replace_dict)
    # DF_O['Type'] = DF_O['结构相'].map(replace_dict)
    DF_O['Type'] = DF_O['晶相'].map(replace_dict)

    # ######################### 调用可视化接口
    # visualize_well_logs(
    #     data=DF_O,
    #     depth_col='深度',
    #     curve_cols=['GR10', '_RD', '_RS', '_DEN', '_AC', '_CNL', '_QF1', '_CLA1', '_GRAY1'],
    #     # curve_cols=['温度', '测量电阻率', '温度校正因子', '去势电阻率'],
    #     type_cols=['Type'],
    #     legend_dict=replace_dict_inversed,
    #     figsize=(22, 10)
    # )

    # 1. 定义映射关系（原始列名:新列名）
    column_mapping = {
        'GR10':'伽马',
        '_RD':'深电阻率',
        '_RS':'浅电阻率',
        '_DEN':'密度',
        '_AC':'声波',
        '_CNL':'中子',
        '_POR':'孔隙度',
        '_GRAY1':'碳酸盐',
        '_QF1':'长英质',
        '_CLA1':'粘土质',
        '_P10':'P10',
        '_P25':'P25',
        '_P75':'P75',
        '_P90':'P90',
    }
    DF_O.rename(columns=column_mapping, inplace=True)

    Attribute_input = ['伽马', '深电阻率', '浅电阻率', '密度', '声波', '中子', '孔隙度', '碳酸盐', '长英质', '粘土质', 'P10', 'P25', 'P75', 'P90']
    # Attribute_input = ['深电阻率', '浅电阻率', '密度', '声波', '中子']
    # Attribute_input = ['伽马', '孔隙度', '碳酸盐', '长英质', '粘土质']
    # Attribute_input = ['伽马', 'P10', 'P25', 'P75', 'P90']
    Attribute_type = 'Type'
    DF_filter, valid_attributes, retained_indices = data_filter(DF_O, attributes=Attribute_input + [Attribute_type], Normed=False)
    DF_filter[Attribute_type] = DF_filter[Attribute_type].astype(int)       # 把Attribute_type列转换成int格式
    print(DF_filter.describe())
    print(valid_attributes)
    print(retained_indices)

    # plot_matrxi_scatter(df=DF_filter, input_names=Attribute_input, target_col=Attribute_type,
    #                     target_col_dict=replace_dict_inversed, figsize=(12, 11))

    # plot_boxes(df=DF_filter, input_names=Attribute_input, target_col = Attribute_type,
    #            target_col_dict=replace_dict_inversed, figsize=(24, 5))

    # 执行分析
    result = data_overview(
        df=DF_filter,
        input_names=Attribute_input,
        target_col=Attribute_type,
        target_col_dict=replace_dict_inversed
    )
    print(result)
    result.to_excel(PATH_FOLDER+'\\result_statics_隐晶亮晶相_all.xlsx', index=True)
    # result.to_csv(PATH_FOLDER+'\\result_statics——1.xlsx', index=False)



    # # ######################### 调用可视化接口
    # # visualize_well_logs(
    # #     data=DF_O.loc[retained_indices],
    # #     depth_col='深度',
    # #     curve_cols=['GR10', '_RD', '_RS', '_DEN', '_AC', '_CNL', '_QF', '_CLA', '_GRAY'],
    # #     # curve_cols=['温度', '测量电阻率', '温度校正因子', '去势电阻率'],
    # #     type_cols=['SOM_Cluster'],
    # #     figsize=(22, 10)
    # # )
    # #


    # df_result_save = pd.concat([data_combined_all[['DEPTH', TARGET_NAME[0]]], df_type_result_new], axis=1)
    # print(df_result_save.describe())
    # # df_result_save.to_csv('target_cluster_result.csv', index=False, encoding='utf-8')
    # result_kmeans = table_2_to_3(df_result_save[['DEPTH', 'KMeans']].values)
    # df_kmeans = pd.DataFrame(result_kmeans, columns=['DEPTH_START', 'DEPTH_END', 'KMeans'])
    # df_kmeans.to_csv('df_kmeans.csv', index=True)
    # result_hierarchical = table_2_to_3(df_result_save[['DEPTH', 'Hierarchical']].values)
    # df_hierarchical = pd.DataFrame(result_hierarchical, columns=['DEPTH_START', 'DEPTH_END', 'Hierarchical'])
    # df_hierarchical.to_csv('df_hierarchical.csv', index=True)
    # result_spectral = table_2_to_3(df_result_save[['DEPTH', 'Spectral']].values)
    # df_spectral = pd.DataFrame(result_spectral, columns=['DEPTH_START', 'DEPTH_END', 'Spectral'])
    # df_spectral.to_csv('df_spectral.csv', index=True)
    # result_gmm = table_2_to_3(df_result_save[['DEPTH', 'GMM']].values)
    # df_gmm = pd.DataFrame(result_gmm, columns=['DEPTH_START', 'DEPTH_END', 'GMM'])
    # df_gmm.to_csv('df_gmm.csv', index=True)
    # result_gmm = table_2_to_3(df_result_save[['DEPTH', TARGET_NAME[0]]].values)
    # df_org = pd.DataFrame(result_gmm, columns=['DEPTH_START', 'DEPTH_END', 'LITHO'])
    # df_org.to_csv('df_org.csv', index=True)

