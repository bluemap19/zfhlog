import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src_data_process.data_dilute import dilute_dataframe
from src_data_process.data_unsupervised import ClusteringPipeline, evaluate_clustering, \
    evaluate_clustering_performance_with_label
from src_file_op.dir_operation import search_files_by_criteria
from src_plot.plot_3D_scatter import interactive_3d_pca
from src_plot.plot_correlation import plot_correlation_analyze
from src_plot.plot_falls import calculate_frequency_distribution, save_distribution_to_excel
from src_plot.plot_matrxi_scatter import plot_matrxi_scatter
from src_table.table_process import table_2_to_3
from src_well_project.LOGGING_PROJECT import LOGGING_PROJECT


if __name__ == '__main__':
    # LG = LOGGING_PROJECT(project_path=r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV')
    # LG = LOGGING_PROJECT(project_path=r'E:\桌面\算法测试-长庆数据收集\logging_CSV')
    LG = LOGGING_PROJECT(project_path=r'F:\桌面\算法测试-长庆数据收集\logging_CSV')
    table = LG.get_table_3_all_data(['SIMU4'])

    path_logging_target = LG.search_target_file_path(well_name='SIMU4', target_path_feature=['logging_data'], target_file_type='logging')
    path_table_target = LG.search_target_file_path(well_name='SIMU4', target_path_feature=['LITHO_TYPE'], target_file_type='table')
    print('get data from path:{},\nget table from:{}'.format(path_logging_target, path_table_target))

    replace_dict = LG.get_all_table_replace_dict(well_names=['SIMU4'], file_path={'SIMU4':path_table_target})
    replace_dict_inversed = {value:key for key, value in replace_dict.items()}
    print('replace table set as:{}, and set inversed replace table as:{}'.format(replace_dict, replace_dict_inversed))
    COL_NAMES_ALL = [
        'STAT_CON', 'STAT_DIS', 'STAT_HOM', 'STAT_ENG', 'STAT_COR', 'STAT_ASM', 'STAT_ENT', 'STAT_XY_CON',
        'STAT_XY_DIS', 'STAT_XY_HOM', 'STAT_XY_ENG', 'STAT_XY_COR', 'STAT_XY_ASM', 'STAT_XY_ENT',
        'DYNA_CON', 'DYNA_DIS', 'DYNA_HOM', 'DYNA_ENG', 'DYNA_COR', 'DYNA_ASM', 'DYNA_ENT', 'DYNA_XY_CON',
        'DYNA_XY_DIS', 'DYNA_XY_HOM', 'DYNA_XY_ENG', 'DYNA_XY_COR', 'DYNA_XY_ASM', 'DYNA_XY_ENT'
    ]
    COL_NAMES = [
        'DYNA_XY_HOM', 'DYNA_XY_COR', 'DYNA_XY_DIS', 'DYNA_COR', 'DYNA_XY_ENT', 'STAT_CON', 'DYNA_CON', 'DYNA_HOM'
        # 'STAT_CON', 'STAT_DIS', 'STAT_HOM', 'STAT_ENG', 'STAT_COR', 'STAT_ASM', 'STAT_ENT', 'STAT_XY_CON',
        # 'STAT_XY_DIS', 'STAT_XY_HOM', 'STAT_XY_ENG', 'STAT_XY_COR', 'STAT_XY_ASM', 'STAT_XY_ENT',
        # 'DYNA_CON', 'DYNA_DIS', 'DYNA_HOM', 'DYNA_ENG', 'DYNA_COR', 'DYNA_ASM', 'DYNA_ENT', 'DYNA_XY_CON',
        # 'DYNA_XY_DIS', 'DYNA_XY_HOM', 'DYNA_XY_ENG', 'DYNA_XY_COR', 'DYNA_XY_ASM', 'DYNA_XY_ENT'
    ]
    TARGET_NAME = ['Type']

    data_combined_all = LG.combined_all_logging_with_type(well_names=['SIMU4'],
                                                   file_path_logging={'SIMU4': path_logging_target},
                                                   file_path_table={'SIMU4': path_table_target},
                                                   replace_dict=replace_dict, type_new_col=TARGET_NAME[0],
                                                   curve_names_logging=COL_NAMES,
                                                   Norm=False)
    print(data_combined_all.describe())

    # replace_dict_inversed = {0: 'E', 1: 'D', 2: 'C', 3: 'B', 4: 'A'}
    # data_combined_all['Type_C'] = data_combined_all['Type'].map(replace_dict_inversed)
    # data_combined_all.to_excel(r'F:\桌面\算法测试-长庆数据收集\logging_CSV\SIMU4\texture_choiced_with_type.xlsx', index=False)
    # # 计算频数分布
    # distribution_results = calculate_frequency_distribution(
    #     df=data_combined_all,
    #     value_columns=COL_NAMES,
    #     type_column=TARGET_NAME[0],
    #     bins=32,
    #     density=False
    # )
    # # 保存所有结果到Excel
    # save_distribution_to_excel(distribution_results, "frequency_distribution.xlsx")
    # print("\n频数分布数据已保存到 'frequency_distribution.xlsx'")
    # exit(0)

    data_diluted_grouped = dilute_dataframe(data_combined_all, ratio=20, method='random', group_by='Type')

    # # 调用接口进行相关性分析，主要功能是，可视化输入属性之间的相关系数R²
    # plot_correlation_analyze(data_combined_all, COL_NAMES,
    #                          method='pearson', figsize=(14, 14),
    #                          return_matrix=True)
    #
    # # 可视化输入属性对类别的散布图，分析输入属性对类别的影响力
    # plot_matrxi_scatter(df=data_diluted_grouped, input_names=COL_NAMES, target_col=TARGET_NAME[0],
    #                     target_col_dict = replace_dict_inversed, figsize=(14, 14),)

    # svd_solver: 使用的SVD求解器类型。可以选择 "auto"、"full"、"arpack"和"randomized"
    # iterated_power: 幂迭代方法的迭代次数。默认值为 "auto"。
    # random_state: 随机数生成器的种子。
    pca = PCA(n_components=0.95)
    pca_result = pca.fit_transform(data_combined_all[COL_NAMES])
    print('PCA data from {} to {} as type {}'.format(data_combined_all.shape, pca_result.shape, type(pca_result)))
    PCA_COLS = [f'PCA{i+1}' for i in range(pca_result.shape[1])]
    PCA_RESULT_DF = pd.DataFrame(pca_result, columns=PCA_COLS)
    PCA_RESULT_DF = pd.concat([data_combined_all, PCA_RESULT_DF], axis=1)
    # PCA_RESULT_DF.to_csv(r'C:\Users\ZFH\Documents\simu-result\PCA_RESULT_DF.csv', index=False)

    # PCA_RESULT_DF['Type_R'] = PCA_RESULT_DF['Type'].map(replace_dict_inversed)
    # # 使用交互式接口
    # interactive_fig = interactive_3d_pca(
    #     PCA_RESULT_DF[['PCA1', 'PCA2', 'PCA3', 'Type_R']],  # 原始数据体
    #     title="Texture PCA Visualization",
    #     XYZ=['PCA1', 'PCA2', 'PCA3'],   # XYZ绘制列配置
    #     Type='Type_R',          # legend 分类信息
    #     point_size=3,           # 点的大小
    #     alpha=0.9,              # 点透明度
    #     grid_width=5,           # 网格线 线宽
    # )
    # data_combined_all = pd.concat(
    #     [data_combined_all, PCA_RESULT_DF], axis=1)
    # # df_pca.to_csv('pca_result.csv', index=False, encoding='utf-8')
    # print(PCA_COLS, '\n', PCA_RESULT_DF.describe())
    # interactive_fig.show()

    # 进行数据的无监督聚类
    # 2. 初始化接口
    Unsupervised_Pipeline = ClusteringPipeline(cluster_num=8, scale_data=True)
    # 3. 模型训练
    Unsupervised_Pipeline.fit(PCA_RESULT_DF[PCA_COLS])
    # # 4. 获取结果
    # results = Unsupervised_Pipeline.get_results()
    # print('无监督聚类结果：')
    # print(results.describe())

    # # 4. 使用无监督聚类进行预测
    results = Unsupervised_Pipeline.predict(PCA_RESULT_DF[PCA_COLS],
                                                algorithm=['KMeans', 'DBSCAN', 'Hierarchical', 'Spectral', 'GMM', 'SOM'])
    print(results.describe())

    # 无监督聚类的评价标准Silhouette、CH、DBI、DVI计算
    unsupervised_evluate_result = evaluate_clustering(PCA_RESULT_DF[PCA_COLS], results[['KMeans', 'Hierarchical', 'Spectral', 'GMM', 'SOM']])
    print(unsupervised_evluate_result)

    df_result_all = pd.concat([data_combined_all[TARGET_NAME], results[['KMeans', 'Hierarchical', 'Spectral', 'GMM', 'SOM']]], axis=1)

    df_type_result_new, acc_df, acc_num_df, replace_dict_dict = evaluate_clustering_performance_with_label(df_result_all, true_col=TARGET_NAME[0])
    print(df_type_result_new.describe())
    print(acc_df)
    print(acc_num_df)
    print(replace_dict_dict)

    df_result_save = pd.concat([data_combined_all[['#DEPTH', TARGET_NAME[0]]], df_type_result_new], axis=1)
    print(df_result_save.describe())

    # {0: '块状构造泥岩', 1: '富有机质富凝灰质页岩', 2: '富有机质粉砂级长英质页岩', 3: '沉凝灰岩', 4: '薄夹层砂岩'}
    replace_dict_inversed = {0: 'E', 1: 'D', 2: 'C', 3: 'B', 4: 'A'}
    # df_result_save.to_csv('target_cluster_result.csv', index=False, encoding='utf-8')
    result_kmeans = table_2_to_3(df_result_save[['#DEPTH', 'KMeans']].values)
    df_kmeans = pd.DataFrame(result_kmeans, columns=['DEPTH_START', 'DEPTH_END', 'KMeans'])
    df_kmeans['KMeans_C'] = df_kmeans['KMeans'].map(replace_dict_inversed)
    df_kmeans.to_csv('df_kmeans.csv', index=True)

    result_hierarchical = table_2_to_3(df_result_save[['#DEPTH', 'Hierarchical']].values)
    df_hierarchical = pd.DataFrame(result_hierarchical, columns=['DEPTH_START', 'DEPTH_END', 'Hierarchical'])
    df_hierarchical['Hierarchical_C'] = df_hierarchical['Hierarchical'].map(replace_dict_inversed)
    df_hierarchical.to_csv('df_hierarchical.csv', index=True)

    result_spectral = table_2_to_3(df_result_save[['#DEPTH', 'Spectral']].values)
    df_spectral = pd.DataFrame(result_spectral, columns=['DEPTH_START', 'DEPTH_END', 'Spectral'])
    df_spectral['Spectral_C'] = df_spectral['Spectral'].map(replace_dict_inversed)
    df_spectral.to_csv('df_spectral.csv', index=True)

    result_gmm = table_2_to_3(df_result_save[['#DEPTH', 'GMM']].values)
    df_gmm = pd.DataFrame(result_gmm, columns=['DEPTH_START', 'DEPTH_END', 'GMM'])
    df_gmm['GMM_C'] = df_gmm['GMM'].map(replace_dict_inversed)
    df_gmm.to_csv('df_gmm.csv', index=True)

    result_org = table_2_to_3(df_result_save[['#DEPTH', TARGET_NAME[0]]].values)
    df_org = pd.DataFrame(result_org, columns=['DEPTH_START', 'DEPTH_END', 'LITHO'])
    df_org['LITHO_C'] = df_org['LITHO'].map(replace_dict_inversed)
    df_org.to_csv('df_org.csv', index=True)

    result_som = table_2_to_3(df_result_save[['#DEPTH', 'SOM']].values)
    df_som = pd.DataFrame(result_som, columns=['DEPTH_START', 'DEPTH_END', 'SOM'])
    df_som['SOM_C'] = df_som['SOM'].map(replace_dict_inversed)
    df_som.to_csv('df_som.csv', index=True)