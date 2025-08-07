import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src_data_process.data_unsupervised import ClusteringPipeline, evaluate_clustering, \
    evaluate_clustering_performance_with_label
from src_file_op.dir_operation import search_files_by_criteria
from src_plot.plot_3D_scatter import interactive_3d_pca
from src_plot.plot_correlation import plot_correlation_analyze
from src_plot.plot_matrxi_scatter import plot_matrxi_scatter
from src_table.table_process import table_2_to_3
from src_well_project.LOGGING_PROJECT import LOGGING_PROJECT


if __name__ == '__main__':
    # LG = LOGGING_PROJECT(project_path=r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV')
    # LG = LOGGING_PROJECT(project_path=r'E:\桌面\算法测试-长庆数据收集\logging_CSV')
    LG = LOGGING_PROJECT(project_path=r'C:\Users\ZFH\Documents\simu-result')
    table = LG.get_table_3_all_data(['simu1'])
    # print(table)

    # print(LG.get_all_table_replace_dict(well_names=['simu1'], file_path={'simu1':path_table}))
    # # dict = {'中GR长英黏土质': 0, '中GR长英黏土质（泥岩）': 0, '中低GR长英质': 1, '中低GR长英质（砂岩）': 1,
    # #         '富有机质长英质': 2, '富有机质长英质页岩': 2, '富有机质黏土质': 3, '富有机质黏土质页岩': 3,
    # #         '高GR富凝灰长英质': 4, '高GR富凝灰长英质（沉凝灰岩）': 4}

    path_logging_target = LG.search_target_file_path(well_name='simu1', target_path_feature=['Texture_ALL', f'_80_5'], target_file_type='logging')
    path_table_target = LG.search_target_file_path(well_name='simu1', target_path_feature=['LITHO_TYPE_2'], target_file_type='table')
    print('get data from path:{},\nget table from:{}'.format(path_logging_target, path_table_target))

    # replace_dict = {'高密度层理': 0, '低密度层理': 1, '孔洞': 2, '高密度低角度层理': 0, '高密度高角度层理': 1, '低密度低角度层理': 2, '低密度高角度层理': 3, '高阻块状': 4, '低阻块状':5}
    replace_dict = {'低阻多斑块': 0, '低阻少斑块': 1, '高阻多斑块': 2, '高阻少斑块': 3, '高阻层理': 4, '高阻纹理': 5, '低阻层理': 6, '低阻纹理': 7}
    COL_NAMES_ALL = [
        'STAT_CON', 'STAT_DIS', 'STAT_HOM', 'STAT_ENG', 'STAT_COR', 'STAT_ASM', 'STAT_ENT', 'STAT_XY_CON',
        'STAT_XY_DIS', 'STAT_XY_HOM', 'STAT_XY_ENG', 'STAT_XY_COR', 'STAT_XY_ASM', 'STAT_XY_ENT',
        'DYNA_CON', 'DYNA_DIS', 'DYNA_HOM', 'DYNA_ENG', 'DYNA_COR', 'DYNA_ASM', 'DYNA_ENT', 'DYNA_XY_CON',
        'DYNA_XY_DIS', 'DYNA_XY_HOM', 'DYNA_XY_ENG', 'DYNA_XY_COR', 'DYNA_XY_ASM', 'DYNA_XY_ENT'
    ]
    COL_NAMES = [
        'DYNA_COR', 'DYNA_XY_DIS', 'STAT_XY_ENG', 'DYNA_XY_COR', 'DYNA_XY_CON', 'DYNA_CON',
        'DYNA_ENG', 'DYNA_XY_HOM', 'DYNA_DIS', 'STAT_XY_ASM', 'STAT_XY_ENT', 'DYNA_XY_ENT'

        # 'STAT_CON', 'STAT_DIS', 'STAT_HOM', 'STAT_ENG', 'STAT_COR', 'STAT_ASM', 'STAT_ENT', 'STAT_XY_CON',
        # 'STAT_XY_DIS', 'STAT_XY_HOM', 'STAT_XY_ENG', 'STAT_XY_COR', 'STAT_XY_ASM', 'STAT_XY_ENT',
        # 'DYNA_CON', 'DYNA_DIS', 'DYNA_HOM', 'DYNA_ENG', 'DYNA_COR', 'DYNA_ASM', 'DYNA_ENT', 'DYNA_XY_CON',
        # 'DYNA_XY_DIS', 'DYNA_XY_HOM', 'DYNA_XY_ENG', 'DYNA_XY_COR', 'DYNA_XY_ASM', 'DYNA_XY_ENT'
    ]
    TARGET_NAME = ['Type']

    data_combined_all = LG.combined_all_logging_with_type(well_names=['simu1'],
                                                   file_path_logging={'simu1': path_logging_target},
                                                   file_path_table={'simu1': path_table_target},
                                                   replace_dict=replace_dict, type_new_col=TARGET_NAME[0],
                                                   curve_names_logging=COL_NAMES,
                                                   Norm=True)
    print(data_combined_all.describe())

    # # 这里是整体上看一下ACC在 窗长-随机森林参数 上的分布特征
    # target_col_dict = {0:'中GR长英黏土质', 1:'中低GR长英质', 2:'富有机质长英质页岩', 3:'富有机质黏土质页岩', 4:'高GR富凝灰长英质'}
    # target_col_dict = {0:'块状构造泥岩', 1:'薄层砂岩', 2:'富有机质粉砂级长英质页岩', 3:'富有机质富凝灰质页岩', 4:'沉凝灰岩'}
    # target_col_dict = {'高密度低角度层理': 0, '高密度高角度层理': 1, '低密度低角度层理': 2, '低密度高角度层理': 3, '高阻块状': 4, '低阻块状':5}
    target_col_dict = {'低阻多斑块': 0, '低阻少斑块': 1, '高阻多斑块': 2, '高阻少斑块': 3, '高阻层理': 4, '高阻纹理': 5, '低阻层理': 6, '低阻纹理': 7}

    # # 调用接口进行相关性分析，主要功能是，可视化输入属性之间的相关系数R²
    # plot_correlation_analyze(data_combined_all, COL_NAMES,
    #                          method='pearson', figsize=(14, 14),
    #                          return_matrix=True)

    # # 可视化输入属性对类别的散布图，分析输入属性对类别的影响力
    # print(data_combined_all[COL_NAMES].describe())
    # plot_matrxi_scatter(df=data_combined_all, input_names=COL_NAMES, target_col=TARGET_NAME[0],
    #                     plot_string='输入属性分布',
    #                     target_col_dict = target_col_dict
    #                     )

    # svd_solver: 使用的SVD求解器类型。可以选择 "auto"、"full"、"arpack"和"randomized"
    # iterated_power: 幂迭代方法的迭代次数。默认值为 "auto"。
    # random_state: 随机数生成器的种子。
    pca = PCA(n_components=0.99)
    pca_result = pca.fit_transform(data_combined_all[COL_NAMES])
    # print(type(pca_result), pca_result.shape, data_combined_all.shape)
    # PCA_RESULT_DF = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2', 'PCA3'])
    # PCA_RESULT_DF = pd.concat([data_combined_all, PCA_RESULT_DF], axis=1)
    # # PCA_RESULT_DF.to_csv(r'C:\Users\ZFH\Documents\simu-result\PCA_RESULT_DF.csv', index=False)
    #
    # # replace_dict_reversed = {value:key for key, value in replace_dict.items()}
    # # print(replace_dict_reversed)
    # # PCA_RESULT_DF['Type_R'] = PCA_RESULT_DF['Type'].map(replace_dict_reversed)
    # # # 使用交互式接口
    # # interactive_fig = interactive_3d_pca(
    # #     PCA_RESULT_DF[['PCA1', 'PCA2', 'PCA3', 'Type_R']],
    # #     title="Texture PCA Visualization",
    # #     XYZ=['PCA1', 'PCA2', 'PCA3'],
    # #     Type='Type_R',
    # #     point_size=3,
    # #     alpha=0.9,
    # #     grid_width=3
    # # )
    # # interactive_fig.show()

    # 进行数据的无监督聚类
    # 2. 初始化接口
    Unsupervised_Pipeline = ClusteringPipeline(cluster_num=9, scale_data=True)
    # 3. 模型训练
    Unsupervised_Pipeline.fit(data_combined_all[COL_NAMES])
    # 4. 获取结果
    results = Unsupervised_Pipeline.get_results()
    print('无监督聚类结果：')
    print(results.describe())

    # # pred_result = Unsupervised_Pipeline.predict(data_combined_all[COL_NAMES],
    # #                                             algorithm=['KMeans', 'DBSCAN', 'Hierarchical', 'Spectral', 'GMM', ])
    # # print(pred_result.describe())

    # 无监督聚类的评价标准Silhouette、CH、DBI、DVI计算
    unsupervised_evluate_result = evaluate_clustering(data_combined_all[COL_NAMES], results[['KMeans', 'Hierarchical', 'Spectral', 'GMM']])
    print(unsupervised_evluate_result)

    df_result_all = pd.concat([data_combined_all[TARGET_NAME], results[['KMeans', 'Hierarchical', 'Spectral', 'GMM']]], axis=1)
    df_type_result_new, acc_df, acc_num_df, replace_dict_dict = evaluate_clustering_performance_with_label(
        df_result_all, true_col=TARGET_NAME[0])

    df_result_save = pd.concat([data_combined_all[['DEPTH', TARGET_NAME[0]]], df_type_result_new], axis=1)
    print(df_result_save.describe())
    # df_result_save.to_csv('target_cluster_result.csv', index=False, encoding='utf-8')
    result_kmeans = table_2_to_3(df_result_save[['DEPTH', 'KMeans']].values)
    df_kmeans = pd.DataFrame(result_kmeans, columns=['DEPTH_START', 'DEPTH_END', 'KMeans'])
    df_kmeans.to_csv('df_kmeans.csv', index=True)
    result_hierarchical = table_2_to_3(df_result_save[['DEPTH', 'Hierarchical']].values)
    df_hierarchical = pd.DataFrame(result_hierarchical, columns=['DEPTH_START', 'DEPTH_END', 'Hierarchical'])
    df_hierarchical.to_csv('df_hierarchical.csv', index=True)
    result_spectral = table_2_to_3(df_result_save[['DEPTH', 'Spectral']].values)
    df_spectral = pd.DataFrame(result_spectral, columns=['DEPTH_START', 'DEPTH_END', 'Spectral'])
    df_spectral.to_csv('df_spectral.csv', index=True)
    result_gmm = table_2_to_3(df_result_save[['DEPTH', 'GMM']].values)
    df_gmm = pd.DataFrame(result_gmm, columns=['DEPTH_START', 'DEPTH_END', 'GMM'])
    df_gmm.to_csv('df_gmm.csv', index=True)
    result_gmm = table_2_to_3(df_result_save[['DEPTH', TARGET_NAME[0]]].values)
    df_org = pd.DataFrame(result_gmm, columns=['DEPTH_START', 'DEPTH_END', 'LITHO'])
    df_org.to_csv('df_org.csv', index=True)

    # # print(df_type_result_new.describe())
    # # print(acc_df)
    # # print(acc_num_df)
    # # print(replace_dict_dict)
