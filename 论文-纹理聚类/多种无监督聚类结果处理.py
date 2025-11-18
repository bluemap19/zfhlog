import logging

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
from src_plot.TEMP_4 import WellLogVisualizer
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
    print(LG.WELL_NAMES)


    path_logging_target = LG.search_target_file_path(well_name='FY1-15', target_path_feature=['logging_data', 'texture'], target_file_type='logging')
    path_table_target = LG.search_target_file_path(well_name='FY1-15', target_path_feature=['LITHO_TYPE'], target_file_type='table')
    print('get data from path:{},\nget table from:{}'.format(path_logging_target, path_table_target))

    table = LG.get_table_3_all_data(['FY1-15'])
    print(table.describe())
    replace_dict = LG.get_all_table_replace_dict(well_names=['FY1-15'], file_path={'FY1-15':path_table_target})
    print(replace_dict)

    replace_dict_inversed = {value:key for key, value in replace_dict.items()}
    replace_dict_inversed_ciflog = {0: 'B', 1: 'C', 2: 'D', 3: 'A', 4: 'E'}
    print('replace table set as:{}, and set inversed replace table as:{}, \nand the ciflog inversed dict is :{}'.format(replace_dict, replace_dict_inversed, replace_dict_inversed_ciflog))

    # COL_NAMES_ALL = [
    #     'CON_MEAN_DYNA', 'DIS_MEAN_DYNA', 'HOM_MEAN_DYNA', 'ENG_MEAN_DYNA', 'COR_MEAN_DYNA', 'ASM_MEAN_DYNA',
    #     'ENT_MEAN_DYNA', 'CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA', 'ENG_SUB_DYNA', 'COR_SUB_DYNA', 'ASM_SUB_DYNA',
    #     'ENT_SUB_DYNA', 'CON_X_DYNA', 'DIS_X_DYNA', 'HOM_X_DYNA', 'ENG_X_DYNA', 'COR_X_DYNA', 'ASM_X_DYNA',
    #     'ENT_X_DYNA', 'CON_Y_DYNA', 'DIS_Y_DYNA', 'HOM_Y_DYNA', 'ENG_Y_DYNA', 'COR_Y_DYNA', 'ASM_Y_DYNA', 'ENT_Y_DYNA',
    #     'CON_MEAN_STAT', 'DIS_MEAN_STAT', 'HOM_MEAN_STAT', 'ENG_MEAN_STAT', 'COR_MEAN_STAT', 'ASM_MEAN_STAT',
    #     'ENT_MEAN_STAT', 'CON_SUB_STAT', 'DIS_SUB_STAT', 'HOM_SUB_STAT', 'ENG_SUB_STAT', 'COR_SUB_STAT', 'ASM_SUB_STAT',
    #     'ENT_SUB_STAT', 'CON_X_STAT', 'DIS_X_STAT', 'HOM_X_STAT', 'ENG_X_STAT', 'COR_X_STAT', 'ASM_X_STAT',
    #     'ENT_X_STAT', 'CON_Y_STAT', 'DIS_Y_STAT', 'HOM_Y_STAT', 'ENG_Y_STAT', 'COR_Y_STAT', 'ASM_Y_STAT', 'ENT_Y_STAT'
    # ]

    COL_NAMES = [
        'HOM_SUB_DYNA', 'COR_SUB_DYNA', 'DIS_SUB_DYNA', 'COR_MEAN_DYNA', 'ENT_SUB_DYNA', 'CON_MEAN_STAT', 'CON_MEAN_DYNA', 'HOM_MEAN_DYNA'
    ]

    TARGET_NAME = ['Type']

    data_combined_all = LG.combined_all_logging_with_type(well_names=['FY1-15'],
                                                   file_path_logging={'FY1-15': path_logging_target},
                                                   file_path_table={'FY1-15': path_table_target},
                                                   replace_dict=replace_dict, type_new_col=TARGET_NAME[0],
                                                   curve_names_logging=COL_NAMES,
                                                   Norm=True,
    )

    print(data_combined_all.describe())

    data_combined_all['Type_Ciflog'] = data_combined_all['Type'].map(replace_dict_inversed_ciflog)
    data_combined_all['Type_Origin'] = data_combined_all['Type'].map(replace_dict_inversed)

    # data_combined_all.to_excel(r'F:\桌面\算法测试-长庆数据收集\logging_CSV\FY1-15\texture_choiced_with_type.xlsx', index=False)
    # # 计算频数分布
    # distribution_results = calculate_frequency_distribution(
    #     df=data_combined_all,
    #     value_columns=COL_NAMES,
    #     type_column=TARGET_NAME[0],
    #     bins=32,
    #     density=False
    # )
    #
    # # 保存所有结果到Excel
    # save_distribution_to_excel(distribution_results, r"F:\桌面\算法测试-长庆数据收集\logging_CSV\FY1-15\frequency_distribution.xlsx")
    # print("\n频数分布数据已保存到 'frequency_distribution.xlsx'")

    # data_diluted_grouped = dilute_dataframe(data_combined_all, ratio=20, method='random', group_by='Type')
    #
    # # # 调用接口进行相关性分析，主要功能是，可视化输入属性之间的相关系数R²
    # # plot_correlation_analyze(data_combined_all, COL_NAMES,
    # #                          method='pearson', figsize=(14, 14),
    # #                          return_matrix=True)
    # #
    # # # 可视化输入属性对类别的散布图，分析输入属性对类别的影响力
    # # plot_matrxi_scatter(df=data_diluted_grouped, input_names=COL_NAMES, target_col=TARGET_NAME[0],
    # #                     target_col_dict = replace_dict_inversed, figsize=(14, 14),)

    # 通过深度截取部分数据，否则数据太大了，无法进行计算
    index_depth = (data_combined_all['DEPTH'] > 3720) & (data_combined_all['DEPTH'] < 3740)
    data_combined_all = data_combined_all[index_depth]
    data_combined_all.reset_index(drop=True, inplace=True)


    # svd_solver: 使用的SVD求解器类型。可以选择 "auto"、"full"、"arpack"和"randomized"
    # iterated_power: 幂迭代方法的迭代次数。默认值为 "auto"。
    # random_state: 随机数生成器的种子。
    pca = PCA(n_components=0.95)
    pca_result = pca.fit_transform(data_combined_all[COL_NAMES])
    print('PCA data from {} to {} as type {}'.format(data_combined_all.shape, pca_result.shape, type(pca_result)))
    PCA_COLS = [f'PCA{i+1}' for i in range(pca_result.shape[1])]
    PCA_RESULT_DF = pd.DataFrame(pca_result, columns=PCA_COLS)
    data_combined_all = pd.concat([data_combined_all, PCA_RESULT_DF], axis=1)
    data_combined_all.to_csv(r'F:\桌面\算法测试-长庆数据收集\logging_CSV\FY1-15\PCA_RESULT_DF.csv', index=False)


    # # 使用交互式接口
    # interactive_fig = interactive_3d_pca(
    #     data_combined_all[['PCA1', 'PCA2', 'PCA3', 'Type_Origin']],  # 原始数据体
    #     title="Texture PCA Visualization",
    #     XYZ=['PCA1', 'PCA2', 'PCA3'],   # XYZ绘制列配置
    #     Type='Type_Origin',          # legend 分类信息
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
    results = Unsupervised_Pipeline.predict(PCA_RESULT_DF[PCA_COLS], algorithm=['KMeans', 'DBSCAN', 'Hierarchical', 'Spectral', 'GMM', 'SOM'])
    df_result_all = pd.concat([data_combined_all[TARGET_NAME], results[['KMeans', 'Hierarchical', 'Spectral', 'GMM', 'SOM', 'DBSCAN']]], axis=1)
    # df_result_all.to_csv('temp_csv.csv', index=True)
    print(results.describe())

    # # 无监督聚类的评价标准Silhouette、CH、DBI、DVI计算
    # unsupervised_evluate_result = evaluate_clustering(PCA_RESULT_DF[PCA_COLS], results[['KMeans', 'Hierarchical', 'Spectral', 'GMM', 'SOM']])
    # print(unsupervised_evluate_result)

    data_combined_all = pd.concat([data_combined_all, results], axis=1)
    print(data_combined_all.columns)

    # # 使用类接口进行可视化
    # print("创建可视化器...")
    # visualizer = WellLogVisualizer()
    # try:
    #     # 启用详细日志
    #     logging.getLogger().setLevel(logging.INFO)
    #
    #     visualizer.visualize(
    #         data=data_combined_all,
    #         depth_col='DEPTH',
    #         # curve_cols=['HOM_SUB_DYNA', 'COR_SUB_DYNA', 'DIS_SUB_DYNA', 'COR_MEAN_DYNA', 'ENT_SUB_DYNA', 'CON_MEAN_STAT', 'CON_MEAN_DYNA', 'HOM_MEAN_DYNA', 'PCA1', 'PCA2', 'PCA3', 'PCA4'],
    #         curve_cols=['PCA1', 'PCA2', 'PCA3', 'PCA4'],
    #         type_cols=['Type', 'KMeans', 'DBSCAN', 'Hierarchical', 'Spectral', 'GMM', 'SOM'],
    #         legend_dict=replace_dict_inversed_ciflog,
    #         # fmi_dict={
    #         #     'depth': fmi_data_dict['DEPTH_DYNA'],
    #         #     'image_data': [fmi_data_dict['DYNA'], fmi_data_dict['STAT']],
    #         #     'title': ['FMI动态', 'FMI静态']
    #         # },
    #         # fmi_dict=None,
    #         # depth_limit_config=[320, 380],  # 只显示320-380米段
    #         figsize=(12, 8)
    #     )
    #
    #     # 显示性能统计
    #     stats = visualizer.get_performance_stats()
    #     print("性能统计:", stats)
    #
    # except Exception as e:
    #     print(f"可视化过程中出现错误: {e}")
    #     import traceback
    #
    #     traceback.print_exc()
    # finally:
    #     # 清理资源
    #     visualizer.close()


    df_type_result_new, acc_df, acc_num_df, replace_dict_dict = evaluate_clustering_performance_with_label(df_result_all, true_col=TARGET_NAME[0])
    print(df_type_result_new.describe())
    print(df_type_result_new.columns)
    print(acc_df)
    print(acc_num_df)
    print(replace_dict_dict)
    # exit(0)

    df_result_save = pd.concat([data_combined_all[['DEPTH', TARGET_NAME[0]]], df_type_result_new], axis=1)
    print(df_result_save.describe())

    df_type_result_new_visulize = df_type_result_new.rename(columns = {'KMeans':'KMeans_N', 'Hierarchical':'Hierarchical_N', 'Spectral':'Spectral_N', 'GMM':'GMM_N', 'SOM':'SOM_N', 'DBSCAN':'DBSCAN_N'})
    print(df_type_result_new_visulize.columns)
    data_combined_all = pd.concat([data_combined_all, df_type_result_new_visulize], axis=1)
    print(data_combined_all.columns)

    # 使用类接口进行可视化
    print("创建可视化器...")
    visualizer = WellLogVisualizer()
    try:
        # 启用详细日志
        logging.getLogger().setLevel(logging.INFO)

        visualizer.visualize(
            data=data_combined_all,
            depth_col='DEPTH',
            # curve_cols=['HOM_SUB_DYNA', 'COR_SUB_DYNA', 'DIS_SUB_DYNA', 'COR_MEAN_DYNA', 'ENT_SUB_DYNA', 'CON_MEAN_STAT', 'CON_MEAN_DYNA', 'HOM_MEAN_DYNA', 'PCA1', 'PCA2', 'PCA3', 'PCA4'],
            curve_cols=['PCA1', 'PCA2', 'PCA3', 'PCA4'],
            # type_cols=['Type', 'KMeans', 'KMeans_N', 'Hierarchical', 'Hierarchical_N', 'Spectral', 'Spectral_N', 'GMM', 'GMM_N', 'SOM', 'SOM_N', 'DBSCAN', 'DBSCAN_N'],
            type_cols=['Type', 'KMeans_N', 'Hierarchical_N', 'Spectral_N', 'GMM_N', 'SOM_N', 'DBSCAN_N'],
            legend_dict=replace_dict_inversed_ciflog,
            figsize=(18, 12)
        )

        # 显示性能统计
        stats = visualizer.get_performance_stats()
        print("性能统计:", stats)

    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 清理资源
        visualizer.close()

    result_origin = table_2_to_3(df_result_save[['DEPTH', TARGET_NAME[0]]].values)
    df_origin = pd.DataFrame(result_origin, columns=['DEPTH_START', 'DEPTH_END', TARGET_NAME[0]])
    df_origin['Type_Ciflog'] = df_origin[TARGET_NAME[0]].map(replace_dict_inversed_ciflog)
    df_origin['Type_Origin'] = df_origin[TARGET_NAME[0]].map(replace_dict_inversed)
    df_origin.to_csv('df_type_origin.csv', index=True)

    # {0: '块状构造泥岩', 1: '富有机质富凝灰质页岩', 2: '富有机质粉砂级长英质页岩', 3: '沉凝灰岩', 4: '薄夹层砂岩'}
    # df_result_save.to_csv('target_cluster_result.csv', index=False, encoding='utf-8')
    result_kmeans = table_2_to_3(df_result_save[['DEPTH', 'KMeans']].values)
    df_kmeans = pd.DataFrame(result_kmeans, columns=['DEPTH_START', 'DEPTH_END', 'KMeans'])
    df_kmeans['KMeans_Ciflog'] = df_kmeans['KMeans'].map(replace_dict_inversed_ciflog)
    df_kmeans['KMeans_Origin'] = df_kmeans['KMeans'].map(replace_dict_inversed)
    df_kmeans.to_csv('df_kmeans.csv', index=True)

    result_hierarchical = table_2_to_3(df_result_save[['DEPTH', 'Hierarchical']].values)
    df_hierarchical = pd.DataFrame(result_hierarchical, columns=['DEPTH_START', 'DEPTH_END', 'Hierarchical'])
    df_hierarchical['Hierarchical_Ciflog'] = df_hierarchical['Hierarchical'].map(replace_dict_inversed_ciflog)
    df_hierarchical['Hierarchical_Origin'] = df_hierarchical['Hierarchical'].map(replace_dict_inversed)
    df_hierarchical.to_csv('df_hierarchical.csv', index=True)

    result_spectral = table_2_to_3(df_result_save[['DEPTH', 'Spectral']].values)
    df_spectral = pd.DataFrame(result_spectral, columns=['DEPTH_START', 'DEPTH_END', 'Spectral'])
    df_spectral['Spectral_Ciflog'] = df_spectral['Spectral'].map(replace_dict_inversed_ciflog)
    df_spectral['Spectral_Origin'] = df_spectral['Spectral'].map(replace_dict_inversed)
    df_spectral.to_csv('df_spectral.csv', index=True)

    result_gmm = table_2_to_3(df_result_save[['DEPTH', 'GMM']].values)
    df_gmm = pd.DataFrame(result_gmm, columns=['DEPTH_START', 'DEPTH_END', 'GMM'])
    df_gmm['GMM_Ciflog'] = df_gmm['GMM'].map(replace_dict_inversed_ciflog)
    df_gmm['GMM_Origin'] = df_gmm['GMM'].map(replace_dict_inversed)
    df_gmm.to_csv('df_gmm.csv', index=True)

    result_org = table_2_to_3(df_result_save[['DEPTH', TARGET_NAME[0]]].values)
    df_org = pd.DataFrame(result_org, columns=['DEPTH_START', 'DEPTH_END', 'LITHO'])
    df_org['LITHO_Ciflog'] = df_org['LITHO'].map(replace_dict_inversed_ciflog)
    df_org['LITHO_Origin'] = df_org['LITHO'].map(replace_dict_inversed)
    df_org.to_csv('df_org.csv', index=True)

    result_som = table_2_to_3(df_result_save[['DEPTH', 'SOM']].values)
    df_som = pd.DataFrame(result_som, columns=['DEPTH_START', 'DEPTH_END', 'SOM'])
    df_som['SOM_Ciflog'] = df_som['SOM'].map(replace_dict_inversed_ciflog)
    df_som['SOM_Origin'] = df_som['SOM'].map(replace_dict_inversed)
    df_som.to_csv('df_som.csv', index=True)