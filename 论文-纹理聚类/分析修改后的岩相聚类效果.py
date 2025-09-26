from src_data_process.data_supervised_evaluation import evaluate_supervised_clustering
from src_logging.logging_combine import combine_logging_data
from src_well_data.DATA_WELL import DATA_WELL


if __name__ == '__main__':
    WELL_CLASS = DATA_WELL(path_folder=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\SIMU4')
    print(WELL_CLASS.get_logging_path_list())
    print(WELL_CLASS.get_table_path_list())

    SOM_PATH = WELL_CLASS.search_data_path_by_charters(target_path_feature=['SOM'], target_file_type='table')
    SOM_TABLE_2 = WELL_CLASS.get_type_2(table_key=SOM_PATH)
    SOM_TABLE_2 = SOM_TABLE_2.rename(columns={"Type": "SOM"})
    print(SOM_TABLE_2.shape, SOM_TABLE_2.columns.tolist(), SOM_TABLE_2.iloc[0, 0], SOM_TABLE_2.iloc[-1, 0])

    AKBO_PATH = WELL_CLASS.search_data_path_by_charters(target_path_feature=['AKBO'], target_file_type='table')
    AKBO_TABLE_2 = WELL_CLASS.get_type_2(table_key=AKBO_PATH)
    AKBO_TABLE_2 = AKBO_TABLE_2.rename(columns={"Type": "AKBO"})
    print(AKBO_TABLE_2.shape, AKBO_TABLE_2.columns.tolist(), AKBO_TABLE_2.iloc[0, 0], AKBO_TABLE_2.iloc[-1, 0])

    HIERARCHICAL_PATH = WELL_CLASS.search_data_path_by_charters(target_path_feature=['HIERARCHICAL'], target_file_type='table')
    HIERARCHICAL_TABLE_2 = WELL_CLASS.get_type_2(table_key=HIERARCHICAL_PATH)
    HIERARCHICAL_TABLE_2 = HIERARCHICAL_TABLE_2.rename(columns={"Type": "HIERARCHICAL"})
    print(HIERARCHICAL_TABLE_2.shape, HIERARCHICAL_TABLE_2.columns.tolist(), HIERARCHICAL_TABLE_2.iloc[0, 0], HIERARCHICAL_TABLE_2.iloc[-1, 0])

    KMEANS_PATH = WELL_CLASS.search_data_path_by_charters(target_path_feature=['KMEANS'], target_file_type='table')
    KMEANS_TABLE_2 = WELL_CLASS.get_type_2(table_key=KMEANS_PATH)
    KMEANS_TABLE_2 = KMEANS_TABLE_2.rename(columns={"Type": "KMEANS"})
    print(KMEANS_TABLE_2.shape, KMEANS_TABLE_2.columns.tolist(), KMEANS_TABLE_2.iloc[0, 0], KMEANS_TABLE_2.iloc[-1, 0])

    ORG_PATH = WELL_CLASS.search_data_path_by_charters(target_path_feature=['ORG'], target_file_type='table')
    ORG_TABLE_2 = WELL_CLASS.get_type_2(table_key=ORG_PATH)
    ORG_TABLE_2 = ORG_TABLE_2.rename(columns={"Type": "ORG"})
    print(ORG_TABLE_2.shape, ORG_TABLE_2.columns.tolist(), ORG_TABLE_2.iloc[0, 0], ORG_TABLE_2.iloc[-1, 0])

    SPECTRAL_PATH = WELL_CLASS.search_data_path_by_charters(target_path_feature=['SPECTRAL'], target_file_type='table')
    SPECTRAL_TABLE_2 = WELL_CLASS.get_type_2(table_key=SPECTRAL_PATH)
    SPECTRAL_TABLE_2 = SPECTRAL_TABLE_2.rename(columns={"Type": "SPECTRAL"})
    print(SPECTRAL_TABLE_2.shape, SPECTRAL_TABLE_2.columns.tolist(), SPECTRAL_TABLE_2.iloc[0, 0], SPECTRAL_TABLE_2.iloc[-1, 0])

    combined_data_dropped = combine_logging_data(
        data_main=ORG_TABLE_2,
        data_vice=[SOM_TABLE_2, AKBO_TABLE_2, HIERARCHICAL_TABLE_2, KMEANS_TABLE_2, SPECTRAL_TABLE_2],
        depth_col='Depth',
        drop=True
    )
    combined_data_dropped[['ORG', 'SOM', 'AKBO', 'HIERARCHICAL', 'KMEANS', 'SPECTRAL']] = combined_data_dropped[['ORG', 'SOM', 'AKBO', 'HIERARCHICAL', 'KMEANS', 'SPECTRAL']].astype('int')
    print(combined_data_dropped.shape, combined_data_dropped.columns.tolist(), '\n',combined_data_dropped.iloc[-10:, :])

    results = evaluate_supervised_clustering(
        df=combined_data_dropped,
        col_org='ORG',
        cols_compare=['SOM', 'AKBO', 'HIERARCHICAL', 'KMEANS', 'SPECTRAL'],
        save_report=True,
        report_path='supervised_clustering_evaluation.xlsx',
    )
    # 打印结果
    print("\n监督聚类评估结果:")
    for method, metrics in results.items():
        print(f"\n{method}:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  加权F1分数: {metrics['f1_weighted']:.4f}")
        print(f"  宏平均F1分数: {metrics['f1_macro']:.4f}")
        print(f"  Cohen Kappa: {metrics['cohen_kappa']:.4f}")