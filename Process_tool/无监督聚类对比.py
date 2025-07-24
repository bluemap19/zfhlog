import numpy as np
import pandas as pd

from src_data_process.data_norm import data_Normalized
from src_data_process.data_unsupervised import ClusteringPipeline, evaluate_clustering

if __name__ == '__main__':
    # 1.初始化数据，合并属性列和分类列
    df = pd.read_excel(r'C:\Users\ZFH\Desktop\SOM-SHANXI.xlsx', sheet_name='Sheet1')

    print(df.describe())

    # 显示数据摘要
    print("数据摘要:")
    print(df.head())

    df.iloc[:, :] = data_Normalized(df.values)

    print(df.describe())

    # 2. 初始化接口
    Unsupervised_Pipeline = ClusteringPipeline(cluster_num=9, scale_data=True)
    # 3. 模型训练
    Unsupervised_Pipeline.fit(df[['GR', 'PE', 'AC', 'CNL', 'DEN']])


    pred_result = Unsupervised_Pipeline.predict(df[['GR', 'PE', 'AC', 'CNL', 'DEN']], algorithm=['KMeans', 'DBSCAN','Hierarchical','Spectral','GMM', 'SOM'])
    print(pred_result.describe())

    # 无监督聚类的评价标准Silhouette、CH、DBI、DVI计算
    unsupervised_evluate_result = evaluate_clustering(df[['GR', 'PE', 'AC', 'CNL', 'DEN']], pred_result)
    print(unsupervised_evluate_result)

    result = pd.concat([df, pred_result], axis=1)

    print(result.describe())
    result.to_excel(r'C:\Users\ZFH\Desktop\SOM-SHANXI-result.xlsx', sheet_name='Sheet1', index=False)
