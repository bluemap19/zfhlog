import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

#
# def pandas_data_filtration(
#     data_input: pd.DataFrame,
#     contamination: float = 0.05,
#     eps: float = 0.5,
#     min_samples: int = 5,
#     pca_variance: float = 0.95,
#     enable_visualization: bool = False
# ) -> pd.DataFrame:
#     assert isinstance(data_input, pd.DataFrame), "Input data should be a Pandas DataFrame"
#     data = data_input.copy()  # 创建副本
#     # 删除任何有空值的行 严格过滤所有空值形式（包括字符串"nan"、空字符串等）
#     invalid_vals = {'', 'nan', 'NaN', 'None', 'NULL', None, np.nan, -2147483648}
#     mask = ~data.isin(invalid_vals).any(axis=1)
#     data = data[mask].reset_index(drop=True)
#
#     """集成异常检测与聚类分析"""
#     # 异常检测
#     iso_forest = IsolationForest(n_estimators=50, contamination=contamination, random_state=42)  # contamination：定义数据集中异常值的预期比例
#     outliers = iso_forest.fit_predict(data.values)
#
#     # PCA处理部分
#     pca = PCA(n_components=pca_variance)  # 降低保留方差比例
#     reduced_data = pca.fit_transform(data)
#     data = pd.DataFrame(reduced_data, index=data.index)  # 重建DataFrame
#
#     # DBSCAN数据剔除
#     cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)
#
#     # 综合筛选条件
#     valid_idx = np.where((outliers == 1) & (cluster_labels != -1))[0]
#     result = data.iloc[valid_idx]
#
#     # 可视化
#     if enable_visualization and result.shape[1] >= 2:
#         import matplotlib.pyplot as plt
#         plt.scatter(result.iloc[:, 0], result.iloc[:, 1])
#         plt.title("Cleaned Data Distribution")
#         plt.show()
#
#     return result
#
#
# # 生成基础数据（包含正常值和异常值）
# np.random.seed(42)
# base_data = np.vstack([
#     make_blobs(n_samples=100, centers=1, cluster_std=0.5)[0],  # 正常数据簇
#     np.array([[1000, -1000], [-1000, 500]]),                   # 明显异常点
#     np.full((5,2), -2147483648)                                # 特殊无效值
# ])
#
# # 构造包含多种无效形式的DataFrame
# test_df = pd.DataFrame(
#     data=base_data,
#     columns=['Feature1', 'Feature2']
# ).sample(frac=1).reset_index(drop=True)  # 打乱顺序
#
# # 在最后一列添加特殊无效值
# invalid_values = ['nan', 'NaN', '', 'None', 'NULL', np.nan, None, -2147483648]
# test_df['LastCol'] = [invalid_values[i%8] for i in range(len(test_df))]
#
# # 添加随机缺失值
# test_df.iloc[::10, :] = np.nan
#
# print("原始测试数据样例：")
# print(test_df.head(10))
#
# pandas_data_filtration(test_df)



def pandas_data_filtration(
        data_input: pd.DataFrame,
        contamination: float = 0.05,
        eps: float = 0.5,
        min_samples: int = 5,
        pca_variance: float = 0.95,
) -> pd.DataFrame:
    # 输入校验a
    assert isinstance(data_input, pd.DataFrame), "Input data should be a Pandas DataFrame"
    data = data_input.copy(deep=True)

    # Step 3: 异常检测（使用清洗后的数据）
    iso_forest = IsolationForest(n_estimators=50, contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(data.values)  # 改为 data.values

    # Step 4: 降维处理
    pca = PCA(n_components=pca_variance)
    reduced_data = pca.fit_transform(data)
    data = pd.DataFrame(reduced_data, index=data.index)

    # Step 5: 聚类分析
    cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)

    # Step 6: 综合筛选
    valid_idx = np.where((outliers == 1) & (cluster_labels != -1))[0]

    return valid_idx


def pdnads_data_drop(data_input: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(data_input, pd.DataFrame), "Input data should be a Pandas DataFrame"
    data = data_input.copy(deep=True)

    # 分列处理无效值 ---
    # 数值列处理
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        # 替换极端值
        data[numeric_cols] = data[numeric_cols].replace(-2147483648, np.nan)
        # 删除全空行
        data = data.dropna(subset=numeric_cols, how='all')
        print('data number drop:{}-->{}'.format(data_input.shape, data.shape))

    # 非数值列处理
    # str_cols = data.select_dtypes(exclude=[np.number]).columns
    invalid_str = {'', 'nan', 'NaN', 'None', 'NULL', np.nan, None, -2147483648, '-2147483648'}
    mask = ~data.isin(invalid_str).any(axis=1)
    data = data[mask].reset_index(drop=True)
    print('data all dropped-->{}'.format(data.shape))

    data = data.reset_index(drop=True)

    # 检查数据有效性 ---
    if data.empty:
        raise ValueError("过滤后数据为空，请调整过滤条件")
    data = data.select_dtypes(include=[np.number])
    if data.empty:
        raise ValueError("无有效数值列")

    return data


# # 生成测试数据
# np.random.seed(42)
# base_data = np.vstack([
#     make_blobs(n_samples=100, centers=1, cluster_std=0.5)[0],
#     np.array([[1000, -1000], [-1000, 500]]),
#     np.full((5, 2), -2147483648),
#     np.array([['', 1], [np.nan, 3], [None, 5], [4, 8]]),
# ])
#
# test_df = pd.DataFrame(
#     data=base_data,
#     columns=['Feature1', 'Feature2']
# ).sample(frac=1).reset_index(drop=True)
#
#
#
# # # 执行清洗
# # cleaned_df = pandas_data_filtration(test_df, enable_visualization=True)
# # print("\n清洗后数据：")
# # print(cleaned_df.head())
#
# a = pdnads_data_drop(test_df)