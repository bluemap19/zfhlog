# PATH_FOLDER = r'C:\Users\ZFH\Desktop\1-15'
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from minisom import MiniSom
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src_file_op.dir_operation import search_files_by_criteria
from src_plot.plot_logging import visualize_well_logs


# 设置中文支持
def setup_chinese_support():
    """配置Matplotlib支持中文显示"""
    # 检查操作系统
    if os.name == 'nt':  # Windows系统
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
    else:  # Linux/Mac系统
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Heiti TC', 'STHeiti', 'SimHei']

    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

    # 设置字体大小
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


# 调用中文支持设置
setup_chinese_support()


def preprocess_data(df, attributes):
    """
    数据预处理函数 - 缺失值处理优化版

    参数:
    df: 原始数据DataFrame
    attributes: 属性列表

    返回:
    scaled_data: 标准化后的数据
    valid_attributes: 有效属性列表
    """
    # 1. 选择有效属性列
    valid_attributes = [attr for attr in attributes if attr and attr in df.columns]
    print(f"有效属性列: {valid_attributes}")

    # 2. 提取数据
    data = df[valid_attributes].copy()

    # 3. 处理缺失值
    # 定义缺失值范围
    missing_threshold_low = -99
    missing_threshold_high = 9999

    # 创建缺失值掩码
    missing_mask = (
            data.isna() |  # 空值
            (data < missing_threshold_low) |  # 小于-99
            (data > missing_threshold_high)  # 大于9999
    )

    # 统计每行缺失值数量
    row_missing_count = missing_mask.sum(axis=1)

    # 删除包含缺失值的行
    # 只删除缺失值比例超过5%的行
    max_missing_per_row = len(valid_attributes) * 0.1
    rows_to_keep = row_missing_count <= max_missing_per_row

    # 在应用筛选前记录保留的索引
    retained_indices = df.index[rows_to_keep]
    # 应用筛选
    data = data[rows_to_keep].copy()

    # 重置索引
    data.reset_index(drop=True, inplace=True)

    # 打印删除的行数
    n_deleted = len(df) - len(data)
    print(f"删除包含过多缺失值的行数: {n_deleted}/{len(df)} ({n_deleted / len(df) * 100:.2f}%)")

    # 4. 处理剩余缺失值
    # 对于剩余缺失值，使用列中位数填充
    for col in data.columns:
        # 计算缺失值数量
        n_missing = data[col].isna().sum()

        # 如果缺失值存在
        if n_missing > 0:
            # 计算中位数
            median_val = data[col].median()

            # 填充缺失值
            data[col].fillna(median_val, inplace=True)

            print(f"列 '{col}' 填充了 {n_missing} 个缺失值 (中位数: {median_val:.4f})")

    # 5. 数据标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    return scaled_data, valid_attributes, retained_indices


# PCA降维
def perform_pca(data, n_components=0.95):
    """
    执行PCA降维
    """
    # 自动确定主成分数量，保留95%的方差
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)

    # 输出解释方差
    print(f"主成分数量: {pca.n_components_}")
    print(f"累计解释方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")

    # 可视化解释方差
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('主成分数量')
    plt.ylabel('累计解释方差')
    plt.title('PCA解释方差曲线')
    plt.grid(True)
    plt.show()

    return pca_data, pca


# SOM聚类
def perform_som_clustering(data, grid_size=(3, 3), sigma=1.0, learning_rate=0.5, num_iteration=1000):
    """
    执行SOM聚类
    """
    # 初始化SOM
    som = MiniSom(
        x=grid_size[0],
        y=grid_size[1],
        input_len=data.shape[1],
        sigma=sigma,
        learning_rate=learning_rate
    )

    # 随机初始化权重
    som.random_weights_init(data)

    # 训练SOM
    print("开始训练SOM...")
    som.train_random(data, num_iteration=num_iteration)
    print("SOM训练完成")

    # 获取聚类结果
    cluster_labels = np.array([som.winner(x) for x in data])

    # 将二维坐标转换为一维聚类标签
    cluster_labels_1d = cluster_labels[:, 0] * grid_size[1] + cluster_labels[:, 1]

    # 计算轮廓系数
    if data.shape[1] > 1:  # 轮廓系数需要至少2个特征
        silhouette_avg = silhouette_score(data, cluster_labels_1d)
        print(f"轮廓系数: {silhouette_avg:.4f}")

    return cluster_labels, cluster_labels_1d, som


# 修改可视化函数
def visualize_som_clustering(som, data, cluster_labels, attributes):
  """
  可视化SOM聚类结果
  """
  # 1. 创建SOM热图
  plt.figure(figsize=(12, 12))
  plt.pcolor(som.distance_map().T, cmap='bone_r')  # 距离图
  plt.colorbar()

  # 标记聚类中心
  for i, x in enumerate(data):
    w = som.winner(x)
    plt.plot(w[0] + 0.5, w[1] + 0.5, 'o', markerfacecolor='None',
             markeredgecolor='r', markersize=10, markeredgewidth=2)

  plt.title('SOM聚类热图')
  plt.show()

  # 2. 特征权重可视化
  # 获取权重矩阵的维度
  n_features = som.get_weights().shape[2]

  # 动态计算子图布局
  n_cols = 4
  n_rows = (n_features + n_cols - 1) // n_cols

  plt.figure(figsize=(15, n_rows * 4))
  for i in range(n_features):  # 只遍历实际存在的特征维度
    plt.subplot(n_rows, n_cols, i + 1)
    plt.title(f'特征 {i + 1}')
    plt.pcolor(som.get_weights()[:, :, i].T, cmap='coolwarm')
    plt.colorbar()
  plt.tight_layout()
  plt.show()

  # 3. 聚类分布直方图
  plt.figure(figsize=(10, 6))
  plt.hist(cluster_labels[:, 0] * som._weights.shape[1] + cluster_labels[:, 1],
           bins=np.arange(0, som._weights.shape[0] * som._weights.shape[1] + 1),
           alpha=0.7)
  plt.title('聚类分布直方图')
  plt.xlabel('聚类标签')
  plt.ylabel('样本数量')
  plt.grid(True)
  plt.show()


if __name__ == '__main__':
    PATH_FOLDER = r'C:\Users\ZFH\Desktop\FY1-4HF'
    # PATH_FOLDER = r'C:\Users\ZFH\Desktop\6b'

    path_list_target = search_files_by_criteria(PATH_FOLDER, name_keywords=['logging'],
                                                file_extensions=['.xlsx'])

    print(path_list_target)
    DF_O = pd.read_excel(path_list_target[0], sheet_name=0, engine='openpyxl')
    print(DF_O.describe())

    print(DF_O.columns)     # ['井号', '深度', 'TVD', 'GR10', '_CAL', '_SPDH', '_RD', '_RS', '_DEN', '_AC', '_CNL', '_TOC', '_POR', '_P10', '_P25', '_P75', '_P90', '_P50', '_TOC实测', '_实测S2', '_实测S1', '_GRAY', '_QF', '_CLA', '_岩相1', '_岩相7']
    ATTRIBUTE_INPUT = ['GR10', '_RD', '_RS', '_DEN', '_AC', '_CNL', '_TOC', '_P10', '_P50', '_P90', '_QF', '_CLA', '_GRAY']

    # 1. 数据预处理
    scaled_data, attributes, retained_indices = preprocess_data(DF_O, ATTRIBUTE_INPUT)

    # 2. PCA降维
    pca_data, pca = perform_pca(scaled_data)

    # 3. SOM聚类
    grid_size = (3, 3)
    cluster_labels, cluster_labels_1d, som = perform_som_clustering(pca_data, grid_size=grid_size)

    # 4. 可视化聚类结果
    visualize_som_clustering(som, pca_data, cluster_labels, attributes)

    # 5. 将聚类结果添加到原始数据框
    # 创建新列并初始化为NaN
    DF_O['SOM_Cluster_2D'] = np.nan
    DF_O['SOM_Cluster'] = np.nan

    # 只对保留的行添加聚类结果
    DF_O.loc[retained_indices, 'SOM_Cluster_2D'] = [f"{x},{y}" for x, y in cluster_labels]
    DF_O.loc[retained_indices, 'SOM_Cluster'] = cluster_labels_1d

    # 6. 分析聚类结果
    # 只分析有聚类结果的行
    clustered_df = DF_O[DF_O['SOM_Cluster'].notna()]

    # 按聚类分组统计
    cluster_stats = clustered_df.groupby('SOM_Cluster')[attributes].agg(['mean', 'std'])
    print("\n聚类统计信息:")
    print(cluster_stats)

    # 'GR10', '_RD', '_RS', '_DEN', '_AC', '_CNL', '_TOC', '_P10', '_P50', '_P90', '_QF', '_CLA', '_GRAY'
    # 调用可视化接口
    visualize_well_logs(
        data=DF_O.loc[retained_indices],
        depth_col='深度',
        curve_cols=['GR10', '_RD', '_RS', '_DEN', '_AC', '_CNL', '_QF', '_CLA', '_GRAY'],
        # curve_cols=['温度', '测量电阻率', '温度校正因子', '去势电阻率'],
        type_cols=['SOM_Cluster'],
        figsize=(22, 10)
    )

    # # 保存结果
    # DF_O.to_excel(PATH_FOLDER+'\SOM聚类结果.xlsx', index=False)

