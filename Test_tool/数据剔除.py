import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.utils import resample
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# 1. 生成模拟测井数据（5个特征对应常见测井曲线）
def generate_well_logging_data():
    np.random.seed(42)

    # 生成平衡数据
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        n_features=5,
        n_informative=3,
        n_clusters_per_class=1,
        random_state=42
    )

    # 手动调整类别分布
    target_distribution = [0.05, 0.15, 0.8]  # 目标比例

    # 按类别分割数据
    samples_per_class = []
    for class_idx in range(3):
        indices = np.where(y == class_idx)[0]
        samples = X[indices], y[indices]
        samples_per_class.append(samples)

    # 计算每个类别需要的样本数
    n_samples = [int(1000 * ratio) for ratio in target_distribution]

    # 下采样每个类别
    resampled_data = []
    for i in range(3):
        X_class, y_class = samples_per_class[i]
        X_res, y_res = resample(X_class, y_class,
                                n_samples=n_samples[i],
                                replace=True,
                                random_state=42)
        resampled_data.append((X_res, y_res))

    # 合并数据
    X_final = np.vstack([d[0] for d in resampled_data])
    y_final = np.hstack([d[1] for d in resampled_data])

    return X_final, y_final


# 2. 异常值检测与可视化函数
def visualize_outlier_detection(data, contamination=0.05):
    """带异常值标注的测井数据可视化"""
    plt.figure(figsize=(20, 12))

    # 训练Isolation Forest模型
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        behaviour='new'
    )
    iso_forest.fit(data)
    scores = iso_forest.decision_function(data)
    data['anomaly_score'] = scores
    data['is_outlier'] = iso_forest.predict(data) == -1

    # 主成分分析降维
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data.iloc[:, :5])
    data['PC1'] = principal_components[:, 0]
    data['PC2'] = principal_components[:, 1]

    # 创建可视化布局
    plt.suptitle(f"测井数据异常值检测 (异常比例: {contamination * 100}%)",
                 fontsize=16, y=1.02)

    # 子图1：主成分空间分布
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
    sns.scatterplot(x='PC1', y='PC2',
                    hue='is_outlier',
                    style='is_outlier',
                    palette={False: 'steelblue', True: 'firebrick'},
                    markers={False: 'o', True: 'X'},
                    data=data,
                    ax=ax1)
    ax1.set_title("主成分空间分布")
    ax1.annotate('异常区域', xy=(0.8, 0.9), xycoords='axes fraction',
                 color='firebrick', fontsize=12)

    # 子图2：各测井曲线箱线图
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
    boxprops = dict(linestyle='-', linewidth=1.5, color='darkorange')
    flierprops = dict(marker='o', markersize=8,
                      markerfacecolor='firebrick', markeredgecolor='none')
    data.iloc[:, :5].plot(kind='box', ax=ax2,
                          patch_artist=True,
                          boxprops=boxprops,
                          flierprops=flierprops)
    ax2.set_title("测井曲线箱线图")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.grid(axis='y', alpha=0.3)

    # 子图3：异常值分布直方图
    ax3 = plt.subplot2grid((3, 4), (1, 2), colspan=2)
    sns.histplot(data=data, x='anomaly_score',
                 hue='is_outlier', element='step',
                 palette={False: 'steelblue', True: 'firebrick'},
                 bins=30, ax=ax3)
    ax3.set_title("异常分数分布")
    ax3.axvline(x=np.percentile(data.anomaly_score, 100 * contamination),
                color='darkorange', linestyle='--', lw=2)
    ax3.annotate(f'阈值: {np.percentile(data.anomaly_score, 100 * contamination):.2f}',
                 xy=(0.7, 0.8), xycoords='axes fraction',
                 color='darkorange', fontsize=12)

    # 子图4：特征相关性矩阵
    ax4 = plt.subplot2grid((3, 4), (2, 0), colspan=2)
    corr_matrix = data.iloc[:, :5].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f",
                cmap='coolwarm', mask=mask,
                linewidths=0.5, ax=ax4)
    ax4.set_title("测井参数相关性矩阵")

    # 子图5：异常值在各参数的分布
    ax5 = plt.subplot2grid((3, 4), (2, 2), colspan=2)
    for i, col in enumerate(data.columns[:5]):
        sns.kdeplot(data=data[~data.is_outlier][col],
                    label='正常数据', color='steelblue', ax=ax5)
        sns.kdeplot(data=data[data.is_outlier][col],
                    label='异常数据', color='firebrick', ax=ax5)
    ax5.set_title("参数分布对比")
    ax5.legend()

    plt.tight_layout()
    plt.show()

    return data


# 3. 执行流程
if __name__ == "__main__":
    # 生成调整后的数据
    X, y = generate_well_logging_data()  # 使用上述生成的X_final和y_final

    # 转换为DataFrame并模拟测井参数
    well_data = pd.DataFrame(X, columns=['GR', 'DEN', 'CNL', 'SP', 'RES'])
    for col in well_data:
        well_data[col] = well_data[col].apply(lambda x: x * 50 + 100)  # 调整到合理范围

    # 异常值检测
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    well_data['is_outlier'] = iso_forest.fit_predict(well_data)

    # 降维可视化
    pca = PCA(n_components=2)
    components = pca.fit_transform(well_data)
    well_data['PC1'] = components[:, 0]
    well_data['PC2'] = components[:, 1]

    # 可视化
    plt.figure(figsize=(12, 8))
    plt.scatter(well_data['PC1'], well_data['PC2'],
                c=well_data['is_outlier'],
                cmap='viridis',
                alpha=0.6,
                edgecolors='w',
                linewidth=0.5)

    plt.title('测井数据异常值分布 (PCA 降维)')
    plt.xlabel('主成分 1')
    plt.ylabel('主成分 2')
    plt.colorbar(label='异常值 (1=正常, -1=异常)')
    plt.grid(True, alpha=0.3)
    plt.show()

    # 输出统计信息
    print("异常值统计:")
    print(f"总样本数: {len(well_data)}")
    print(f"异常值数量: {(well_data['is_outlier'] == -1).sum()}")