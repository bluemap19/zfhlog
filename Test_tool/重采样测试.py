import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False



# 1. 生成不平衡数据集
X, y = make_classification(n_samples=1000, n_classes=3, n_features=5, n_informative=3,
                           weights=[0.05, 0.15, 0.8], random_state=42)
print("原始数据类别分布:", np.bincount(y))


# 2. 定义采样方法
# 欠采样
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X, y)
print("欠采样后分布:", np.bincount(y_rus))


# 过采样
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)
print("过采样后分布:", np.bincount(y_smote))


# 组合采样
smote_enn = SMOTEENN(random_state=42)
X_smoteenn, y_smoteenn = smote_enn.fit_resample(X, y)
print("组合采样后分布:", np.bincount(y_smoteenn))

# 新增代码部分（在原代码基础上添加）
# 3. PCA降维与可视化对比
plt.figure(figsize=(18, 12))
pca = PCA(n_components=2)
cmap = plt.cm.get_cmap('viridis', 3)  # 使用3色分类

# 定义数据集列表
datasets = [
    (X, y, "Original Data"),
    (X_rus, y_rus, "RandomUnderSampler"),
    (X_smote, y_smote, "SMOTE"),
    (X_smoteenn, y_smoteenn, "SMOTEENN")
]

# 创建2x2子图布局
for i, (data, labels, title) in enumerate(datasets):
    ax = plt.subplot(2, 2, i + 1)

    # 执行PCA降维
    pca_result = pca.fit_transform(data)  # 独立降维展示各采样方法分布

    # 绘制散点图
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],
                         c=labels, cmap=cmap, alpha=0.6,
                         edgecolors='w', linewidth=0.5)

    # 设置方差贡献率标签
    explained_var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({explained_var[0] * 100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({explained_var[1] * 100:.1f}%)", fontsize=10)

    # 添加标题和网格
    ax.set_title(f"{title}\nDistribution: {np.bincount(labels)}", pad=12)
    ax.grid(alpha=0.3, linestyle=':')



# 添加全局图例
plt.tight_layout(rect=[0, 0, 0.85, 0.96])
cbar_ax = plt.gcf().add_axes([0.88, 0.15, 0.02, 0.7])  # 右侧添加颜色条
plt.colorbar(scatter, cax=cbar_ax, ticks=[0, 1, 2])
cbar_ax.set_ylabel('Class Label', rotation=270, labelpad=15)

# 添加整体标题
plt.suptitle("Imbalanced Data Treatment Comparison via PCA Projection",
             y=0.98, fontsize=14, fontweight='bold')
plt.show()