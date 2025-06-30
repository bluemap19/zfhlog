import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

from src_plot.plot_heatmap import set_ultimate_chinese_font


def analyze_correlation(df, col_names, method='pearson', figsize=(10, 8),
                        annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                        return_matrix=True, plot_diag=False):
    """
    分析DataFrame中指定列的相关性，并绘制热力图

    参数：
    df: 输入的DataFrame
    col_names: 需要分析相关性的列名列表
    method: 相关性计算方法 ('pearson', 'kendall', 'spearman')
            默认为'pearson'皮尔逊相关系数
    figsize: 图表大小，默认为(10, 8)
    annot: 是否在热力图中显示数值，默认为True
    cmap: 热力图颜色映射，默认为'coolwarm'
    vmin: 颜色映射最小值，默认为-1
    vmax: 颜色映射最大值，默认为1
    return_matrix: 是否返回相关系数矩阵，默认为True
    plot_diag: 是否在热力图对角线上绘制分布图，默认为False

    返回：
    correlation_matrix: 相关系数矩阵（可选）
    """
    # 验证输入
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df必须是pandas DataFrame")

    if not isinstance(col_names, list) or len(col_names) == 0:
        raise ValueError("col_names必须是非空列表")

    # 检查所有列是否存在
    missing_cols = [col for col in col_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"以下列在DataFrame中不存在: {missing_cols}")


    # 设置样式# 1. 设置字体 - 使用终极方案
    chinese_font_prop = set_ultimate_chinese_font()

    # 提取目标列
    target_df = df[col_names].copy()

    # 检查数据是否为空
    if target_df.empty:
        raise ValueError("目标数据为空，无法计算相关性")

    # 计算相关系数
    if method in ['pearson', 'kendall', 'spearman']:
        correlation_matrix = target_df.corr(method=method)
    else:
        raise ValueError(f"不支持的相关性计算方法: {method}。请选择 'pearson', 'kendall' 或 'spearman'")

    # 创建可视化
    plt.figure(figsize=figsize)

    if plot_diag:
        # 在对角线上添加分布图
        g = sns.PairGrid(target_df, diag_sharey=False)
        g.map_upper(sns.scatterplot, s=10, alpha=0.5)
        g.map_lower(sns.kdeplot, fill=True, warn_singular=False)
        g.map_diag(sns.histplot, kde=True, fill=True, element="step")

        # 添加主标题
        title = f"特征相关性分析 ({method.upper()}系数)"
        plt.suptitle(title, fontsize=16, y=1.01, fontproperties=chinese_font_prop)
    else:
        # 创建热力图
        mask = None
        if not annot:
            # 非显著相关性使用白色遮盖
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        # 生成热力图
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=annot,
            fmt=".2f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            square=True
        )

        # 添加标题
        title = f"特征相关性热力图 ({method.upper()}系数)"
        plt.title(title, fontsize=14, fontproperties=chinese_font_prop)

        # 调整标签
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()

    # 标记显著相关
    print("\n显著相关关系 (|r| > 0.02):")
    strong_corrs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            col_i = correlation_matrix.columns[i]
            col_j = correlation_matrix.columns[j]
            corr_value = correlation_matrix.loc[col_i, col_j]
            if abs(corr_value) > 0.02:
                relation = "正相关" if corr_value > 0 else "负相关"
                strong_corrs.append((col_i, col_j, corr_value))
                print(f"{col_i} 与 {col_j}: {relation} (r = {corr_value:.2f})")

    if not strong_corrs:
        print("没有发现显著相关关系 (|r| > 0.02)")

    # 返回相关系数矩阵
    if return_matrix:
        return correlation_matrix


if __name__ == '__main__':
    # 假设我们有以下DataFrame
    data = {
        'STAT_ENT': np.random.rand(100),
        'STAT_DIS': np.random.rand(100) * 0.5 + np.random.rand(100),
        'STAT_CON': np.random.rand(100) * 0.3,
        'STAT_XY_HOM': np.random.rand(100) * 2,
        'STAT_HOM': np.random.rand(100) * 0.7,
        'STAT_XY_CON': np.random.rand(100) * 0.9,
        'DYNA_DIS': np.random.rand(100) * 1.5,
        'STAT_ENG': np.random.rand(100) * 0.6,
    }
    df = pd.DataFrame(data)

    # 定义要分析的列
    COL_NAMES = ['STAT_ENT', 'STAT_DIS', 'STAT_CON', 'STAT_XY_HOM',
                 'STAT_HOM', 'STAT_XY_CON', 'DYNA_DIS', 'STAT_ENG']

    # 调用接口进行分析
    corr_matrix = analyze_correlation(
        df=df,
        col_names=COL_NAMES,
        method='pearson',  # 可以改为'spearman'或'kendall'
        figsize=(12, 10),
        annot=True,  # 在热力图中显示数值
        cmap='vlag',  # 使用不同的颜色映射
        plot_diag=True  # 在对角线上绘制分布图
    )

    # 查看相关系数矩阵
    print("\n相关系数矩阵:")
    print(corr_matrix)