import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Callable
from pylab import mpl
import seaborn as sns
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False

def plot_clustering_heatmap(
        df: pd.DataFrame,
        title: str = "Clustering Algorithm Accuracy Distribution",
        condition_formatter: Optional[Callable] = None,
        figsize: tuple = (10, 6),
        cmap: str = "YlGnBu",
        font_scale: float = 1.0,
        figure: Optional[plt.Figure] = None  # 新增参数，用于接收外部figure
) -> plt.Axes:
    """生成dataframe准确率分布热力图接口"""
    # 参数校验
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须为Pandas DataFrame")

    # 创建画布（使用传入的figure或新建）
    if figure is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = figure
        fig.clear()  # 清除之前的绘图内容

    ax = fig.add_subplot(111)

    # 生成热力图核心逻辑
    heatmap = sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Accuracy Rate'},
        annot_kws={"size": 12 * font_scale},
        vmin=0,
        vmax=1,
        ax=ax  # 指定在传入的axes上绘制
    )

    # 设置标题
    ax.set_title(title, pad=20, fontsize=14 * font_scale)
    ax.set_xlabel('Algorithms', fontsize=10 * font_scale)
    ax.set_ylabel('Experimental Conditions', fontsize=10 * font_scale)

    # 格式化行标签
    if condition_formatter:
        y_labels = [condition_formatter(i) for i in df.index]
    else:
        y_labels = [f"Condition {i}" for i in df.index]
    x_labels = list(df.columns)
    heatmap.set_yticklabels(y_labels, rotation=45)
    heatmap.set_xticklabels(x_labels, rotation=45, ha='right')

    # 优化布局
    fig.tight_layout()
    return ax  # 返回axes对象