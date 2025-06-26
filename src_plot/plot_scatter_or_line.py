import matplotlib.pyplot as plt
import numpy as np


def plot_dataframe(df, X, Y, title=None, X_ticks=None, Y_ticks=None, figure_type='line'):
    """
    基于DataFrame绘制折线图或散点图

    参数:
    df : DataFrame - 包含绘图数据的Pandas DataFrame
    X : string - 作为X轴的列名
    Y : string - 作为Y轴的列名(可以传入多个列名组成的列表)
    title : string - 图表标题 (可选)
    X_ticks : dict - X轴刻度配置字典 (可选)
        {'values': [value1, value2, ...],  # 刻度位置
         'labels': [label1, label2, ...],  # 刻度标签
         'rotation': 45,  # 旋转角度
         'fontsize': 10,   # 字体大小
         'step': 5}        # 固定间隔刻度
    Y_ticks : dict - Y轴刻度配置字典 (可选)
        {'values': [value1, value2, ...],
         'labels': [label1, label2, ...],
         'rotation': 45,
         'fontsize': 10,
         'step': 5}
    figure_type : string - 图表类型 ('line'或'scatter')
    """

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 处理单列或多列Y的情况
    if isinstance(Y, str):
        Y = [Y]  # 转换为列表形式

    # 绘图
    for y_col in Y:
        if figure_type.lower() == 'scatter':
            # 绘制散点图
            plt.scatter(df[X], df[y_col], label=y_col, alpha=0.7, edgecolors='w')
        else:
            # 默认绘制折线图
            plt.plot(df[X], df[y_col], label=y_col, marker='o', markersize=4, linewidth=2)

    # 设置图表标题
    if title:
        plt.title(title, fontsize=14, pad=15)

    # 设置坐标轴标签
    plt.xlabel(X, fontsize=12)
    if len(Y) > 1:
        plt.ylabel('Value', fontsize=12)
    else:
        plt.ylabel(Y[0], fontsize=12)

    # 配置X轴刻度
    if X_ticks:
        if 'step' in X_ticks:
            # 等间距刻度
            min_x = df[X].min()
            max_x = df[X].max()
            step = X_ticks['step']
            tick_values = np.arange(min_x, max_x + step, step)
            plt.xticks(tick_values)
        elif 'values' in X_ticks and 'labels' in X_ticks:
            # 自定义刻度和标签
            plt.xticks(X_ticks['values'], X_ticks['labels'])
        elif 'values' in X_ticks:
            # 仅设置刻度位置
            plt.xticks(X_ticks['values'])

        # 设置旋转和字体大小
        if 'rotation' in X_ticks:
            plt.xticks(rotation=X_ticks['rotation'])
        if 'fontsize' in X_ticks:
            plt.tick_params(axis='x', labelsize=X_ticks['fontsize'])


    # 配置Y轴刻度
    if Y_ticks:
        if 'step' in Y_ticks:
            # 等间距刻度
            all_y = df[Y].values.flatten()
            min_y = np.nanmin(all_y)
            max_y = np.nanmax(all_y)
            step = Y_ticks['step']
            tick_values = np.arange(min_y, max_y + step, step)
            plt.yticks(tick_values)
        elif 'values' in Y_ticks and 'labels' in Y_ticks:
            # 自定义刻度和标签
            plt.yticks(Y_ticks['values'], Y_ticks['labels'])
        elif 'values' in Y_ticks:
            # 仅设置刻度位置
            plt.yticks(Y_ticks['values'])

        # 设置旋转和字体大小
        if 'rotation' in Y_ticks:
            plt.yticks(rotation=Y_ticks['rotation'])
        if 'fontsize' in Y_ticks:
            plt.tick_params(axis='y', labelsize=Y_ticks['fontsize'])

    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    if len(Y) > 1:
        plt.legend(loc='best', frameon=True)

    # 调整布局
    plt.tight_layout()

    # 显示图表
    plt.show()

if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # 创建示例数据
    dates = pd.date_range('2023-01-01', periods=30)
    data = {
        'Date': dates,
        'Temperature': np.sin(np.linspace(0, 6, 30)) * 10 + 25,
        'Humidity': np.cos(np.linspace(0, 5, 30)) * 30 + 60,
        'Pressure': np.linspace(990, 1030, 30) + np.random.normal(0, 2, 30),
        'Depth': np.linspace(0, 290, 30)
    }

    df = pd.DataFrame(data)

    # 示例1: 简单折线图
    plot_dataframe(
        df,
        X='Date',
        Y='Temperature',
        title='Temperature over Time'
    )

    # 示例2: 自定义刻度散点图
    x_ticks = {
        'values': pd.date_range('2023-01-01', '2023-01-30', freq='5d'),
        'rotation': 45,
        'fontsize': 9
    }

    y_ticks = {
        'step': 5
    }

    plot_dataframe(
        df,
        X='Date',
        Y='Humidity',
        figure_type='scatter',
        X_ticks=x_ticks,
        Y_ticks=y_ticks,
        title='Humidity Measurement'
    )

    # 示例3: 双Y轴折线图（自定义刻度标签）
    depth_ticks = {
        'values': [0, 100, 200, 300],
        'labels': ['Surface', '100m', '200m', '300m']
    }

    temperature_ticks = {
        'values': [20, 25, 30, 35],
        'labels': ['Cool', 'Warm', 'Hot', 'Very Hot']
    }

    plot_dataframe(
        df,
        X='Depth',
        Y=['Temperature', 'Pressure'],
        X_ticks=depth_ticks,
        Y_ticks=temperature_ticks,
        title='Deep Environment Measurements'
    )