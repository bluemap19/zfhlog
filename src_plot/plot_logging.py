import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from src_logging.curve_preprocess import get_resolution_by_depth

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False


def visualize_well_logs(data: pd.DataFrame,
                        depth_col: str = 'Depth',
                        curve_cols: list = ['L1', 'L2', 'L3', 'L4'],
                        type_cols: list = ['Type1', 'Type2', 'Type3', 'Type4'],
                        figsize: tuple = (24, 10),
                        # color_palette: list = ['#FF0000', '#00FF00', '#0000FF', '#FFA500', '#FFFF00', '#00FFFF', '#FF00FF', '#8000FF', '#00FF80', '#FF0080' ],
                        colors: list = ['#FF0000', '#00FF00', '#0000FF', '#00FFFF', '#FF00FF', '#8000FF', '#00FF80', '#FF0080', '#FFA500', '#FFFF00'],
                        figure=None
                        ):
    """
    测井数据可视化接口
    参数：
    data : pd.DataFrame
        输入数据，必须包含深度列、测井曲线列和分类列
    depth_col : str
        深度列名称，默认'Depth'
    curve_cols : list
        测井曲线列名称列表，默认['L1', 'L2', 'L3', 'L4']
    type_cols : list
        分类结果列名称列表，默认['Type1', 'Type2', 'Type3', 'Type4']
    figsize : tuple
        图像尺寸，默认(24, 10)
    color_palette : list
        分类颜色列表，默认[红, 绿, 蓝, 橙]
    litho_width_config : dict
        分类显示宽度配置，键为分类标签，值为右侧边界位置比例
    """
    # 初始化可视化参数
    if figure is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = figure
    axs = fig.subplots(1, len(curve_cols) + len(type_cols), sharey=True)
    plt.subplots_adjust(left=0.03, right=0.93, bottom=0.03)
    # fig, axs = plt.subplots(1, len(curve_cols) + len(type_cols), figsize=figsize, sharey=True)
    # plt.subplots_adjust(left=0.03, right=0.93, bottom=0.03)

    # 数据验证
    required_cols = [depth_col] + curve_cols + type_cols
    if not set(required_cols).issubset(data.columns):
        missing = set(required_cols) - set(data.columns)
        raise ValueError(f"数据缺少必要列: {missing}")
        return

    if len(type_cols) > 0:
        # 类别长方形宽度设置
        Type_data = data[type_cols].values
        Type_list = list(np.unique(Type_data))
        litho_width_config = {}
        for i in range(len(Type_list)):
            if not np.isnan(Type_list[i]):      # Type_list[i] != np.nan的判断无效。NaN在Python中不等于任何值，包括自身，必须使用np.isnan()检测：
                index = int(Type_list[i])
                litho_width_config[index] = 0.1+i/len(Type_list)
            # litho_width_config: dict = {0: 0.3, 1: 0.4, 2: 0.5, 3: 0.6}

    # 按深度排序确保显示正确
    data = data.sort_values(depth_col).reset_index(drop=True)

    # 初始化显示范围
    current_top = data[depth_col].min()
    current_bottom = data[depth_col].max()
    window_size = (current_bottom - current_top) * 0.1  # 初始显示10%范围
    resolution_logging = get_resolution_by_depth(data[depth_col].dropna().values)
    print(f"current drawing well information is:{depth_col} {current_bottom:.2f}--->{current_top}, res:{resolution_logging:.2f}")

    # 创建滑动条
    ax_slider = plt.axes([0.94, 0.1, 0.01, 0.8])
    depth_slider = Slider(
        ax=ax_slider,
        label='Depth (m)',
        valmin=current_top,
        valmax=current_bottom - window_size,
        valinit=current_top,
        valstep=(current_bottom - current_top) / 1000,
        orientation='vertical'
    )

    # 绘制测井曲线
    def init_curves():
        plots = []
        for i, col in enumerate(curve_cols):
            ax = axs[i]
            line, = ax.plot(data[col], data[depth_col], color=colors[i % len(colors)], lw=0.5)
            ax.set_title(col)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(data[col].min() * 0.9, data[col].max() * 1.1)
            plots.append(line)
        return plots

    plots = init_curves()

    # 分类结果显示函数
    def plot_classification(ax, depth, litho):
        span_kwargs = {
            'edgecolor': 'none',  # 边界颜色设置为透明
            'linewidth': 0  # 边界线宽设为0
        }
        """绘制单个分类结果"""
        if math.isnan(litho):
            ax.axhspan(depth-resolution_logging/2, depth+resolution_logging/2, color=(0, 0, 0, 0))
        elif litho < 0:
            ax.axvspan(depth-resolution_logging/2, depth+resolution_logging/2, color=(0, 0, 0, 0))
        else:
            litho = int(litho)
            xmax = litho_width_config[litho]
            ax.axhspan(depth-resolution_logging/2, depth+resolution_logging/2, xmin=0, xmax=xmax,
                       facecolor=colors[litho % len(colors)], alpha=0.7, **span_kwargs)

    # 初始化分类面板
    def init_class_panels():
        class_axes = []
        for i, col in enumerate(type_cols):
            ax = axs[len(curve_cols) + i]
            ax.set_xticks([])
            ax.set_title(f"{type_cols[i]}")
            for depth, litho in zip(data[depth_col], data[col]):
                plot_classification(ax, depth, litho)
            class_axes.append(ax)
        return class_axes

    class_axes = init_class_panels()

    # 动态更新函数
    def update_display(val=None):
        top = depth_slider.val
        bottom = top + window_size

        # 更新所有坐标轴范围
        for ax in axs:
            ax.set_ylim(bottom, top)

        # 更新分类显示
        visible_data = data[(data[depth_col] >= top) & (data[depth_col] <= bottom)]
        for i, col in enumerate(type_cols):
            ax = class_axes[i]
            ax.clear()
            ax.set_xticks([])
            ax.set_title(f"{type_cols[i]}")
            for depth, litho in zip(visible_data[depth_col], visible_data[col]):
                plot_classification(ax, depth, litho)

        fig.canvas.draw_idle()

    # 事件绑定
    depth_slider.on_changed(update_display)

    # 鼠标滚动缩放
    def on_scroll(event):
        nonlocal window_size
        scale_factor = 1.3 if event.button == 'up' else 0.7
        new_size = window_size * scale_factor
        window_size = np.clip(new_size,
                              (current_bottom - current_top) * 0.01,  # 最小显示1%范围
                              (current_bottom - current_top))  # 最大显示全范围
        update_display()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    sample_data = pd.DataFrame(
        np.hstack([
            np.linspace(300, 400, 1000).reshape(-1, 1),
            np.random.randn(1000, 4),
            np.random.choice([0, 1, 2, 3], size=(1000, 4))
        ]),
        columns=['Depth', 'L1', 'L2', 'L3', 'L4', 'Type1', 'Type2', 'Type3', 'Type4']
    )

    # 调用可视化接口
    visualize_well_logs(
        data=sample_data,
        depth_col='Depth',
        curve_cols=['L1', 'L2', 'L3', 'L4'],
        type_cols=['Type1', 'Type2'],
        # type_cols=[],
    )