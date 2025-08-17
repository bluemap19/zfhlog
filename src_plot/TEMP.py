import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from src_logging.curve_preprocess import get_resolution_by_depth

# 设置中文字体支持 - 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False


def visualize_well_logs(data: pd.DataFrame,
                        depth_col: str = 'Depth',
                        curve_cols: list = ['L1', 'L2', 'L3', 'L4'],
                        type_cols: list = [],
                        figsize: tuple = (12, 10),
                        colors: list = ['#FF0000', '#00FF00', '#0000FF', '#00FFFF',
                                        '#FF00FF', '#8000FF', '#00FF80', '#FF0080',
                                        '#FFA500', '#FFFF00'],
                        legend_dict: dict = {},
                        figure=None):
    """
    测井数据可视化接口 - 详细注释版

    功能：创建专业的测井数据可视化图像，包含曲线图、分类面板、深度滑动条和交互功能

    参数详解：
    data : pd.DataFrame - 输入数据，必须包含深度列、测井曲线列和分类列
    depth_col : str - 深度列名称，默认'Depth'
    curve_cols : list - 测井曲线列名称列表，默认['L1','L2','L3','L4']
    type_cols : list - 分类结果列名称列表，默认['Type1','Type2','Type3','Type4']
    figsize : tuple - 图像尺寸，默认(12, 10)
    colors : list - 颜色列表，用于曲线和分类显示
    legend_dict : dict - 图例映射字典，键为类型ID，值为类型名称
    figure : matplotlib Figure对象 - 可选，可传入现有图形对象

    返回：无，直接显示图像
    """
    # 设置图像绘制底部从那开始，这个取决于是否绘制图例legend_dict
    logging_bottle_with_legend = 0.065
    logging_bottle_without_legend = 0.03

    # ===================== 1. 初始化阶段 =====================
    # 1.1 数据验证 - 确保数据包含所有必要列
    required_cols = [depth_col] + curve_cols + type_cols
    if not set(required_cols).issubset(data.columns):
        # 计算缺失的列
        missing = set(required_cols) - set(data.columns)
        # 抛出错误并显示缺失列
        raise ValueError(f"数据缺少必要列: {missing}")

    # 1.2 创建图形和子图布局
    # 计算总子图数量 = 曲线图数量 + 分类面板数量
    n_plots = len(curve_cols) + len(type_cols)

    if figure is None:
        # 创建新图形和子图
        fig, axs = plt.subplots(
            1, n_plots,  # 1行，n_plots列
            figsize=figsize,
            sharey=True,  # 所有子图共享Y轴（深度轴）
            gridspec_kw={'wspace': 0.0}  # 子图之间无水平间距
        )
    else:
        # 使用现有图形
        fig = figure
        axs = fig.subplots(1, n_plots, sharey=True, gridspec_kw={'wspace': 0.0})

    # 1.3 调整图形布局 - 紧凑布局
    # 动态调整底部空间 - 新添加的关键修改
    # 如果有图例，预留更多底部空间（高度增加20%），否则使用默认空间
    legend_adjustment = logging_bottle_with_legend if legend_dict else logging_bottle_without_legend
    # 1.3 调整图形布局 - 根据图例状态调整底部空间
    plt.subplots_adjust(
        left=0.02, right=0.97,
        bottom=legend_adjustment,  # 动态底部空间
        top=0.95,
        wspace=0.0
    )

    # 1.4 数据预处理
    # 按深度排序确保正确显示
    data = data.sort_values(depth_col).reset_index(drop=True)
    # 获取深度范围
    depth_min = data[depth_col].min()
    depth_max = data[depth_col].max()
    # 初始显示窗口大小为总深度的10%
    window_size = (depth_max - depth_min) * 0.1
    # 计算深度分辨率（相邻深度点之间的平均距离）
    resolution = get_resolution_by_depth(data[depth_col].dropna().values)

    # 打印调试信息
    print(f"绘制井数据: {depth_col} {depth_min:.2f}-{depth_max:.2f}m, "
          f"分辨率: {resolution:.4f}m, 数据量: {len(data)}")

    # 1.5 创建分类宽度配置（用于分类面板显示）
    if type_cols:
        # 提取所有分类数据
        type_data = data[type_cols].values
        # 获取唯一分类值（排除NaN）
        unique_types = np.unique(type_data[~np.isnan(type_data)])
        type_count = len(unique_types)

        # 自动计算宽度配置（均匀分布）
        litho_width_config = {}
        for i, t in enumerate(sorted(unique_types)):
            litho_width_config[int(t)] = (i + 1) / type_count

        # 如果只有一种类型，填充整个空间
        if type_count == 1:
            litho_width_config[unique_types[0]] = 1.0
        print(f'设置分类宽度配置: {litho_width_config}')
    else:
        litho_width_config = {}

    # ===================== 2. 创建滑动条 =====================
    # 2.1 创建滑动条区域
    slider_bottle_adjustment = logging_bottle_with_legend if legend_dict else logging_bottle_without_legend
    slider_length_adjustment = 0.95 - slider_bottle_adjustment
    ax_slider = plt.axes([0.97, slider_bottle_adjustment, 0.035, slider_length_adjustment])  # 位置和尺寸

    # 2.2 创建滑动条对象
    depth_slider = Slider(
        ax=ax_slider,
        label='',  # 清空默认标签
        valmin=depth_min,  # 最小值
        valmax=depth_max - window_size,  # 最大值
        valinit=depth_min,  # 初始值
        valstep=(depth_max - depth_min) / 1000,  # 步长
        orientation='vertical'  # 垂直方向
    )

    # 2.3 添加垂直文本标签
    ax_slider.text(
        x=0.5, y=0.5,  # 居中位置
        s='Depth(m)',  # 文本内容
        rotation=270,  # 旋转270度变为垂直
        ha='center', va='center',  # 水平和垂直居中
        transform=ax_slider.transAxes,  # 使用相对坐标
        fontsize=10  # 字体大小
    )

    # ===================== 3. 绘制测井曲线 =====================
    def init_curves():
        """初始化并绘制测井曲线"""
        plots = []  # 存储曲线对象

        # 定义标题道参数
        title_margin = 0.001  # 标题道边距
        title_height = 0.04  # 标题道高度

        # 遍历每个曲线列
        for i, col in enumerate(curve_cols):
            ax = axs[i]  # 获取当前子图

            # 3.1 获取原始子图位置
            orig_pos = ax.get_position()

            # 3.2 创建标题道背景矩形
            # 计算标题道位置和尺寸
            title_bbox = [
                orig_pos.x0 + title_margin,  # x0位置
                orig_pos.y0 + orig_pos.height + title_margin,  # y0位置
                orig_pos.width - 2 * title_margin,  # 宽度
                title_height  # 高度
            ]

            # 创建矩形对象
            title_rect = plt.Rectangle(
                (title_bbox[0], title_bbox[1]),  # 左下角坐标
                title_bbox[2], title_bbox[3],  # 宽度和高度
                transform=fig.transFigure,  # 使用图形坐标系
                facecolor='#f5f5f5',  # 浅灰色背景
                edgecolor='#aaaaaa',  # 浅灰色边框
                linewidth=1,  # 边框宽度
                clip_on=False,  # 禁用裁剪
                zorder=10  # 图层顺序（确保在最上层）
            )
            fig.add_artist(title_rect)  # 添加到图形

            # 3.3 添加测井道的标题文本
            title_text = plt.Text(
                title_bbox[0] + title_bbox[2] / 2,  # 水平居中
                title_bbox[1] + title_bbox[3] / 2,  # 垂直居中
                col,  # 曲线名称
                fontsize=12,  # 字体大小
                fontweight='bold',  # 粗体
                color='#222222',  # 深灰色文字
                ha='center', va='center',  # 水平和垂直居中
                transform=fig.transFigure,  # 使用图形坐标系
                clip_on=False,  # 禁用裁剪
                zorder=11  # 在矩形上层
            )
            fig.add_artist(title_text)

            # 3.4 添加专业修饰元素（彩色装饰条）
            fig.add_artist(plt.Rectangle(
                (title_bbox[0] + title_bbox[2] * 0.6, title_bbox[1] + title_bbox[3] * 0.5),  # 位置
                title_bbox[2] * 0.3, title_bbox[3] * 0.05,  # 尺寸
                transform=fig.transFigure,  # 使用图形坐标系
                facecolor=colors[i % len(colors)],  # 使用曲线颜色
                edgecolor='none',  # 无边框
                clip_on=False,  # 禁用裁剪
                zorder=12  # 在最上层
            ))

            # 3.5 绘制测井曲线
            line, = ax.plot(
                data[col],  # X轴数据（测井值）
                data[depth_col],  # Y轴数据（深度）
                color=colors[i % len(colors)],  # 曲线颜色
                lw=1.0,  # 线宽
                linestyle='-'  # 实线
            )

            # 3.6 设置子图属性
            ax.set_title(col, fontsize=10, pad=5)  # 紧凑标题
            ax.grid(True, alpha=0.3)  # 添加网格（半透明）
            # 设置X轴范围（测井值范围）
            ax.set_xlim(data[col].min() * 0.9, data[col].max() * 1.1)

            # 3.7 特殊处理第一个子图（显示深度轴）
            if i == 0:
                # 设置Y轴范围（深度范围）
                ax.set_ylim([depth_min, depth_max])
                # 添加专业深度刻度（最多10个刻度）
                ax.yaxis.set_major_locator(plt.MaxNLocator(10))
                ax.tick_params(axis='y', labelsize=8, rotation=90, pad=1)  # 刻度标签大小
            else:
                # 其他子图不显示Y轴
                ax.set_ylim(None)  # 清除预设范围
                ax.tick_params(left=False, labelleft=False)  # 隐藏Y轴刻度和标签


        return plots

    # 调用函数初始化曲线
    plots = init_curves()

    # ===================== 4. 绘制分类面板 =====================
    # 4.1 定义分类显示函数
    def plot_classification(ax, depth, litho):
        """在分类面板上绘制单个分类数据点"""
        # 消除矩形之间的间隙
        span_kwargs = {
            'edgecolor': 'none',  # 无边框
            'linewidth': 0,  # 线宽为0
            'clip_on': False  # 禁用裁剪，确保边缘对齐
        }

        # 处理无效值
        if math.isnan(litho):
            return  # 不绘制NaN值
        if litho < 0:
            return  # 忽略负值

        litho = int(litho)  # 转换为整数
        # 获取分类宽度（默认为0.1）
        xmax = litho_width_config.get(litho, 0.1)

        # 在分类面板上绘制矩形
        ax.axhspan(
            depth - resolution / 2,  # 矩形顶部
            depth + resolution / 2,  # 矩形底部
            xmin=0,  # 左侧位置
            xmax=xmax,  # 右侧位置
            facecolor=colors[litho % len(colors)],  # 颜色
            alpha=1.0,  # 不透明度
            **span_kwargs  # 其他参数
        )

    # 4.2 初始化分类面板
    def init_class_panels():
        """初始化并绘制分类面板"""
        class_axes = []  # 存储分类面板对象

        # 遍历每个分类列
        for i, col in enumerate(type_cols):
            # 计算子图索引（曲线图之后）
            ax_idx = len(curve_cols) + i
            ax = axs[ax_idx]  # 获取当前子图

            # 4.2.1 定义标题道参数
            title_margin = 0.001  # 标题道边距
            title_height = 0.04  # 标题道高度

            # 获取原始子图位置
            orig_pos = ax.get_position()

            # 4.2.2 创建标题道背景矩形
            title_bbox = [
                orig_pos.x0 + title_margin,  # x0位置
                orig_pos.y0 + orig_pos.height + title_margin + title_margin,  # y0位置
                orig_pos.width - 2 * title_margin,  # 宽度
                title_height  # 高度
            ]

            # 创建矩形对象
            title_rect = plt.Rectangle(
                (title_bbox[0], title_bbox[1]),  # 左下角坐标
                title_bbox[2], title_bbox[3],  # 宽度和高度
                transform=fig.transFigure,  # 使用图形坐标系
                facecolor='#f5f5f5',  # 浅灰色背景
                edgecolor='#aaaaaa',  # 浅灰色边框
                linewidth=1,  # 边框宽度
                clip_on=False,  # 禁用裁剪
                zorder=10  # 图层顺序（确保在最上层）
            )
            fig.add_artist(title_rect)  # 添加到图形

            # 4.2.3 添加标题文本
            title_text = plt.Text(
                title_bbox[0] + title_bbox[2] / 2,  # 水平居中
                title_bbox[1] + title_bbox[3] / 2,  # 垂直居中
                col,  # 分类列名称
                fontsize=12,  # 字体大小
                fontweight='bold',  # 粗体
                color='#222222',  # 深灰色文字
                ha='center', va='center',  # 水平和垂直居中
                transform=fig.transFigure,  # 使用图形坐标系
                clip_on=False,  # 禁用裁剪
                zorder=11  # 在矩形上层
            )
            fig.add_artist(title_text)

            # 4.2.4 设置分类面板属性
            ax.set_xticks([])  # 隐藏X轴刻度
            ax.set_title(f"{col}", fontsize=10, pad=5)  # 紧凑标题

            # 4.2.5 绘制分类数据
            for depth, litho in zip(data[depth_col], data[col]):
                plot_classification(ax, depth, litho)

            # 4.2.6 分类面板不设置Y轴范围
            ax.set_ylim(None)  # 清除预设范围
            ax.tick_params(left=False, labelleft=False)  # 隐藏Y轴刻度和标签

            class_axes.append(ax)  # 保存分类面板对象

        return class_axes

    # 调用函数初始化分类面板
    class_axes = init_class_panels()

    # ===================== 5. 动态更新函数 =====================
    def update_display(val=None):
        """更新显示的函数（响应深度调整滑动条变化）"""
        # 5.1 获取当前显示范围
        top = depth_slider.val  # 顶部深度
        bottom = top + window_size  # 底部深度

        # 5.2 更新所有子图的Y轴范围
        for ax in axs:
            ax.set_ylim(bottom, top)

        # 5.3 更新分类显示（仅显示可见区域）
        if type_cols:
            # 获取当前可见深度范围的数据
            visible_data = data[(data[depth_col] >= top) & (data[depth_col] <= bottom)]

            # 更新每个分类面板
            for i, col in enumerate(type_cols):
                ax = class_axes[i]  # 获取分类面板
                ax.clear()  # 清除原有内容
                ax.set_xticks([])  # 隐藏X轴刻度
                ax.set_title(f"{col}", fontsize=10, pad=5)  # 重新设置标题

                # 重新绘制可见区域的分类数据
                for depth, litho in zip(visible_data[depth_col], visible_data[col]):
                    plot_classification(ax, depth, litho)

        # 5.4 刷新图形
        fig.canvas.draw_idle()

    # ===================== 6. 事件绑定 =====================
    # 6.1 绑定滑动条事件
    depth_slider.on_changed(update_display)

    # 6.2 鼠标滚轮缩放功能
    def on_scroll(event):
        """处理鼠标滚轮事件（缩放功能）"""
        nonlocal window_size  # 声明使用外部变量

        # 确定缩放方向（上滚放大，下滚缩小）
        scale_factor = 1.3 if event.button == 'up' else 0.7
        new_size = window_size * scale_factor  # 计算新窗口大小

        # 计算最小和最大窗口大小
        min_size = (depth_max - depth_min) * 0.01  # 最小显示1%范围
        max_size = depth_max - depth_min  # 最大显示全范围

        # 应用缩放（在合理范围内）
        if min_size <= new_size <= max_size:
            window_size = new_size
            # 更新滑动条最大值（确保不超出范围）
            depth_slider.valmax = depth_max - window_size
            # 更新显示
            update_display()

    # 6.3 绑定鼠标滚轮事件
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # ===================== 7. 创建统一图例 =====================
    if legend_dict:
        # 7.1 计算图例参数
        n_items = len(legend_dict)  # 图例项数量
        legend_height = 0.02  # 图例高度占图像高度的比例
        legend_width = 0.8  # 图例宽度占图像宽度的比例

        # 7.2 创建图例区域
        legend_ax = plt.axes(
            [0.1, 0.01, legend_width, legend_height],  # 位置和尺寸
            frameon=False  # 无边框
        )
        legend_ax.set_axis_off()  # 隐藏坐标轴

        # 7.3 创建图例项
        handles = []  # 图例句柄（颜色块）
        labels = []  # 图例标签（文本）
        sorted_keys = sorted(legend_dict.keys())  # 排序键

        # 遍历每个图例项
        for key in sorted_keys:
            # 创建颜色块
            color = colors[key % len(colors)]  # 获取颜色
            patch = plt.Rectangle(
                (0, 0), 1, 1,  # 尺寸
                facecolor=color,  # 填充颜色
                edgecolor='none'  # 无边框
            )
            handles.append(patch)
            labels.append(legend_dict[key])

        # 7.4 创建图例
        legend = legend_ax.legend(
            handles, labels,  # 句柄和标签
            loc='center',  # 居中
            ncol=n_items,  # 水平排列
            frameon=True,  # 显示边框
            framealpha=0.8,  # 边框透明度
            fancybox=True,  # 圆角边框
            fontsize=10,  # 字体大小
            handlelength=1.5,  # 句柄长度
            handleheight=1.5,  # 句柄高度
            handletextpad=0.5,  # 句柄与文本间距
            borderpad=0.5,  # 边框内边距
            columnspacing=1.0  # 列间距
        )

        # 7.5 设置图例样式
        frame = legend.get_frame()  # 获取图例边框
        frame.set_facecolor('#f8f8f8')  # 浅灰色背景
        frame.set_edgecolor('#aaaaaa')  # 边框颜色

    # ===================== 8. 初始显示和展示 =====================
    update_display()  # 初始更新显示
    plt.show()  # 显示图形


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    sample_data = pd.DataFrame(
        np.hstack([
            np.linspace(300, 400, 1000).reshape(-1, 1),  # 深度列
            np.random.randn(1000, 4),  # 4条测井曲线
            np.random.choice([0, 1, 2, 3], size=(1000, 4))  # 4个分类列
        ]),
        columns=['Depth', 'L1', 'L2', 'L3', 'L4', 'D2', 'D4', 'D3', 'D5']
    )

    # 调用可视化接口
    visualize_well_logs(
        data=sample_data,
        depth_col='Depth',
        curve_cols=['L1', 'L2', 'L3', 'L4'],
        type_cols=['D2', 'D5'],
        legend_dict={0: 'Type0', 1: 'Type1', 2: 'Type2', 3: 'Type3'}
    )