"""
测井数据可视化模块
功能：交互式测井数据显示界面，支持缩放和深度导航
作者：优化版本
日期：2024年
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from typing import List, Dict, Tuple, Optional, Union
from src_logging.curve_preprocess import get_resolution_by_depth

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False


class WellLogVisualizer:
    """
    测井数据可视化类
    主要功能：
    - 多曲线测井数据显示
    - 分类数据面板显示
    - 鼠标滚轮深度导航
    - 滑动条窗口大小控制
    - 交互式图例显示
    """

    # 类常量定义
    DEFAULT_COLORS = ['#FF0000', '#00FF00', '#0000FF', '#00FFFF', '#FF00FF',
                      '#8000FF', '#00FF80', '#FF0080', '#FFA500', '#FFFF00']

    # 布局参数
    LAYOUT_CONFIG = {
        'left_margin': 0.04,
        'right_margin': 0.96,
        'top_margin': 0.95,
        'legend_bottom_margin': 0.065,
        'no_legend_bottom_margin': 0.03,
        'title_margin': 0.001,
        'title_height': 0.04,
        'slider_width': 0.045
    }

    def __init__(self):
        """初始化可视化器"""
        self.fig = None
        self.axs = None
        self.data = None
        self.depth_col = None
        self.curve_cols = None
        self.type_cols = None

        # 状态变量
        self.depth_min = 0
        self.depth_max = 0
        self.depth_position = 0  # 当前显示窗口的顶部深度
        self.window_size = 0  # 当前显示窗口的大小
        self.resolution = 0  # 深度分辨率

        # 图形对象
        self.window_size_slider = None
        self.plots = []
        self.class_axes = []
        self.litho_width_config = {}

    def _validate_data(self, data: pd.DataFrame, depth_col: str,
                       curve_cols: List[str], type_cols: List[str]) -> None:
        """
        验证输入数据的完整性

        Args:
            data: 测井数据DataFrame
            depth_col: 深度列名
            curve_cols: 曲线列名列表
            type_cols: 分类数据列名列表

        Raises:
            ValueError: 当缺少必要列时
        """
        required_cols = [depth_col] + curve_cols + type_cols
        missing_cols = set(required_cols) - set(data.columns)

        if missing_cols:
            raise ValueError(f"数据缺少必要列: {missing_cols}")

    def _setup_layout(self, figure: Optional[plt.Figure], n_plots: int,
                      has_legend: bool) -> None:
        """
        设置图形布局

        Args:
            figure: 现有的matplotlib图形对象
            n_plots: 子图数量
            has_legend: 是否显示图例
        """
        # 创建或使用现有图形
        if figure is None:
            self.fig, self.axs = plt.subplots(
                1, n_plots,
                figsize=(12, 10),
                sharey=True,
                gridspec_kw={'wspace': 0.0}
            )
        else:
            self.fig = figure
            self.axs = self.fig.subplots(1, n_plots, sharey=True,
                                         gridspec_kw={'wspace': 0.0})

        # 计算底部边距
        bottom_margin = (self.LAYOUT_CONFIG['legend_bottom_margin'] if has_legend
                         else self.LAYOUT_CONFIG['no_legend_bottom_margin'])

        # 调整布局
        plt.subplots_adjust(
            left=self.LAYOUT_CONFIG['left_margin'],
            right=self.LAYOUT_CONFIG['right_margin'],
            bottom=bottom_margin,
            top=self.LAYOUT_CONFIG['top_margin'],
            wspace=0.0
        )

    def _create_window_size_slider(self, has_legend: bool) -> None:
        """
        创建窗口大小控制滑动条

        Args:
            has_legend: 是否显示图例，用于计算滑动条位置
        """
        # 计算滑动条位置
        bottom_margin = (self.LAYOUT_CONFIG['legend_bottom_margin'] if has_legend
                         else self.LAYOUT_CONFIG['no_legend_bottom_margin'])
        slider_height = self.LAYOUT_CONFIG['top_margin'] - bottom_margin

        ax_slider = plt.axes([
            self.LAYOUT_CONFIG['right_margin'],
            bottom_margin,
            self.LAYOUT_CONFIG['slider_width'],
            slider_height
        ])

        # 计算窗口大小范围
        depth_range = self.depth_max - self.depth_min
        min_window_size = depth_range * 0.01  # 最小窗口大小1%
        max_window_size = depth_range  # 最大窗口大小100%

        # 创建滑动条
        self.window_size_slider = Slider(
            ax=ax_slider,
            label='',
            valmin=min_window_size,
            valmax=max_window_size,
            valinit=self.window_size,
            valstep=(max_window_size - min_window_size) / 100,
            orientation='vertical'
        )

        # 添加滑动条标签
        ax_slider.text(
            x=0.5, y=0.5,
            s='Window Size(m)',
            rotation=270,
            ha='center', va='center',
            transform=ax_slider.transAxes,
            fontsize=10
        )

    def _add_title_box(self, ax: plt.Axes, title: str, color: str, index: int) -> None:
        """
        为每个子图添加标题框

        Args:
            ax: 子图对象
            title: 标题文本
            color: 标题颜色
            index: 子图索引
        """
        orig_pos = ax.get_position()
        title_bbox = [
            orig_pos.x0 + self.LAYOUT_CONFIG['title_margin'],
            orig_pos.y0 + orig_pos.height + self.LAYOUT_CONFIG['title_margin'],
            orig_pos.width - 2 * self.LAYOUT_CONFIG['title_margin'],
            self.LAYOUT_CONFIG['title_height']
        ]

        # 添加标题背景矩形
        title_rect = Rectangle(
            (title_bbox[0], title_bbox[1]),
            title_bbox[2], title_bbox[3],
            transform=self.fig.transFigure,
            facecolor='#f5f5f5',
            edgecolor='#aaaaaa',
            linewidth=1,
            clip_on=False,
            zorder=10
        )
        self.fig.add_artist(title_rect)

        # 添加标题文本
        title_text = Text(
            title_bbox[0] + title_bbox[2] / 2,
            title_bbox[1] + title_bbox[3] / 2,
            title,
            fontsize=12,
            fontweight='bold',
            color=color,
            ha='center', va='center',
            transform=self.fig.transFigure,
            clip_on=False,
            zorder=11
        )
        self.fig.add_artist(title_text)

    def _calculate_curve_limits(self, curve_data: pd.Series) -> Tuple[float, float]:
        """
        计算曲线数据的显示范围

        Args:
            curve_data: 曲线数据序列

        Returns:
            Tuple[float, float]: (最小值, 最大值)
        """
        # 过滤异常值
        valid_data = curve_data[(curve_data >= -99) & (curve_data <= 999)]

        if valid_data.empty:
            return curve_data.min(), curve_data.max()

        min_val = valid_data.min()
        max_val = valid_data.max()

        # 应用智能缩放
        if min_val >= 0:
            min_val *= 0.95  # 正值范围缩小5%
        else:
            min_val *= 1.05  # 负值范围扩大5%

        if max_val >= 0:
            max_val *= 1.05  # 正值范围扩大5%
        else:
            max_val *= 0.95  # 负值范围缩小5%

        # 处理特殊情况
        if abs(min_val - max_val) < 1e-5:
            min_val -= 1
            max_val += 1

        return min_val, max_val

    def _plot_curves(self, colors: List[str]) -> None:
        """
        绘制测井曲线

        Args:
            colors: 颜色列表
        """
        for i, col in enumerate(self.curve_cols):
            ax = self.axs[i]

            # 添加标题框
            self._add_title_box(ax, col, colors[i % len(colors)], i)

            # 绘制曲线
            line, = ax.plot(
                self.data[col],
                self.data[self.depth_col],
                color=colors[i % len(colors)],
                lw=1.0,
                linestyle='-'
            )
            self.plots.append(line)

            # 设置坐标轴
            min_val, max_val = self._calculate_curve_limits(self.data[col])
            ax.set_xlim(min_val, max_val)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            ax.set_title(col, fontsize=10, pad=5)

            # 设置Y轴（只在第一个子图显示）
            if i == 0:
                ax.set_ylim([self.depth_min, self.depth_max])
                ax.yaxis.set_major_locator(plt.MaxNLocator(10))
                ax.tick_params(axis='y', labelsize=8, rotation=90, pad=1)
            else:
                ax.tick_params(left=False, labelleft=False)

    def _plot_classification(self, ax: plt.Axes, depth: float, litho: float) -> None:
        """
        在分类面板中绘制单个分类矩形

        Args:
            ax: 子图对象
            depth: 深度值
            litho: 岩性分类值
        """
        if math.isnan(litho) or litho < 0:
            return

        litho_int = int(litho)
        xmax = self.litho_width_config.get(litho_int, 0.1)

        ax.axhspan(
            depth - self.resolution / 2,
            depth + self.resolution / 2,
            xmin=0, xmax=xmax,
            facecolor=self.DEFAULT_COLORS[litho_int % len(self.DEFAULT_COLORS)],
            edgecolor='none',
            linewidth=0,
            clip_on=False
        )

    def _plot_class_panels(self) -> None:
        """绘制分类数据面板"""
        if not self.type_cols:
            return

        for i, col in enumerate(self.type_cols):
            ax_idx = len(self.curve_cols) + i
            ax = self.axs[ax_idx]
            self.class_axes.append(ax)

            # 添加标题框
            self._add_title_box(ax, col, '#222222', ax_idx)

            # 绘制分类数据
            for depth, litho in zip(self.data[self.depth_col], self.data[col]):
                self._plot_classification(ax, depth, litho)

            # 设置坐标轴
            ax.set_xticks([])
            ax.set_title(col, fontsize=10, pad=5)
            ax.tick_params(left=False, labelleft=False)
            ax.invert_yaxis()

    def _setup_litho_width_config(self) -> None:
        """设置岩性分类宽度配置"""
        if not self.type_cols:
            self.litho_width_config = {}
            return

        # 获取所有唯一的分类值
        type_data = self.data[self.type_cols].values
        unique_types = np.unique(type_data[~np.isnan(type_data)])
        type_count = len(unique_types)

        # 配置每个分类的显示宽度
        for i, litho_type in enumerate(sorted(unique_types)):
            if type_count == 1:
                self.litho_width_config[int(litho_type)] = 1.0
            else:
                self.litho_width_config[int(litho_type)] = (i + 1) / type_count

    def _on_window_size_change(self, val: float) -> None:
        """
        窗口大小改变事件处理

        Args:
            val: 新的窗口大小值
        """
        self.window_size = val
        # 确保深度位置在有效范围内
        if self.depth_position > self.depth_max - self.window_size:
            self.depth_position = self.depth_max - self.window_size
        self._update_display()

    def _on_scroll(self, event) -> None:
        """
        鼠标滚轮事件处理

        Args:
            event: 鼠标事件对象
        """
        # 计算滚动步长（窗口大小的10%）
        step = self.window_size * 0.1

        # 根据滚动方向调整深度位置
        if event.button == 'up':  # 向上滚动
            self.depth_position -= step
        elif event.button == 'down':  # 向下滚动
            self.depth_position += step

        # 确保深度位置在有效范围内
        self.depth_position = np.clip(
            self.depth_position,
            self.depth_min,
            self.depth_max - self.window_size
        )

        self._update_display()

    def _update_display(self) -> None:
        """更新显示内容"""
        # 计算显示范围
        top = self.depth_position
        bottom = self.depth_position + self.window_size

        # 更新所有子图深度范围
        for ax in self.axs:
            ax.set_ylim(bottom, top)

        # 更新分类面板显示
        self._update_class_panels(top, bottom)

        # 更新深度指示器
        self._update_depth_indicator(top, bottom)

        # 重绘图形
        self.fig.canvas.draw_idle()

    def _update_class_panels(self, top: float, bottom: float) -> None:
        """
        更新分类面板显示

        Args:
            top: 显示范围顶部深度
            bottom: 显示范围底部深度
        """
        if not self.type_cols:
            return

        # 获取可见范围内的数据
        visible_data = self.data[
            (self.data[self.depth_col] >= top) &
            (self.data[self.depth_col] <= bottom)
            ]

        for i, col in enumerate(self.type_cols):
            ax = self.class_axes[i]
            ax.clear()
            ax.set_xticks([])
            ax.set_title(col, fontsize=10, pad=5)

            # 重新绘制可见的分类数据
            for depth, litho in zip(visible_data[self.depth_col], visible_data[col]):
                self._plot_classification(ax, depth, litho)

    def _update_depth_indicator(self, top: float, bottom: float) -> None:
        """
        更新深度位置指示器

        Args:
            top: 顶部深度
            bottom: 底部深度
        """
        indicator_text = f"当前深度: {top:.1f} - {bottom:.1f} m"

        if hasattr(self.fig, 'depth_indicator'):
            self.fig.depth_indicator.set_text(indicator_text)
        else:
            self.fig.depth_indicator = self.fig.text(
                0.5, 0.97,
                indicator_text,
                ha='center', va='top',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2')
            )

    def _create_legend(self, legend_dict: Dict) -> None:
        """
        创建图例

        Args:
            legend_dict: 图例映射字典
        """
        if not legend_dict:
            return

        n_items = len(legend_dict)
        legend_height = 0.02
        legend_width = 0.8

        # 创建图例坐标轴
        legend_ax = plt.axes([0.1, 0.01, legend_width, legend_height], frameon=False)
        legend_ax.set_axis_off()

        # 创建图例句柄和标签
        handles = []
        labels = []
        sorted_keys = sorted(legend_dict.keys())

        for key in sorted_keys:
            patch = Rectangle((0, 0), 1, 1,
                              facecolor=self.DEFAULT_COLORS[key % len(self.DEFAULT_COLORS)],
                              edgecolor='none')
            handles.append(patch)
            labels.append(legend_dict[key])

        # 创建图例
        legend = legend_ax.legend(
            handles, labels,
            loc='center',
            ncol=n_items,
            frameon=True,
            framealpha=0.8,
            fancybox=True,
            fontsize=10,
            handlelength=1.5,
            handleheight=1.5,
            handletextpad=0.5,
            borderpad=0.5,
            columnspacing=1.0
        )

        # 设置图例样式
        frame = legend.get_frame()
        frame.set_facecolor('#f8f8f8')
        frame.set_edgecolor('#aaaaaa')

    def visualize(self,
                  data: pd.DataFrame,
                  depth_col: str = 'Depth',
                  curve_cols: List[str] = None,
                  type_cols: List[str] = None,
                  figsize: Tuple[float, float] = (12, 10),
                  colors: List[str] = None,
                  legend_dict: Dict = None,
                  figure: Optional[plt.Figure] = None) -> None:
        """
        主可视化函数

        Args:
            data: 测井数据DataFrame
            depth_col: 深度列名，默认为'Depth'
            curve_cols: 曲线列名列表，默认为['L1', 'L2', 'L3', 'L4']
            type_cols: 分类数据列名列表，默认为空列表
            figsize: 图形大小，默认为(12, 10)
            colors: 颜色列表，默认为预定义颜色
            legend_dict: 图例映射字典，默认为空字典
            figure: 现有图形对象，默认为None（创建新图形）
        """
        # 参数默认值处理
        if curve_cols is None:
            curve_cols = ['L1', 'L2', 'L3', 'L4']
        if type_cols is None:
            type_cols = []
        if colors is None:
            colors = self.DEFAULT_COLORS
        if legend_dict is None:
            legend_dict = {}

        # 数据验证和初始化
        self._validate_data(data, depth_col, curve_cols, type_cols)
        self.data = data.sort_values(depth_col).reset_index(drop=True)
        self.depth_col = depth_col
        self.curve_cols = curve_cols
        self.type_cols = type_cols

        # 计算深度范围
        self.depth_min = self.data[depth_col].min()
        self.depth_max = self.data[depth_col].max()
        self.window_size = (self.depth_max - self.depth_min) * 1.0  # 初始窗口大小
        self.depth_position = self.depth_min
        self.resolution = get_resolution_by_depth(self.data[depth_col].dropna().values)

        # 设置图形布局
        n_plots = len(curve_cols) + len(type_cols)
        self._setup_layout(figure, n_plots, bool(legend_dict))

        # 创建滑动条
        self._create_window_size_slider(bool(legend_dict))

        # 设置岩性分类配置
        self._setup_litho_width_config()

        # 绘制曲线和分类面板
        self._plot_curves(colors)
        self._plot_class_panels()

        # 绑定事件
        self.window_size_slider.on_changed(self._on_window_size_change)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

        # 创建图例
        self._create_legend(legend_dict)

        # 初始显示
        self._update_display()
        plt.show()


# 兼容旧接口的函数
def visualize_well_logs(data: pd.DataFrame,
                        depth_col: str = 'Depth',
                        curve_cols: List[str] = None,
                        type_cols: List[str] = None,
                        figsize: Tuple[float, float] = (12, 10),
                        colors: List[str] = None,
                        legend_dict: Dict = None,
                        figure: Optional[plt.Figure] = None) -> None:
    """
    测井数据可视化接口 - 兼容旧版本

    参数说明详见WellLogVisualizer.visualize方法
    """
    visualizer = WellLogVisualizer()
    visualizer.visualize(
        data=data,
        depth_col=depth_col,
        curve_cols=curve_cols,
        type_cols=type_cols,
        figsize=figsize,
        colors=colors,
        legend_dict=legend_dict,
        figure=figure
    )


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    sample_data = pd.DataFrame(
        np.hstack([
            np.linspace(300, 400, 1000).reshape(-1, 1),
            np.random.randn(1000, 4),
            np.random.choice([0, 1, 2, 3], size=(1000, 4))
        ]),
        columns=['Depth', 'L1', 'L2', 'L3', 'L4', 'Type2', 'Type4', 'Type3', 'Type5']
    )

    depth_FMI = sample_data.Depth.values
    print(depth_FMI.shape, depth_FMI[0], depth_FMI[-1])
    FMI_DYNA = np.random.randint(0, 256, size=[depth_FMI.shape[0], 256], dtype=np.uint8)
    FMI_STAT = np.random.randint(0, 256, size=[depth_FMI.shape[0], 256], dtype=np.uint8)
    print(FMI_DYNA.shape, FMI_STAT.shape) # (1000, 256) (1000, 256)

    # 使用类接口
    visualizer = WellLogVisualizer()
    visualizer.visualize(
        data=sample_data,
        depth_col='Depth',
        curve_cols=['L1', 'L2', 'L3', 'L4'],
        type_cols=['Type2', 'Type4', 'Type3', 'Type5'],
        legend_dict={0: 'Type0', 1: 'Type1', 2: 'Type2', 3: 'Type3'},
        FMI_dict = {'depth':depth_FMI, 'image_data':[FMI_DYNA, FMI_STAT]},
        depth_limit_config=[depth_FMI[0], depth_FMI[-1]],
    )

    # 或使用函数接口（兼容旧版本）
    # visualize_well_logs(
    #     data=sample_data,
    #     depth_col='Depth',
    #     curve_cols=['L1', 'L2', 'L3', 'L4'],
    #     type_cols=['Type2', 'Type4', 'Type3', 'Type5'],
    #     legend_dict={0: 'Type0', 1: 'Type1', 2: 'Type2', 3: 'Type3'}
    # )