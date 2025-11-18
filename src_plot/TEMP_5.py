import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.collections import PolyCollection
from typing import List, Dict, Tuple, Optional, Any, Union
from collections import OrderedDict
import time
import logging
import cv2
# 导入分辨率计算模块
from src_logging.curve_preprocess import get_resolution_by_depth

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WellLogVisualizer:
    """
    测井数据可视化类 - 优化缓存版本

    主要优化点：
    1. 简化缓存机制，移除冗余缓存层
    2. 使用单一高效的数据缓存策略
    3. 删除不必要的预渲染逻辑
    4. 精简代码结构，提高可读性
    """

    # 默认曲线颜色序列
    DEFAULT_CURVE_COLORS = [
        '#FF0000', '#00FF00', '#0000FF', '#00FFFF', '#FF00FF', '#8000FF', '#00FF80', '#FF0080', '#FFA500', '#FFFF00'
    ]

    # 布局配置常量
    LAYOUT_CONFIG = {
        'left_margin': 0.05,
        'right_margin': 0.98,
        'top_margin': 0.96,
        'legend_bottom_margin': 0.065,
        'no_legend_bottom_margin': 0.05,
        'title_margin': 0.001,
        'title_height': 0.04,
        'slider_width': 0.02,
        'fmi_panel_width': 0.1,
        'min_window_size_ratio': 0.01,
        'scroll_step_ratio': 0.1
    }

    def __init__(self, performance_config: Dict[str, Any] = None):
        """
        初始化测井数据可视化器

        Args:
            performance_config: 性能配置字典
        """
        # 性能配置
        self.performance_config = {
            'cache_enabled': True,
            'max_cache_size': 100,
            'cache_margin_ratio': 0.5  # 缓存扩展比例
        }

        if performance_config:
            self.performance_config.update(performance_config)

        # 数据相关属性
        self.data: Optional[pd.DataFrame] = None
        self.depth_col: Optional[str] = None
        self.curve_cols: Optional[List[str]] = None
        self.type_cols: Optional[List[str]] = None
        self.fmi_dict: Optional[Dict[str, Any]] = None
        self.depth_limit_config: Optional[List[float]] = None

        # 显示状态属性
        self.depth_min: float = 0.0
        self.depth_max: float = 0.0
        self.depth_position: float = 0.0
        self.window_size: float = 0.0
        self.resolution: float = 0.0
        self.litho_width_config: Dict[int, float] = {}

        # 图形对象属性
        self.fig: Optional[plt.Figure] = None
        self.axs: Optional[List[plt.Axes]] = None
        self.window_size_slider: Optional[Slider] = None
        self.plots: List[Any] = []
        self.class_axes: List[plt.Axes] = []
        self.fmi_axes: List[plt.Axes] = []
        self.fmi_images: List[Any] = []

        # 缓存属性 - 简化后的单一缓存
        self._data_cache: OrderedDict = OrderedDict()  # LRU缓存
        self._cache_stats = {'hits': 0, 'misses': 0, 'total_renders': 0}

        logger.info("WellLogVisualizer初始化完成，缓存%s",
                    "启用" if self.performance_config['cache_enabled'] else "禁用")

    def _validate_input_parameters(self, data: pd.DataFrame, depth_col: str,
                                   curve_cols: List[str], type_cols: List[str]) -> None:
        """验证输入参数的有效性"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data参数必须是pandas DataFrame类型")

        if not depth_col or not isinstance(depth_col, str):
            raise ValueError("depth_col必须是非空字符串")

        if not curve_cols or not isinstance(curve_cols, list):
            raise ValueError("curve_cols必须是非空列表")

        if type_cols is None:
            type_cols = []
        elif not isinstance(type_cols, list):
            raise ValueError("type_cols必须是列表或None")

        # 检查列名存在性
        required_cols = [depth_col] + curve_cols + type_cols
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"数据中缺少以下必要列: {missing_cols}")

        if data.empty:
            raise ValueError("输入数据不能为空DataFrame")

        if data[depth_col].isna().all():
            raise ValueError("深度列数据全部为空")

        logger.info("输入参数验证通过")

    def _validate_fmi_data(self, fmi_dict: Optional[Dict[str, Any]]) -> None:
        """验证FMI图像数据结构"""
        if fmi_dict is None:
            logger.info("无FMI数据")
            return

        required_keys = ['depth', 'image_data']
        for key in required_keys:
            if key not in fmi_dict:
                raise ValueError(f"FMI字典缺少必要键: {key}")

        if not isinstance(fmi_dict['image_data'], list) or len(fmi_dict['image_data']) == 0:
            raise ValueError("FMI图像数据必须是非空列表")

        depth_data = fmi_dict['depth']
        for i, image_data in enumerate(fmi_dict['image_data']):
            if not isinstance(image_data, np.ndarray):
                raise ValueError(f"FMI图像数据[{i}]必须是numpy数组")
            if len(depth_data) != image_data.shape[0]:
                raise ValueError(f"FMI图像数据[{i}]深度维度不匹配")

        # 自动生成标题（如果未提供）
        if 'title' not in fmi_dict or fmi_dict['title'] is None:
            fmi_dict['title'] = [f'FMI_{i}' for i in range(len(fmi_dict['image_data']))]

        logger.info("FMI数据验证通过，包含%d个图像", len(fmi_dict['image_data']))

    def _setup_depth_limits(self, depth_limit_config: Optional[List[float]]) -> None:
        """设置深度显示范围"""
        full_depth_min = self.data[self.depth_col].min()
        full_depth_max = self.data[self.depth_col].max()

        if depth_limit_config is not None:
            if len(depth_limit_config) != 2:
                raise ValueError("depth_limit_config必须是包含2个元素的列表")

            config_min, config_max = depth_limit_config
            if config_min >= config_max:
                raise ValueError(f"深度范围配置无效: min={config_min} >= max={config_max}")

            self.depth_min = max(full_depth_min, config_min)
            self.depth_max = min(full_depth_max, config_max)

            if self.depth_min >= self.depth_max:
                raise ValueError("深度范围配置与数据范围无重叠")
        else:
            self.depth_min = full_depth_min
            self.depth_max = full_depth_max

        # 过滤数据
        original_size = len(self.data)
        mask = (self.data[self.depth_col] >= self.depth_min) & (self.data[self.depth_col] <= self.depth_max)
        self.data = self.data[mask].reset_index(drop=True)

        logger.info("深度范围设置: %.2f - %.2f, 数据点: %d -> %d",
                    self.depth_min, self.depth_max, original_size, len(self.data))

    def _setup_lithology_width_config(self) -> None:
        """设置岩性分类显示宽度配置"""
        if not self.type_cols:
            self.litho_width_config = {}
            return

        # 收集所有分类值
        all_type_values = []
        for col in self.type_cols:
            valid_data = self.data[col].dropna()
            all_type_values.extend(valid_data.unique())

        if not all_type_values:
            self.litho_width_config = {}
            return

        # 为每个分类类型分配宽度
        unique_types = np.unique(all_type_values)
        for i, litho_type in enumerate(sorted(unique_types)):
            litho_int = int(litho_type)
            self.litho_width_config[litho_int] = (i + 1) / len(unique_types)

        logger.info("岩性宽度配置: %s", self.litho_width_config)

    def _get_cached_data(self, depth_range: Tuple[float, float]) -> Optional[pd.DataFrame]:
        """
        从缓存获取数据 - 核心缓存方法

        Args:
            depth_range: 请求的深度范围 (start, end)

        Returns:
            缓存数据或None（未命中时）
        """
        if not self.performance_config['cache_enabled']:
            return None

        # 生成缓存键
        cache_key = f"{depth_range[0]:.2f}_{depth_range[1]:.2f}"

        if cache_key in self._data_cache:
            # 缓存命中：移动数据到末尾（标记为最近使用）
            self._data_cache.move_to_end(cache_key)
            self._cache_stats['hits'] += 1
            logger.debug("缓存命中: %s", cache_key)
            return self._data_cache[cache_key]

        # 缓存未命中
        self._cache_stats['misses'] += 1
        return None

    def _set_cached_data(self, depth_range: Tuple[float, float], data: pd.DataFrame) -> None:
        """
        设置缓存数据

        Args:
            depth_range: 缓存的深度范围
            data: 要缓存的数据
        """
        if not self.performance_config['cache_enabled']:
            return

        cache_key = f"{depth_range[0]:.2f}_{depth_range[1]:.2f}"

        # 添加数据到缓存
        self._data_cache[cache_key] = data
        self._data_cache.move_to_end(cache_key)

        # 检查缓存大小，移除最旧数据
        if len(self._data_cache) > self.performance_config['max_cache_size']:
            oldest_key = next(iter(self._data_cache))
            del self._data_cache[oldest_key]
            logger.debug("缓存清理: 移除%s", oldest_key)

    def _get_visible_data(self, top_depth: float, bottom_depth: float) -> pd.DataFrame:
        """
        获取可见深度范围内的数据（带缓存优化）

        策略：优先检查缓存，如果未命中则查询数据并缓存扩展范围
        """
        # 首先尝试从缓存获取
        cached_data = self._get_cached_data((top_depth, bottom_depth))
        if cached_data is not None:
            return cached_data

        # 缓存未命中，查询数据
        logger.debug("缓存未命中，查询数据: [%.2f-%.2f]", top_depth, bottom_depth)
        visible_data = self.data[
            (self.data[self.depth_col] >= top_depth) &
            (self.data[self.depth_col] <= bottom_depth)
            ]

        # 缓存扩展范围的数据（提高后续缓存命中率）
        cache_margin = self.window_size * self.performance_config['cache_margin_ratio']
        cache_top = max(self.depth_min, top_depth - cache_margin)
        cache_bottom = min(self.depth_max, bottom_depth + cache_margin)

        extended_data = self.data[
            (self.data[self.depth_col] >= cache_top) &
            (self.data[self.depth_col] <= cache_bottom)
            ]

        # 缓存扩展范围数据
        self._set_cached_data((cache_top, cache_bottom), extended_data)

        return visible_data

    def _calculate_subplot_count(self) -> int:
        """计算子图总数"""
        n_plots = len(self.curve_cols) + len(self.type_cols)
        if self.fmi_dict and self.fmi_dict.get('image_data'):
            n_plots += len(self.fmi_dict['image_data'])

        return n_plots

    def _setup_figure_layout(self, figure: Optional[plt.Figure], n_plots: int,
                             has_legend: bool, figsize: Tuple[float, float]) -> None:
        """设置图形布局"""
        if n_plots == 0:
            raise ValueError("没有可显示的子图内容")

        # 创建或重用图形
        if figure is None:
            self.fig, self.axs = plt.subplots(1, n_plots, figsize=figsize, sharey=True, gridspec_kw={'wspace': 0.0})
        else:
            self.fig = figure
            self.fig.clear()
            self.axs = self.fig.subplots(1, n_plots, sharey=True, gridspec_kw={'wspace': 0.0})

        # 确保axs为列表
        if n_plots == 1:
            self.axs = [self.axs]

        # 调整布局
        bottom_margin = (self.LAYOUT_CONFIG['legend_bottom_margin'] if has_legend
                         else self.LAYOUT_CONFIG['no_legend_bottom_margin'])

        plt.subplots_adjust(
            left=self.LAYOUT_CONFIG['left_margin'],
            right=self.LAYOUT_CONFIG['right_margin'],
            bottom=bottom_margin,
            top=self.LAYOUT_CONFIG['top_margin'],
            wspace=0.0
        )

    def _create_window_size_slider(self, has_legend: bool) -> None:
        """创建窗口大小滑动条"""
        bottom_margin = (self.LAYOUT_CONFIG['legend_bottom_margin'] if has_legend
                         else self.LAYOUT_CONFIG['no_legend_bottom_margin'])
        slider_height = self.LAYOUT_CONFIG['top_margin'] - bottom_margin

        slider_ax = plt.axes([
            self.LAYOUT_CONFIG['right_margin'],
            bottom_margin,
            self.LAYOUT_CONFIG['slider_width'],
            slider_height
        ])

        # 计算滑动条范围
        depth_range = self.depth_max - self.depth_min
        min_window_size = depth_range * self.LAYOUT_CONFIG['min_window_size_ratio']
        max_window_size = depth_range
        initial_window_size = depth_range * 0.5
        self.window_size = initial_window_size

        self.window_size_slider = Slider(
            ax=slider_ax,
            label='',
            valmin=min_window_size,
            valmax=max_window_size,
            valinit=initial_window_size,
            orientation='vertical'
        )

        # 添加滑动条标签
        slider_ax.text(0.5, 0.5, '窗口大小(m)', rotation=270, ha='center', va='center',
                       transform=slider_ax.transAxes, fontsize=8)

    def _create_title_box(self, ax: plt.Axes, title: str, color: str, index: int) -> None:
        """为子图创建标题框"""
        orig_pos = ax.get_position()

        title_bbox = [
            orig_pos.x0 + self.LAYOUT_CONFIG['title_margin'],
            orig_pos.y0 + orig_pos.height + self.LAYOUT_CONFIG['title_margin'],
            orig_pos.width - 2 * self.LAYOUT_CONFIG['title_margin'],
            self.LAYOUT_CONFIG['title_height']
        ]

        # 标题背景
        title_rect = Rectangle(
            (title_bbox[0], title_bbox[1]), title_bbox[2], title_bbox[3],
            transform=self.fig.transFigure,
            facecolor='#f5f5f5', edgecolor='#aaaaaa', linewidth=1,
            clip_on=False, zorder=10
        )
        self.fig.add_artist(title_rect)

        # 标题文本
        title_text = Text(
            title_bbox[0] + title_bbox[2] / 2, title_bbox[1] + title_bbox[3] / 2,
            title, fontsize=12, fontweight='bold', color=color,
            ha='center', va='center', transform=self.fig.transFigure,
            clip_on=False, zorder=11
        )
        self.fig.add_artist(title_text)

    def _calculate_curve_display_limits(self, curve_data: pd.Series) -> Tuple[float, float]:
        """计算曲线显示范围"""
        # 过滤异常值
        valid_mask = (curve_data > -999) & (curve_data < 999) & ~curve_data.isna()
        valid_data = curve_data[valid_mask]

        if valid_data.empty:
            min_val, max_val = curve_data.min(), curve_data.max()
        else:
            min_val, max_val = valid_data.min(), valid_data.max()

        # 处理常数数据
        if abs(max_val - min_val) < 1e-10:
            margin = abs(min_val) * 0.1 if min_val != 0 else 1.0
            min_val -= margin
            max_val += margin
        else:
            # 添加5%边距
            data_range = max_val - min_val
            margin = data_range * 0.05
            min_val = max(0, min_val - margin) if min_val >= 0 else min_val - margin
            max_val += margin

        return min_val, max_val

    def _plot_fmi_panel(self, ax: plt.Axes, image_data: np.ndarray, title: str, index: int) -> None:
        """绘制FMI图像面板"""
        self._create_title_box(ax, title, '#222222', index)
        fmi_depth = self.fmi_dict['depth']

        # 根据图像维度选择显示方法
        if len(image_data.shape) == 2:
            img = ax.imshow(image_data, aspect='auto', cmap='hot',
                            extent=[0, image_data.shape[1], fmi_depth[-1], fmi_depth[0]])
        elif len(image_data.shape) == 3 and image_data.shape[2] in [1, 3, 4]:
            display_data = image_data if image_data.shape[2] != 1 else image_data[:, :, 0]
            img = ax.imshow(display_data, aspect='auto',
                            extent=[0, image_data.shape[1], fmi_depth[-1], fmi_depth[0]])
        else:
            raise ValueError(f"不支持的图像维度: {image_data.shape}")

        self.fmi_images.append(img)

        # 设置坐标轴
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_xlabel('')

        if index == 0:  # 第一个面板显示Y轴
            ax.set_ylabel('深度 (m)', fontsize=10)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_ylim([self.depth_max, self.depth_min])
        else:
            ax.tick_params(left=False, labelleft=False)

        ax.invert_yaxis()

    def _plot_all_fmi_panels(self) -> None:
        """绘制所有FMI面板"""
        self.fmi_axes = []
        self.fmi_images = []

        if not self.fmi_dict:
            return

        for i, (image_data, title) in enumerate(zip(self.fmi_dict['image_data'], self.fmi_dict['title'])):
            ax = self.axs[i]
            self.fmi_axes.append(ax)
            self._plot_fmi_panel(ax, image_data, title, i)

    def _plot_curve_panel(self, ax: plt.Axes, curve_col: str, color: str, index: int) -> None:
        """绘制曲线面板"""
        self._create_title_box(ax, curve_col, color, index)

        # 绘制曲线
        line, = ax.plot(self.data[curve_col].values, self.data[self.depth_col].values,
                        color=color, linewidth=1.0, linestyle='-', label=curve_col)
        self.plots.append(line)

        # 设置显示范围
        min_val, max_val = self._calculate_curve_display_limits(self.data[curve_col])
        ax.set_xlim(min_val, max_val)

        # 设置坐标轴
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        # Y轴显示逻辑
        if not self.fmi_axes and index == 0:  # 无FMI时的第一个曲线面板
            ax.set_ylabel('深度 (m)', fontsize=10)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_ylim([self.depth_max, self.depth_min])
        else:
            ax.tick_params(left=False, labelleft=False)

    def _plot_all_curves(self, colors: List[str]) -> None:
        """绘制所有曲线面板"""
        self.plots = []
        start_index = len(self.fmi_axes)

        for i, col in enumerate(self.curve_cols):
            ax_index = start_index + i
            color = colors[i % len(colors)]
            self._plot_curve_panel(self.axs[ax_index], col, color, ax_index)

    def _batch_render_classification(self, ax: plt.Axes, class_col: str, visible_data: pd.DataFrame) -> None:
        """批量渲染分类数据（性能优化）"""
        if visible_data.empty:
            return

        class_groups = visible_data.groupby(class_col)
        vertices_list = []
        colors_list = []

        for class_val, group in class_groups:
            if pd.isna(class_val) or class_val < 0:
                continue

            class_int = int(class_val)
            xmax = self.litho_width_config.get(class_int, 0.1)
            color = self.DEFAULT_CURVE_COLORS[class_int % len(self.DEFAULT_CURVE_COLORS)]

            for depth in group[self.depth_col]:
                y_bottom = depth - self.resolution / 2
                y_top = depth + self.resolution / 2

                vertices = [[0, y_bottom], [xmax, y_bottom], [xmax, y_top], [0, y_top]]
                vertices_list.append(vertices)
                colors_list.append(color)

        if vertices_list:
            poly_collection = PolyCollection(vertices_list, facecolors=colors_list,
                                             edgecolors='none', linewidths=0)
            ax.add_collection(poly_collection)

    def _plot_classification_panel(self, ax: plt.Axes, class_col: str, index: int) -> None:
        """绘制分类数据面板（初始绘制）"""
        self._create_title_box(ax, class_col, '#222222', index)

        for depth, class_val in zip(self.data[self.depth_col], self.data[class_col]):
            if pd.isna(class_val) or class_val < 0:
                continue

            class_int = int(class_val)
            xmax = self.litho_width_config.get(class_int, 0.1)

            ax.axhspan(ymin=depth - self.resolution / 2, ymax=depth + self.resolution / 2,
                       xmin=0, xmax=xmax,
                       facecolor=self.DEFAULT_CURVE_COLORS[class_int % len(self.DEFAULT_CURVE_COLORS)],
                       edgecolor='none')

        ax.set_xticks([])
        ax.tick_params(left=False, labelleft=False)
        ax.invert_yaxis()

    def _plot_all_classification_panels(self) -> None:
        """绘制所有分类面板"""
        self.class_axes = []
        if not self.type_cols:
            return

        base_index = len(self.fmi_axes) + len(self.curve_cols)
        for i, col in enumerate(self.type_cols):
            ax_idx = base_index + i
            ax = self.axs[ax_idx]
            self.class_axes.append(ax)
            self._plot_classification_panel(ax, col, ax_idx)

    def _optimize_fmi_rendering(self) -> None:
        """FMI图像渲染优化"""
        if not self.fmi_dict:
            return

        for i, image_data in enumerate(self.fmi_dict['image_data']):
            if image_data.dtype != np.uint8:
                if image_data.max() > image_data.min():
                    normalized = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255
                    self.fmi_dict['image_data'][i] = normalized.astype(np.uint8)
                else:
                    self.fmi_dict['image_data'][i] = np.full_like(image_data, 128, dtype=np.uint8)

    def _create_legend_panel(self, legend_dict: Dict[int, str]) -> None:
        """创建图例面板"""
        if not legend_dict:
            return

        n_items = len(legend_dict)
        legend_height, legend_width = 0.02, min(0.8, n_items * 0.15)
        legend_ax = plt.axes([0.5 - legend_width / 2, 0.01, legend_width, legend_height], frameon=False)
        legend_ax.set_axis_off()

        # 准备图例句柄
        handles, labels = [], []
        for key in sorted(legend_dict.keys()):
            patch = Rectangle((0, 0), 1, 1,
                              facecolor=self.DEFAULT_CURVE_COLORS[key % len(self.DEFAULT_CURVE_COLORS)],
                              edgecolor='black', linewidth=0.5)
            handles.append(patch)
            labels.append(legend_dict[key])

        legend = legend_ax.legend(handles, labels, loc='center', ncol=min(n_items, 6),
                                  frameon=True, framealpha=0.9, fontsize=9)

        # 设置图例样式
        frame = legend.get_frame()
        frame.set_facecolor('#f8f8f8')
        frame.set_edgecolor('#666666')

    def _on_window_size_change(self, val: float) -> None:
        """窗口大小变化事件处理"""
        self.window_size = val
        max_valid_position = self.depth_max - self.window_size
        if self.depth_position > max_valid_position:
            self.depth_position = max_valid_position
        self._update_display()

    def _on_mouse_scroll(self, event) -> None:
        """鼠标滚轮事件处理"""
        if event.inaxes not in self.axs:
            return

        step = self.window_size * self.LAYOUT_CONFIG['scroll_step_ratio']
        if event.button == 'up':
            new_position = self.depth_position - step
        elif event.button == 'down':
            new_position = self.depth_position + step
        else:
            return

        self.depth_position = np.clip(new_position, self.depth_min, self.depth_max - self.window_size)
        self._update_display()

    def _update_display(self) -> None:
        """更新显示内容（核心刷新函数）"""
        start_time = time.time()
        self._cache_stats['total_renders'] += 1

        # 计算显示范围
        top_depth = self.depth_position
        bottom_depth = self.depth_position + self.window_size

        # 更新坐标轴范围
        for ax in self.axs:
            ax.set_ylim(bottom_depth, top_depth)

        # 更新各组件
        self._update_classification_panels(top_depth, bottom_depth)
        self._update_fmi_display(top_depth, bottom_depth)
        self._update_depth_indicator(top_depth, bottom_depth)

        # 重绘图形
        self.fig.canvas.draw_idle()

        # 记录性能
        render_time = (time.time() - start_time) * 1000
        logger.debug("渲染完成: %.1fms", render_time)

    def _update_classification_panels(self, top_depth: float, bottom_depth: float) -> None:
        """更新分类面板显示"""
        if not self.class_axes:
            return

        # 获取可见数据（使用缓存）
        visible_data = self._get_visible_data(top_depth, bottom_depth)

        for i, (ax, col) in enumerate(zip(self.class_axes, self.type_cols)):
            ax.clear()
            ax.set_xticks([])
            ax.tick_params(left=False, labelleft=False)
            ax.invert_yaxis()
            ax.set_ylim(bottom_depth, top_depth)

            # 批量渲染
            self._batch_render_classification(ax, col, visible_data)

    def _update_fmi_display(self, top_depth: float, bottom_depth: float) -> None:
        """更新FMI图像显示"""
        if not self.fmi_dict or not self.fmi_images:
            return

        fmi_depth = self.fmi_dict['depth']
        visible_indices = (fmi_depth >= top_depth) & (fmi_depth <= bottom_depth)

        if not np.any(visible_indices):
            return

        for img, image_data in zip(self.fmi_images, self.fmi_dict['image_data']):
            visible_data = image_data[visible_indices]
            if len(visible_data) > 0:
                if len(visible_data.shape) == 2:
                    img.set_data(visible_data)
                elif len(visible_data.shape) == 3:
                    display_data = visible_data if visible_data.shape[2] != 1 else visible_data[:, :, 0]
                    img.set_data(display_data)

                # 更新显示范围
                visible_depths = fmi_depth[visible_indices]
                if len(visible_depths) > 0:
                    img.set_extent([0, visible_data.shape[1], visible_depths[-1], visible_depths[0]])

    def _update_depth_indicator(self, top_depth: float, bottom_depth: float) -> None:
        """更新深度指示器"""
        indicator_text = (f"深度: {top_depth:.1f} - {bottom_depth:.1f} m | "
                          f"窗口: {self.window_size:.1f} m")

        if hasattr(self, '_depth_indicator'):
            self._depth_indicator.set_text(indicator_text)
        else:
            self._depth_indicator = self.fig.text(
                0.99, 0.01, indicator_text, ha='right', va='bottom', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.1')
            )

    def visualize(self,
                  data: pd.DataFrame,
                  depth_col: str = 'Depth',
                  curve_cols: List[str] = None,
                  type_cols: List[str] = None,
                  figsize: Tuple[float, float] = (12, 10),
                  colors: List[str] = None,
                  legend_dict: Dict[int, str] = None,
                  fmi_dict: Dict[str, Any] = None,
                  depth_limit_config: Optional[List[float]] = None,
                  figure: Optional[plt.Figure] = None) -> None:
        """主可视化函数"""
        try:
            logger.info("开始测井数据可视化")
            start_time = time.time()

            # 参数预处理
            if curve_cols is None:
                curve_cols = ['L1', 'L2', 'L3', 'L4']
            if type_cols is None:
                type_cols = []
            if colors is None:
                colors = self.DEFAULT_CURVE_COLORS
            if legend_dict is None:
                legend_dict = {}

            # 参数验证
            self._validate_input_parameters(data, depth_col, curve_cols, type_cols)
            self._validate_fmi_data(fmi_dict)

            # 数据初始化
            self.data = data.sort_values(depth_col).reset_index(drop=True)
            self.depth_col = depth_col
            self.curve_cols = curve_cols
            self.type_cols = type_cols
            self.fmi_dict = fmi_dict
            self.depth_limit_config = depth_limit_config

            # 数据预处理
            self._setup_depth_limits(depth_limit_config)
            self._setup_lithology_width_config()

            # 计算显示参数
            depth_range = self.depth_max - self.depth_min
            self.window_size = depth_range * 0.5
            self.depth_position = self.depth_min
            self.resolution = get_resolution_by_depth(self.data[depth_col].dropna().values)

            # 图形设置
            n_plots = self._calculate_subplot_count()
            has_legend = bool(legend_dict)
            self._setup_figure_layout(figure, n_plots, has_legend, figsize)
            self._create_window_size_slider(has_legend)

            # 优化和绘制
            self._optimize_fmi_rendering()
            self._plot_all_fmi_panels()
            self._plot_all_curves(colors)
            self._plot_all_classification_panels()

            # 交互功能
            self.window_size_slider.on_changed(self._on_window_size_change)
            self.fig.canvas.mpl_connect('scroll_event', self._on_mouse_scroll)
            self._create_legend_panel(legend_dict)

            # 初始显示
            self._update_display()

            total_time = time.time() - start_time
            logger.info("可视化完成，耗时: %.2fs", total_time)
            plt.show()

        except Exception as e:
            logger.error("可视化失败: %s", str(e))
            if self.fig:
                plt.close(self.fig)
            raise

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        total_attempts = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = (self._cache_stats['hits'] / total_attempts * 100) if total_attempts > 0 else 0

        stats = {
            'total_renders': self._cache_stats['total_renders'],
            'cache_hits': self._cache_stats['hits'],
            'cache_misses': self._cache_stats['misses'],
            'cache_hit_rate': hit_rate,
            'cache_size': len(self._data_cache),
            'imaging_curves_count': len(self.fmi_axes) if self.fmi_dict else 0,
            'regular_curves_count': len(self.curve_cols)
        }

        logger.info("性能统计: 渲染%d次, 缓存命中率%.1f%%, 缓存大小%d",
                    stats['total_renders'], hit_rate, stats['cache_size'])
        return stats

    def enable_performance_mode(self, enabled: bool = True) -> None:
        """启用/禁用性能模式"""
        self.performance_config['cache_enabled'] = enabled
        logger.info("性能模式%s", "启用" if enabled else "禁用")

    def clear_cache(self) -> None:
        """清空缓存"""
        self._data_cache.clear()
        self._cache_stats = {'hits': 0, 'misses': 0, 'total_renders': 0}
        logger.info("缓存已清空")

    def close(self) -> None:
        """关闭可视化器"""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
        self.clear_cache()
        logger.info("可视化器已关闭")


# 兼容性接口函数
@staticmethod
def visualize_well_logs(data: pd.DataFrame,
                        depth_col: str = 'Depth',
                        curve_cols: List[str] = None,
                        type_cols: List[str] = None,
                        figsize: Tuple[float, float] = (12, 10),
                        colors: List[str] = None,
                        legend_dict: Dict[int, str] = None,
                        fmi_dict: Dict[str, Any] = None,
                        depth_limit_config: Optional[List[float]] = None,
                        figure: Optional[plt.Figure] = None) -> None:
    """测井数据可视化兼容接口"""
    visualizer = WellLogVisualizer()
    visualizer.visualize(
        data=data, depth_col=depth_col, curve_cols=curve_cols, type_cols=type_cols,
        figsize=figsize, colors=colors, legend_dict=legend_dict, fmi_dict=fmi_dict,
        depth_limit_config=depth_limit_config, figure=figure
    )


# ==================== 使用示例和测试代码 ====================
if __name__ == "__main__":
    # 生成示例测试数据
    print("生成示例测井数据...")
    n_points = 1000

    # 创建示例测井数据
    sample_data = pd.DataFrame({
        'Depth': np.linspace(300, 400, n_points),
        'GR': np.random.normal(60, 15, n_points),  # 伽马射线
        'GR_X': np.random.normal(60, 15, n_points),  # 方向X伽马射线
        'GR_Y': np.random.normal(60, 15, n_points),  # 方向Y伽马射线
        'RT': np.random.lognormal(2, 0.3, n_points),  # 电阻率
        'RX': np.random.lognormal(2, 0.3, n_points),  # 电阻率
        'NPHI': np.random.uniform(0.15, 0.45, n_points),  # 中子孔隙度
        'RHOB': np.random.uniform(2.3, 2.7, n_points),  # 体积密度
        'LITHOLOGY': np.random.choice([0, 1, 2, 3], n_points, p=[0.4, 0.3, 0.2, 0.1]),
        'FACIES': np.random.choice([0, 1, 2], n_points, p=[0.5, 0.3, 0.2])
    })

    # 准备FMI示例数据
    depth_fmi = sample_data.Depth.values[200:-200]
    print(f"FMI深度数据形状: {depth_fmi.shape}, 起始深度: {depth_fmi[0]}, 结束深度: {depth_fmi[-1]}")

    # 生成示例FMI图像数据
    FMI_RAND = np.random.randint(0, 256, size=[20, 8], dtype=np.uint8)
    # FMI_STAT = np.random.randint(0, 256, size=[20, 8], dtype=np.uint8)

    # 将小图像扩展到完整深度范围
    fmi_dynamic = cv2.resize(FMI_RAND, (256, depth_fmi.shape[0]), interpolation=cv2.INTER_NEAREST)
    fmi_static = cv2.resize(FMI_RAND, (256, depth_fmi.shape[0]), interpolation=cv2.INTER_CUBIC)
    print(f"FMI动态图像形状: {fmi_dynamic.shape}, FMI静态图像形状: {fmi_static.shape}")

    # 使用类接口进行可视化
    print("创建可视化器...")
    visualizer = WellLogVisualizer()

    try:
        # 启用详细日志
        logging.getLogger().setLevel(logging.INFO)

        visualizer.visualize(
            data=sample_data,
            depth_col='Depth',
            curve_cols=['GR', 'RT', 'NPHI', 'RHOB'],
            type_cols=['LITHOLOGY', 'FACIES'],
            legend_dict={
                0: '砂岩',
                1: '页岩',
                2: '石灰岩',
                3: '白云岩'
            },
            fmi_dict={
                'depth': depth_fmi,
                'image_data': [fmi_dynamic, fmi_static],
                'title': ['FMI动态', 'FMI静态']
            },
            # fmi_dict=None,
            # depth_limit_config=[320, 380],  # 只显示320-380米段
            figsize=(12, 8)
        )

        # 显示性能统计
        stats = visualizer.get_performance_stats()
        print("性能统计:", stats)

    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 清理资源
        visualizer.close()