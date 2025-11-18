"""
测井数据可视化模块 - 重构优化版本
功能：交互式测井数据显示界面，支持缩放、深度导航和FMI图像显示
优化重点：代码结构清晰化、性能优化、错误处理增强、注释完善
作者：重构版本
日期：2024年
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.collections import PolyCollection
from typing import List, Dict, Tuple, Optional, Any
from collections import OrderedDict
import time
import logging
import cv2

from src_logging.curve_preprocess import get_resolution_by_depth

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False


class WellLogVisualizer:
    """
    测井数据可视化类 - 重构优化版本

    主要功能架构：
    1. 数据管理层：数据验证、预处理、缓存管理
    2. 界面管理层：布局设置、组件创建
    3. 绘图管理层：曲线、分类、FMI图像绘制
    4. 交互控制层：缩放、滚动、实时更新
    5. 性能优化层：缓存、预渲染、批量处理
    """

    # ==================== 类常量定义 ====================

    # 默认曲线颜色
    DEFAULT_CURVE_COLORS = [
        '#FF0000', '#00FF00', '#0000FF', '#00FFFF', '#FF00FF',
        '#8000FF', '#00FF80', '#FF0080', '#FFA500', '#FFFF00'
    ]

    # 布局配置
    LAYOUT_CONFIG = {
        'left_margin': 0.04,
        'right_margin': 0.96,
        'top_margin': 0.95,
        'legend_bottom_margin': 0.065,
        'no_legend_bottom_margin': 0.03,
        'title_margin': 0.001,
        'title_height': 0.04,
        'slider_width': 0.045,
        'fmi_panel_width': 0.1,
        'min_window_size_ratio': 0.01,
        'scroll_step_ratio': 0.1
    }

    # 性能优化配置
    PERFORMANCE_CONFIG = {
        'cache_enabled': True,
        'lazy_rendering': True,
        'batch_size': 1000,
        'max_cache_size': 100,
        'pre_render_margin': 0.3,
    }

    def __init__(self, performance_config: Dict = None):
        """
        初始化可视化器
        Args:
            performance_config: 性能配置字典，可覆盖默认配置
        """
        # 合并性能配置
        if performance_config:
            self.PERFORMANCE_CONFIG.update(performance_config)

        # 数据相关属性
        self.data = None
        self.depth_col = None
        self.curve_cols = None
        self.type_cols = None
        self.fmi_dict = None
        self.depth_limit_config = None

        # 显示状态属性
        self.depth_min = 0
        self.depth_max = 0
        self.depth_position = 0
        self.window_size = 0
        self.resolution = 0
        self.litho_width_config = {}

        # 图形对象属性
        self.fig = None
        self.axs = None
        self.window_size_slider = None
        self.plots = []
        self.class_axes = []
        self.fmi_axes = []
        self.fmi_images = []

        # 性能优化属性
        self._data_cache = OrderedDict()
        self._visible_data_cache = None
        self._cache_valid = False
        self._prerendered_data = None
        self._prerendered_range = (0, 0)
        self._render_times = []
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info("WellLogVisualizer初始化完成")

    # ==================== 数据管理层 ====================

    def _validate_input_parameters(self, data: pd.DataFrame, depth_col: str,
                                 curve_cols: List[str], type_cols: List[str]) -> None:
        """验证输入参数的完整性"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data参数必须是pandas DataFrame")

        if not depth_col or not isinstance(depth_col, str):
            raise ValueError("depth_col必须是非空字符串")

        if not curve_cols or not isinstance(curve_cols, list):
            raise ValueError("curve_cols必须是非空列表")

        if type_cols is None:
            type_cols = []

        # 列名存在性验证
        required_cols = [depth_col] + curve_cols + type_cols
        missing_cols = set(required_cols) - set(data.columns)

        if missing_cols:
            raise ValueError(f"数据中缺少以下必要列: {missing_cols}")

        if data.empty:
            raise ValueError("输入数据不能为空")

        if data[depth_col].isna().all():
            raise ValueError("深度列数据全部为空")

        logger.info("输入参数验证通过")

    def _validate_fmi_data(self, fmi_dict: Optional[Dict]) -> None:
        """验证FMI数据字典的结构和完整性"""
        if fmi_dict is None:
            logger.info("未提供FMI数据，跳过FMI显示")
            return

        required_keys = ['depth', 'image_data']
        for key in required_keys:
            if key not in fmi_dict:
                raise ValueError(f"FMI字典缺少必要键: {key}")

        if not isinstance(fmi_dict['image_data'], list):
            raise ValueError("FMI字典的image_data必须是列表")

        depth_data = fmi_dict['depth']
        image_data_list = fmi_dict['image_data']

        if len(image_data_list) == 0:
            raise ValueError("FMI图像数据列表不能为空")

        # 深度和图像数据形状匹配验证
        for i, image_data in enumerate(image_data_list):
            if not isinstance(image_data, np.ndarray):
                raise ValueError(f"FMI图像数据[{i}]必须是numpy数组")

            if len(depth_data) != image_data.shape[0]:
                raise ValueError(
                    f"FMI图像数据[{i}]的深度维度不匹配: "
                    f"深度长度={len(depth_data)}, 图像深度维度={image_data.shape[0]}"
                )

        # 标题数据处理
        if 'title' not in fmi_dict or fmi_dict['title'] is None:
            fmi_dict['title'] = [f'FMI_{i}' for i in range(len(image_data_list))]
        else:
            title_list = fmi_dict['title']
            if len(title_list) < len(image_data_list):
                for i in range(len(title_list), len(image_data_list)):
                    title_list.append(f'FMI_{i}')

        logger.info(f"FMI数据验证通过，包含{len(image_data_list)}个图像")

    def _setup_depth_limits(self, depth_limit_config: Optional[List[float]]) -> None:
        """设置深度显示范围限制"""
        full_depth_min = self.data[self.depth_col].min()
        full_depth_max = self.data[self.depth_col].max()

        if depth_limit_config is not None:
            if len(depth_limit_config) != 2:
                raise ValueError("depth_limit_config必须是包含2个元素的列表 [min_depth, max_depth]")

            config_min, config_max = depth_limit_config
            if config_min >= config_max:
                raise ValueError(f"深度范围配置无效: min={config_min} >= max={config_max}")

            self.depth_min = max(full_depth_min, config_min)
            self.depth_max = min(full_depth_max, config_max)

            if self.depth_min >= self.depth_max:
                raise ValueError(
                    f"计算后的深度范围无效: min={self.depth_min}, max={self.depth_max}"
                )
        else:
            self.depth_min = full_depth_min
            self.depth_max = full_depth_max

        # 过滤数据
        original_size = len(self.data)
        self.data = self.data[
            (self.data[self.depth_col] >= self.depth_min) &
            (self.data[self.depth_col] <= self.depth_max)
        ].reset_index(drop=True)

        logger.info(f"深度范围设置: {self.depth_min:.2f} - {self.depth_max:.2f}")
        logger.info(f"数据过滤: {original_size} -> {len(self.data)} 个数据点")

    def _identify_imaging_curves(self, curve_cols: List[str]) -> Tuple[List[str], List[str]]:
        """智能识别成像道曲线并分离"""
        imaging_curves = []
        regular_curves = []

        for col in curve_cols:
            is_imaging = any(keyword in col.upper() for keyword in self.IMAGING_LOG_KEYWORDS)
            if is_imaging:
                imaging_curves.append(col)
                logger.info(f"识别为成像道: {col}")
            else:
                regular_curves.append(col)

        return imaging_curves, regular_curves

    def _optimize_curve_order(self, curve_cols: List[str]) -> List[str]:
        """优化曲线显示顺序：成像道 -> 常规曲线"""
        imaging_curves, regular_curves = self._identify_imaging_curves(curve_cols)

        self.imaging_curve_cols = imaging_curves
        self.regular_curve_cols = regular_curves

        optimized_order = imaging_curves + regular_curves

        logger.info(f"曲线显示顺序优化: {len(imaging_curves)}成像道 + {len(regular_curves)}常规曲线")
        return optimized_order

    def _setup_lithology_width_config(self) -> None:
        """设置岩性分类的显示宽度配置"""
        if not self.type_cols:
            self.litho_width_config = {}
            return

        all_type_values = []
        for col in self.type_cols:
            valid_data = self.data[col].dropna()
            all_type_values.extend(valid_data.unique())

        if not all_type_values:
            self.litho_width_config = {}
            return

        unique_types = np.unique(all_type_values)
        type_count = len(unique_types)

        for i, litho_type in enumerate(sorted(unique_types)):
            litho_int = int(litho_type)
            if type_count == 1:
                self.litho_width_config[litho_int] = 1.0
            else:
                self.litho_width_config[litho_int] = (i + 1) / type_count

        logger.info(f"岩性分类宽度配置: {len(unique_types)}个分类类型")

    # ==================== 性能优化层 ====================

    def _get_cached_data(self, key: str, depth_range: Tuple[float, float]) -> Optional[Any]:
        """智能数据缓存获取"""
        if not self.PERFORMANCE_CONFIG['cache_enabled']:
            return None

        cache_key = f"{key}_{depth_range[0]:.2f}_{depth_range[1]:.2f}"

        if cache_key in self._data_cache:
            self._data_cache.move_to_end(cache_key)
            self._cache_hits += 1
            logger.debug(f"缓存命中: {cache_key}")
            return self._data_cache[cache_key]

        self._cache_misses += 1
        return None

    def _set_cached_data(self, key: str, depth_range: Tuple[float, float], data: Any) -> None:
        """设置缓存数据"""
        if not self.PERFORMANCE_CONFIG['cache_enabled']:
            return

        cache_key = f"{key}_{depth_range[0]:.2f}_{depth_range[1]:.2f}"

        self._data_cache[cache_key] = data
        self._data_cache.move_to_end(cache_key)

        if len(self._data_cache) > self.PERFORMANCE_CONFIG['max_cache_size']:
            oldest_key = next(iter(self._data_cache))
            del self._data_cache[oldest_key]
            logger.debug(f"清理过期缓存: {oldest_key}")

    def _get_visible_data(self, top_depth: float, bottom_depth: float) -> pd.DataFrame:
        """获取当前显示深度范围内的数据（带缓存优化）"""
        # 使用缓存避免重复计算
        if (self._cache_valid and self._visible_data_cache is not None and
                len(self._visible_data_cache) > 0):
            cache_top = self._visible_data_cache[self.depth_col].min()
            cache_bottom = self._visible_data_cache[self.depth_col].max()

            if cache_top <= top_depth and cache_bottom >= bottom_depth:
                visible_data = self._visible_data_cache[
                    (self._visible_data_cache[self.depth_col] >= top_depth) &
                    (self._visible_data_cache[self.depth_col] <= bottom_depth)
                ]
                return visible_data

        # 重新查询数据
        visible_data = self.data[
            (self.data[self.depth_col] >= top_depth) &
            (self.data[self.depth_col] <= bottom_depth)
        ]

        # 更新缓存
        cache_margin = self.window_size * 0.5
        cache_top = max(self.depth_min, top_depth - cache_margin)
        cache_bottom = min(self.depth_max, bottom_depth + cache_margin)

        self._visible_data_cache = self.data[
            (self.data[self.depth_col] >= cache_top) &
            (self.data[self.depth_col] <= cache_bottom)
        ]
        self._cache_valid = True

        return visible_data

    def _prerender_visible_data(self, current_top: float, current_bottom: float) -> None:
        """预渲染当前窗口前后范围的数据"""
        if not self.PERFORMANCE_CONFIG['lazy_rendering']:
            return

        margin = self.window_size * self.PERFORMANCE_CONFIG['pre_render_margin']
        prerender_top = max(self.depth_min, current_top - margin)
        prerender_bottom = min(self.depth_max, current_bottom + margin)

        if (prerender_top, prerender_bottom) != self._prerendered_range:
            self._prerendered_data = self._get_visible_data(prerender_top, prerender_bottom)
            self._prerendered_range = (prerender_top, prerender_bottom)
            logger.debug(f"预渲染数据范围: {prerender_top:.1f}-{prerender_bottom:.1f}")

    def _get_optimized_visible_data(self, top_depth: float, bottom_depth: float) -> pd.DataFrame:
        """优化后的可见数据获取（带预渲染支持）"""
        if (self._prerendered_data is not None and
                self._prerendered_range[0] <= top_depth and
                self._prerendered_range[1] >= bottom_depth):

            visible_data = self._prerendered_data[
                (self._prerendered_data[self.depth_col] >= top_depth) &
                (self._prerendered_data[self.depth_col] <= bottom_depth)
            ]
            logger.debug("使用预渲染数据")
            return visible_data

        return self._get_visible_data(top_depth, bottom_depth)

    # ==================== 界面管理层 ====================
    def _calculate_subplot_count(self) -> int:
        """计算需要的子图总数"""
        n_plots = len(self.curve_cols) + len(self.type_cols)

        if self.fmi_dict and self.fmi_dict.get('image_data'):
            n_plots += len(self.fmi_dict['image_data'])

        logger.info(f"子图数量计算: {n_plots}")
        return n_plots

    def _setup_figure_layout(self, figure: Optional[plt.Figure], n_plots: int,
                           has_legend: bool) -> None:
        """设置图形布局和子图排列"""
        if n_plots == 0:
            raise ValueError("没有可显示的子图内容")

        # 创建或使用现有图形
        if figure is None:
            self.fig, self.axs = plt.subplots(
                1, n_plots,
                figsize=(12, 10),
                sharey=True,
                gridspec_kw={'wspace': 0.0}
            )
            logger.info("创建新图形")
        else:
            self.fig = figure
            self.fig.clear()
            self.axs = self.fig.subplots(1, n_plots, sharey=True,
                                       gridspec_kw={'wspace': 0.0})
            logger.info("使用现有图形")

        # 确保axs总是为数组形式
        if n_plots == 1:
            self.axs = [self.axs]

        # 计算边距
        bottom_margin = (self.LAYOUT_CONFIG['legend_bottom_margin'] if has_legend
                        else self.LAYOUT_CONFIG['no_legend_bottom_margin'])

        # 应用布局调整
        plt.subplots_adjust(
            left=self.LAYOUT_CONFIG['left_margin'],
            right=self.LAYOUT_CONFIG['right_margin'],
            bottom=bottom_margin,
            top=self.LAYOUT_CONFIG['top_margin'],
            wspace=0.0
        )

        logger.info("图形布局设置完成")

    def _create_window_size_slider(self, has_legend: bool) -> None:
        """创建窗口大小控制滑动条"""
        bottom_margin = (self.LAYOUT_CONFIG['legend_bottom_margin'] if has_legend
                        else self.LAYOUT_CONFIG['no_legend_bottom_margin'])
        slider_height = self.LAYOUT_CONFIG['top_margin'] - bottom_margin

        slider_ax = plt.axes([
            self.LAYOUT_CONFIG['right_margin'],
            bottom_margin,
            self.LAYOUT_CONFIG['slider_width'],
            slider_height
        ])

        # 计算滑动条数值范围
        depth_range = self.depth_max - self.depth_min
        min_window_size = depth_range * self.LAYOUT_CONFIG['min_window_size_ratio']
        max_window_size = depth_range

        # 初始窗口大小
        initial_window_size = depth_range * 0.5
        self.window_size = initial_window_size

        # 创建滑动条
        self.window_size_slider = Slider(
            ax=slider_ax,
            label='',
            valmin=min_window_size,
            valmax=max_window_size,
            valinit=initial_window_size,
            valstep=(max_window_size - min_window_size) / 100,
            orientation='vertical'
        )

        # 添加滑动条标签
        slider_ax.text(
            x=0.5, y=0.5,
            s='窗口大小(m)',
            rotation=270,
            ha='center', va='center',
            transform=slider_ax.transAxes,
            fontsize=10
        )

        logger.info(f"窗口大小滑动条创建完成")

    def _create_title_box(self, ax: plt.Axes, title: str, color: str, index: int) -> None:
        """为每个子图创建统一的标题框"""
        orig_pos = ax.get_position()

        title_bbox = [
            orig_pos.x0 + self.LAYOUT_CONFIG['title_margin'],
            orig_pos.y0 + orig_pos.height + self.LAYOUT_CONFIG['title_margin'],
            orig_pos.width - 2 * self.LAYOUT_CONFIG['title_margin'],
            self.LAYOUT_CONFIG['title_height']
        ]

        # 创建标题背景矩形
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

        # 创建标题文本
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

        logger.debug(f"为子图{index}创建标题: {title}")

    # ==================== 绘图管理层 ====================
    def _calculate_curve_display_limits(self, curve_data: pd.Series) -> Tuple[float, float]:
        """智能计算曲线的显示范围"""
        # 过滤异常值
        valid_mask = (curve_data > -999) & (curve_data < 999) & ~curve_data.isna()
        valid_data = curve_data[valid_mask]

        if valid_data.empty:
            min_val, max_val = curve_data.min(), curve_data.max()
            logger.warning(f"曲线数据异常，使用原始范围: {min_val:.2f}-{max_val:.2f}")
        else:
            min_val, max_val = valid_data.min(), valid_data.max()

        # 数据范围检查
        if abs(max_val - min_val) < 1e-10:
            margin = abs(min_val) * 0.1 if min_val != 0 else 1.0
            min_val -= margin
            max_val += margin
        else:
            data_range = max_val - min_val
            if min_val >= 0:
                min_val = max(0, min_val - data_range * 0.05)
                max_val += data_range * 0.05
            else:
                margin = data_range * 0.05
                min_val -= margin
                max_val += margin

        if min_val >= max_val:
            min_val, max_val = max_val - 1, min_val + 1

        return min_val, max_val

    def _plot_curve_panel(self, ax: plt.Axes, curve_col: str, color: str, index: int) -> None:
        """绘制单个曲线面板"""
        self._create_title_box(ax, curve_col, color, index)

        # 绘制曲线
        line, = ax.plot(
            self.data[curve_col].values,  # 使用.values提高性能
            self.data[self.depth_col].values,
            color=color,
            linewidth=1.0,
            linestyle='-',
            label=curve_col
        )
        self.plots.append(line)

        # 设置坐标轴范围
        min_val, max_val = self._calculate_curve_display_limits(self.data[curve_col])
        ax.set_xlim(min_val, max_val)

        # 设置Y轴
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        if index == 0:
            ax.set_ylabel('深度 (m)', fontsize=10)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_ylim([self.depth_max, self.depth_min])
        else:
            ax.tick_params(left=False, labelleft=False)

        ax.set_title(curve_col, fontsize=10, pad=5)

    def _plot_all_curves(self, colors: List[str]) -> None:
        """绘制所有曲线面板"""
        self.plots = []

        # 使用优化后的显示顺序
        display_curves = self._optimize_curve_order(self.curve_cols)

        for i, col in enumerate(display_curves):
            ax = self.axs[i]
            color = colors[i % len(colors)]
            self._plot_curve_panel(ax, col, color, i)

        logger.info(f"绘制{len(display_curves)}条曲线完成")

    def _batch_render_classification(self, ax: plt.Axes, class_col: str,
                                   visible_data: pd.DataFrame) -> None:
        """批量渲染分类数据（性能优化）"""
        if visible_data.empty:
            return

        # 按分类值分组，批量绘制
        class_groups = visible_data.groupby(class_col)
        vertices_list = []
        colors_list = []

        for class_val, group in class_groups:
            if pd.isna(class_val) or class_val < 0:
                continue

            class_int = int(class_val)
            xmax = self.litho_width_config.get(class_int, 0.1)
            color = self.DEFAULT_CURVE_COLORS[class_int % len(self.DEFAULT_CURVE_COLORS)]

            # 批量计算矩形顶点
            for depth in group[self.depth_col]:
                y_bottom = depth - self.resolution / 2
                y_top = depth + self.resolution / 2

                vertices = [
                    [0, y_bottom], [xmax, y_bottom],
                    [xmax, y_top], [0, y_top]
                ]
                vertices_list.append(vertices)
                colors_list.append(color)

        # 使用PolyCollection批量绘制
        if vertices_list:
            poly_collection = PolyCollection(
                vertices_list,
                facecolors=colors_list,
                edgecolors='none',
                linewidths=0,
                closed=True
            )
            ax.add_collection(poly_collection)

        logger.debug(f"批量渲染分类数据: {len(vertices_list)}个矩形")

    def _plot_classification_panel(self, ax: plt.Axes, class_col: str, index: int) -> None:
        """绘制单个分类数据面板"""
        self._create_title_box(ax, class_col, '#222222', index)

        # 绘制分类数据
        for depth, class_val in zip(self.data[self.depth_col], self.data[class_col]):
            if pd.isna(class_val) or class_val < 0:
                continue

            class_int = int(class_val)
            xmax = self.litho_width_config.get(class_int, 0.1)

            ax.axhspan(
                ymin=depth - self.resolution / 2,
                ymax=depth + self.resolution / 2,
                xmin=0, xmax=xmax,
                facecolor=self.DEFAULT_CURVE_COLORS[class_int % len(self.DEFAULT_CURVE_COLORS)],
                edgecolor='none',
                linewidth=0,
                clip_on=False
            )

        # 设置面板属性
        ax.set_xticks([])
        ax.set_title(class_col, fontsize=10, pad=5)
        ax.tick_params(left=False, labelleft=False)
        ax.invert_yaxis()

    def _plot_all_classification_panels(self) -> None:
        """绘制所有分类数据面板"""
        self.class_axes = []

        if not self.type_cols:
            return

        base_index = len(self.curve_cols)

        for i, col in enumerate(self.type_cols):
            ax_idx = base_index + i
            ax = self.axs[ax_idx]
            self.class_axes.append(ax)
            self._plot_classification_panel(ax, col, ax_idx)

        logger.info(f"绘制{len(self.type_cols)}个分类面板完成")

    def _optimize_fmi_rendering(self) -> None:
        """FMI图像渲染优化"""
        if not self.fmi_dict:
            return

        for i, image_data in enumerate(self.fmi_dict['image_data']):
            cache_key = f"fmi_{i}"

            # 图像预处理优化
            if image_data.dtype != np.uint8:
                if image_data.max() > image_data.min():
                    normalized = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255
                    self.fmi_dict['image_data'][i] = normalized.astype(np.uint8)

    def _plot_fmi_panel(self, ax: plt.Axes, image_data: np.ndarray,
                       title: str, index: int) -> None:
        """绘制单个FMI图像面板"""
        self._create_title_box(ax, title, '#222222', index)

        fmi_depth = self.fmi_dict['depth']

        # 根据图像维度选择显示方式
        if len(image_data.shape) == 2:
            img = ax.imshow(
                image_data,
                aspect='auto',
                cmap='gray',
                extent=[0, image_data.shape[1], fmi_depth[-1], fmi_depth[0]]
            )
        elif len(image_data.shape) == 3:
            if image_data.shape[2] in [1, 3, 4]:
                if image_data.shape[2] == 1:
                    display_data = image_data[:, :, 0]
                else:
                    display_data = image_data

                img = ax.imshow(
                    display_data,
                    aspect='auto',
                    extent=[0, image_data.shape[1], fmi_depth[-1], fmi_depth[0]]
                )
            else:
                raise ValueError(f"不支持的图像维度: {image_data.shape}")
        else:
            raise ValueError(f"不支持的图像维度: {image_data.shape}")

        self.fmi_images.append(img)
        ax.set_title(title, fontsize=10, pad=5)
        ax.set_xlabel('通道', fontsize=8)
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(left=False, labelleft=False)
        ax.invert_yaxis()

        logger.debug(f"FMI面板{index}绘制完成: {title}")

    def _plot_all_fmi_panels(self) -> None:
        """绘制所有FMI图像面板"""
        self.fmi_axes = []
        self.fmi_images = []

        if not self.fmi_dict:
            return

        base_index = len(self.curve_cols) + len(self.type_cols)
        image_data_list = self.fmi_dict['image_data']
        title_list = self.fmi_dict['title']

        for i, (image_data, title) in enumerate(zip(image_data_list, title_list)):
            ax_idx = base_index + i
            ax = self.axs[ax_idx]
            self.fmi_axes.append(ax)
            self._plot_fmi_panel(ax, image_data, title, ax_idx)

        logger.info(f"绘制{len(image_data_list)}个FMI面板完成")

    def _create_legend_panel(self, legend_dict: Dict) -> None:
        """创建分类图例面板"""
        if not legend_dict:
            return

        n_items = len(legend_dict)
        if n_items == 0:
            return

        # 计算图例尺寸和位置
        legend_height = 0.02
        legend_width = min(0.8, n_items * 0.15)

        # 创建图例坐标轴
        legend_ax = plt.axes([0.5 - legend_width / 2, 0.01, legend_width, legend_height],
                           frameon=False)
        legend_ax.set_axis_off()

        # 准备图例句柄和标签
        handles = []
        labels = []
        sorted_keys = sorted(legend_dict.keys())

        for key in sorted_keys:
            patch = Rectangle((0, 0), 1, 1,
                            facecolor=self.DEFAULT_CURVE_COLORS[key % len(self.DEFAULT_CURVE_COLORS)],
                            edgecolor='black',
                            linewidth=0.5)
            handles.append(patch)
            labels.append(legend_dict[key])

        # 创建图例
        legend = legend_ax.legend(
            handles, labels,
            loc='center',
            ncol=min(n_items, 6),
            frameon=True,
            framealpha=0.9,
            fancybox=True,
            fontsize=9,
            handlelength=1.2,
            handleheight=1.2,
            handletextpad=0.5,
            borderpad=0.4,
            columnspacing=1.0
        )

        # 设置图例样式
        frame = legend.get_frame()
        frame.set_facecolor('#f8f8f8')
        frame.set_edgecolor('#666666')
        frame.set_linewidth(0.5)

        logger.info(f"创建图例面板: {n_items}个分类项")

    # ==================== 交互控制层 ====================

    def _on_window_size_change(self, val: float) -> None:
        """窗口大小滑动条变化事件处理"""
        old_window_size = self.window_size
        self.window_size = val

        # 调整深度位置
        max_valid_position = self.depth_max - self.window_size
        if self.depth_position > max_valid_position:
            self.depth_position = max_valid_position

        logger.debug(f"窗口大小变化: {old_window_size:.1f} -> {self.window_size:.1f}m")
        self._update_display()

    def _on_mouse_scroll(self, event) -> None:
        """鼠标滚轮事件处理 - 深度导航"""
        if event.inaxes not in self.axs:
            return

        # 计算滚动步长
        step = self.window_size * self.LAYOUT_CONFIG['scroll_step_ratio']

        # 根据滚动方向调整深度位置
        if event.button == 'up':
            new_position = self.depth_position - step
            direction = "向上(浅层)"
        elif event.button == 'down':
            new_position = self.depth_position + step
            direction = "向下(深层)"
        else:
            return

        # 限制深度位置
        self.depth_position = np.clip(
            new_position,
            self.depth_min,
            self.depth_max - self.window_size
        )

        logger.debug(f"滚轮滚动{direction}: {self.depth_position:.1f}m")
        self._update_display()

    def _update_display(self) -> None:
        """更新所有显示内容"""
        start_time = time.time()

        # 计算显示范围
        top_depth = self.depth_position
        bottom_depth = self.depth_position + self.window_size

        # 预渲染下一帧数据
        if self.PERFORMANCE_CONFIG['lazy_rendering']:
            self._prerender_visible_data(top_depth, bottom_depth)

        # 更新所有坐标轴范围
        for ax in self.axs:
            ax.set_ylim(bottom_depth, top_depth)

        # 更新各组件
        self._update_classification_panels(top_depth, bottom_depth)
        self._update_fmi_display(top_depth, bottom_depth)
        self._update_depth_indicator(top_depth, bottom_depth)

        # 重绘图形
        self.fig.canvas.draw_idle()

        # 性能监控
        render_time = (time.time() - start_time) * 1000
        self._render_times.append(render_time)

        # 修复除零错误：添加安全检查
        if len(self._render_times) > 10:
            avg_time = np.mean(self._render_times[-10:])
            total_cache_attempts = self._cache_hits + self._cache_misses

            # 只有在有缓存尝试时才计算命中率
            if total_cache_attempts > 0:
                hit_rate = self._cache_hits / total_cache_attempts * 100
                logger.debug(f"渲染性能: 平均{avg_time:.1f}ms, 缓存命中率: {hit_rate:.1f}%")
            else:
                logger.debug(f"渲染性能: 平均{avg_time:.1f}ms, 暂无缓存数据")

    def _update_classification_panels(self, top_depth: float, bottom_depth: float) -> None:
        """更新分类面板显示内容"""
        if not self.class_axes:
            return

        # 获取可见数据
        visible_data = self._get_optimized_visible_data(top_depth, bottom_depth)

        for i, (ax, col) in enumerate(zip(self.class_axes, self.type_cols)):
            ax.clear()
            ax.set_xticks([])
            ax.set_title(col, fontsize=10, pad=5)
            ax.tick_params(left=False, labelleft=False)
            ax.invert_yaxis()

            # 使用批量渲染
            self._batch_render_classification(ax, col, visible_data)

    def _update_fmi_display(self, top_depth: float, bottom_depth: float) -> None:
        """更新FMI图像显示"""
        if not self.fmi_dict or not self.fmi_images:
            return

        fmi_depth = self.fmi_dict['depth']
        visible_indices = (fmi_depth >= top_depth) & (fmi_depth <= bottom_depth)

        if not np.any(visible_indices):
            return

        for i, (img, image_data) in enumerate(zip(self.fmi_images, self.fmi_dict['image_data'])):
            visible_data = image_data[visible_indices]

            if len(visible_data) > 0:
                if len(visible_data.shape) == 2:
                    img.set_data(visible_data)
                elif len(visible_data.shape) == 3:
                    if visible_data.shape[2] == 1:
                        img.set_data(visible_data[:, :, 0])
                    else:
                        img.set_data(visible_data)

                # 更新显示范围
                visible_depths = fmi_depth[visible_indices]
                if len(visible_depths) > 0:
                    img.set_extent([0, visible_data.shape[1], visible_depths[-1], visible_depths[0]])

    def _update_depth_indicator(self, top_depth: float, bottom_depth: float) -> None:
        """更新深度位置指示器"""
        indicator_text = (f"当前深度: {top_depth:.1f} - {bottom_depth:.1f} m | "
                         f"窗口: {self.window_size:.1f} m | "
                         f"限制范围: {self.depth_min:.1f} - {self.depth_max:.1f} m")

        if hasattr(self, '_depth_indicator'):
            self._depth_indicator.set_text(indicator_text)
        else:
            self._depth_indicator = self.fig.text(
                0.5, 0.97,
                indicator_text,
                ha='center', va='top',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2')
            )

    # ==================== 主入口函数 ====================

    def visualize(self,
                  data: pd.DataFrame,
                  depth_col: str = 'Depth',
                  curve_cols: List[str] = None,
                  type_cols: List[str] = None,
                  figsize: Tuple[float, float] = (12, 10),
                  colors: List[str] = None,
                  legend_dict: Dict = None,
                  fmi_dict: Dict = None,
                  depth_limit_config: Optional[List[float]] = None,
                  figure: Optional[plt.Figure] = None) -> None:
        """
        主可视化函数 - 测井数据交互式显示

        Args:
            data: 测井数据DataFrame
            depth_col: 深度列名
            curve_cols: 曲线列名列表
            type_cols: 分类数据列名列表
            figsize: 图形大小
            colors: 颜色列表
            legend_dict: 图例映射字典
            fmi_dict: FMI图像数据字典
            depth_limit_config: 深度限制配置
            figure: 现有图形对象
        """
        try:
            logger.info("开始测井数据可视化流程")
            start_time = time.time()

            # 参数处理
            if curve_cols is None:
                curve_cols = ['L1', 'L2', 'L3', 'L4']
            if type_cols is None:
                type_cols = []
            if colors is None:
                colors = self.DEFAULT_CURVE_COLORS
            if legend_dict is None:
                legend_dict = {}

            # 输入验证
            self._validate_input_parameters(data, depth_col, curve_cols, type_cols)
            self._validate_fmi_data(fmi_dict)

            # 数据初始化
            self.data = data.sort_values(depth_col).reset_index(drop=True)
            self.depth_col = depth_col
            self.curve_cols = curve_cols
            self.type_cols = type_cols
            self.fmi_dict = fmi_dict
            self.depth_limit_config = depth_limit_config

            # 关键优化：曲线顺序优化（成像道前置）
            optimized_curves = self._optimize_curve_order(curve_cols)
            self.curve_cols = optimized_curves

            # 数据预处理
            self._setup_depth_limits(depth_limit_config)

            # 计算显示参数
            depth_range = self.depth_max - self.depth_min
            self.window_size = depth_range * 0.5
            self.depth_position = self.depth_min
            self.resolution = get_resolution_by_depth(self.data[depth_col].dropna().values)

            # 图形界面设置
            n_plots = self._calculate_subplot_count()
            has_legend = bool(legend_dict)

            self._setup_figure_layout(figure, n_plots, has_legend)
            self._create_window_size_slider(has_legend)

            # 配置和优化
            self._setup_lithology_width_config()
            self._optimize_fmi_rendering()

            # 绘制各组件
            self._plot_all_curves(colors)
            self._plot_all_classification_panels()
            self._plot_all_fmi_panels()

            # 交互功能绑定
            self.window_size_slider.on_changed(self._on_window_size_change)
            self.fig.canvas.mpl_connect('scroll_event', self._on_mouse_scroll)

            # 辅助组件
            self._create_legend_panel(legend_dict)

            # 初始预渲染
            self._prerender_visible_data(self.depth_position, self.depth_position + self.window_size)

            # 初始显示
            self._update_display()

            total_time = time.time() - start_time
            logger.info(f"可视化初始化完成，总耗时: {total_time:.2f}s")

            # 显示图形
            plt.show()

        except Exception as e:
            logger.error(f"测井可视化失败: {str(e)}")
            if self.fig:
                plt.close(self.fig)
            raise

    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        stats = {
            'total_renders': len(self._render_times),
            'avg_render_time': np.mean(self._render_times) if self._render_times else 0,
            'cache_hit_rate': (self._cache_hits / (self._cache_hits + self._cache_misses)
                             if (self._cache_hits + self._cache_misses) > 0 else 0),
            'cache_size': len(self._data_cache),
            'imaging_curves_count': len(self.imaging_curve_cols),
            'regular_curves_count': len(self.regular_curve_cols)
        }
        return stats

    def enable_performance_mode(self, enabled: bool = True) -> None:
        """动态启用/禁用性能模式"""
        self.PERFORMANCE_CONFIG['cache_enabled'] = enabled
        self.PERFORMANCE_CONFIG['lazy_rendering'] = enabled
        logger.info(f"性能模式: {'启用' if enabled else '禁用'}")


# ==================== 兼容性接口函数 ====================

def visualize_well_logs(data: pd.DataFrame,
                        depth_col: str = 'Depth',
                        curve_cols: List[str] = None,
                        type_cols: List[str] = None,
                        figsize: Tuple[float, float] = (12, 10),
                        colors: List[str] = None,
                        legend_dict: Dict = None,
                        fmi_dict: Dict = None,
                        depth_limit_config: Optional[List[float]] = None,
                        figure: Optional[plt.Figure] = None) -> None:
    """
    测井数据可视化兼容接口函数
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
        fmi_dict=fmi_dict,
        depth_limit_config=depth_limit_config,
        figure=figure
    )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 生成示例测试数据
    print("生成示例测井数据...")
    n_points = 1000

    # 创建示例数据
    sample_data = pd.DataFrame({
        'Depth': np.linspace(300, 400, n_points),
        'GR': np.random.normal(60, 15, n_points),  # 伽马射线
        'RT': np.random.lognormal(2, 0.3, n_points),  # 电阻率
        'NPHI': np.random.uniform(0.15, 0.45, n_points),  # 中子孔隙度
        'RHOB': np.random.uniform(2.3, 2.7, n_points),  # 体积密度
        'LITHOLOGY': np.random.choice([0, 1, 2, 3], n_points, p=[0.4, 0.3, 0.2, 0.1]),
        'FACIES': np.random.choice([0, 1, 2], n_points, p=[0.5, 0.3, 0.2])
    })

    # # 准备FMI示例数据
    depth_fmi = sample_data.Depth.values
    print(f"FMI深度数据形状: {depth_fmi.shape}, 起始深度: {depth_fmi[0]}, 结束深度: {depth_fmi[-1]}")

    FMI_DYNA = np.random.randint(0, 256, size=[20, 8], dtype=np.uint8)
    FMI_STAT = np.random.randint(0, 256, size=[20, 8], dtype=np.uint8)
    fmi_dynamic = cv2.resize(FMI_DYNA, (256, depth_fmi.shape[0]), interpolation=cv2.INTER_NEAREST)
    fmi_static = cv2.resize(FMI_STAT, (256, depth_fmi.shape[0]), interpolation=cv2.INTER_NEAREST)
    print(f"FMI动态图像形状: {FMI_DYNA.shape}, FMI静态图像形状: {FMI_STAT.shape}")

    # 使用类接口进行可视化
    print("创建可视化器...")
    visualizer = WellLogVisualizer()

    try:
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
            depth_limit_config=[320, 380],  # 只显示320-380米段
            figsize=(14, 8)
        )

    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback

        traceback.print_exc()