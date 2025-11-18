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
    测井数据可视化类 - FMI前置版本

    核心架构说明：
    本类采用分层架构设计，将功能划分为5个主要层次：
    1. 数据管理层：负责数据验证、预处理和缓存管理
    2. 界面管理层：负责图形布局、组件创建和界面设置
    3. 绘图管理层：负责FMI图像、曲线、分类数据的实际绘制
    4. 交互控制层：处理用户交互事件（缩放、滚动等）
    5. 性能优化层：实现缓存、预渲染等性能优化功能

    缓存系统说明：
    - 使用LRU(最近最少使用)缓存策略管理数据
    - 支持预渲染机制，提前渲染可视区域附近的数据
    - 批量绘制技术减少绘图调用次数
    """

    # ==================== 类常量定义区域 ====================
    # 这些常量定义了系统的默认配置，便于统一管理和修改

    # 默认曲线颜色序列，当用户未指定颜色时使用
    DEFAULT_CURVE_COLORS = [
        '#FF0000', '#00FF00', '#0000FF', '#00FFFF', '#FF00FF',  # 红、绿、蓝、青、洋红
        '#8000FF', '#00FF80', '#FF0080', '#FFA500', '#FFFF00'  # 紫、春绿、玫瑰红、橙、黄
    ]

    # 布局配置常量 - 控制图形各元素的相对位置和大小
    LAYOUT_CONFIG = {
        'left_margin': 0.05,  # 左侧边距（图形宽度比例）
        'right_margin': 0.98,  # 右侧边距
        'top_margin': 0.96,  # 顶部边距
        'legend_bottom_margin': 0.065,  # 有图例时的底部边距
        'no_legend_bottom_margin': 0.05,  # 无图例时的底部边距
        'title_margin': 0.001,  # 标题边距
        'title_height': 0.04,  # 标题区域高度
        'slider_width': 0.02,  # 滑动条宽度
        'fmi_panel_width': 0.1,  # FMI面板宽度
        'min_window_size_ratio': 0.01,  # 最小窗口大小与总深度范围的比例
        'scroll_step_ratio': 0.1  # 滚轮滚动步长与窗口大小的比例
    }

    # 性能优化配置 - 控制缓存、预渲染等性能相关参数
    PERFORMANCE_CONFIG = {
        'cache_enabled': True,  # 是否启用缓存
        'lazy_rendering': True,  # 是否启用延迟渲染
        'batch_size': 1000,  # 批量处理的数据点数量
        'max_cache_size': 500,  # 最大缓存条目数
        'pre_render_margin': 0.3,  # 预渲染范围超出可视区域的比例
    }

    def __init__(self, performance_config: Dict[str, Any] = None):
        """
        初始化测井数据可视化器

        Args:
            performance_config: 可选的性能配置字典，用于覆盖默认配置
                - cache_enabled: 是否启用数据缓存
                - lazy_rendering: 是否启用延迟渲染
                - max_cache_size: 最大缓存大小
                - 等...

        属性初始化说明：
        - 数据相关属性：存储原始数据、列名配置等
        - 显示状态属性：记录当前显示范围、深度位置等
        - 图形对象属性：存储matplotlib图形对象
        - 性能优化属性：管理缓存、渲染统计等
        """
        # 合并用户提供的性能配置（如果存在）
        if performance_config:
            self.PERFORMANCE_CONFIG.update(performance_config)
            logger.info(f"应用自定义性能配置: {performance_config}")

        # === 数据相关属性 ===
        self.data: Optional[pd.DataFrame] = None  # 原始测井数据
        self.depth_col: Optional[str] = None  # 深度列名
        self.curve_cols: Optional[List[str]] = None  # 曲线数据列名列表
        self.type_cols: Optional[List[str]] = None  # 分类数据列名列表
        self.fmi_dict: Optional[Dict[str, Any]] = None  # FMI图像数据字典
        self.depth_limit_config: Optional[List[float]] = None  # 深度显示限制配置

        # === 显示状态属性 ===
        self.depth_min: float = 0.0  # 最小显示深度
        self.depth_max: float = 0.0  # 最大显示深度
        self.depth_position: float = 0.0  # 当前显示区域的顶部深度
        self.window_size: float = 0.0  # 当前显示窗口的深度范围大小
        self.resolution: float = 0.0  # 数据的深度分辨率
        self.litho_width_config: Dict[int, float] = {}  # 岩性分类显示宽度配置

        # === 图形对象属性 ===
        self.fig: Optional[plt.Figure] = None  # 主图形对象
        self.axs: Optional[List[plt.Axes]] = None  # 子图坐标轴列表
        self.window_size_slider: Optional[Slider] = None  # 窗口大小滑动条
        self.plots: List[Any] = []  # 曲线绘图对象列表
        self.class_axes: List[plt.Axes] = []  # 分类数据坐标轴列表
        self.fmi_axes: List[plt.Axes] = []  # FMI图像坐标轴列表
        self.fmi_images: List[Any] = []  # FMI图像对象列表

        # === 性能优化属性 ===
        self._data_cache: OrderedDict[str, Any] = OrderedDict()  # 数据缓存（LRU缓存）
        self._visible_data_cache: Optional[pd.DataFrame] = None  # 当前可见数据缓存
        self._cache_valid: bool = False  # 缓存有效性标志
        self._prerendered_data: Optional[pd.DataFrame] = None  # 预渲染数据
        self._prerendered_range: Tuple[float, float] = (0.0, 0.0)  # 预渲染范围
        self._render_times: List[float] = []  # 渲染时间记录（用于性能分析）
        self._cache_hits: int = 0  # 缓存命中次数
        self._cache_misses: int = 0  # 缓存未命中次数

        logger.info("WellLogVisualizer初始化完成，性能模式: %s",
                    "启用" if self.PERFORMANCE_CONFIG['cache_enabled'] else "禁用")

    # ==================== 数据管理层方法 ====================
    def _validate_input_parameters(self, data: pd.DataFrame, depth_col: str,
                                   curve_cols: List[str], type_cols: List[str]) -> None:
        """
        验证输入参数的完整性和有效性

        参数验证流程：
        1. 数据类型检查
        2. 非空检查
        3. 列名存在性检查
        4. 数据有效性检查

        Args:
            data: 测井数据DataFrame
            depth_col: 深度列名
            curve_cols: 曲线数据列名列表
            type_cols: 分类数据列名列表

        Raises:
            ValueError: 当参数不符合要求时抛出
        """
        # 1. 基本类型检查
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data参数必须是pandas DataFrame类型")

        if not depth_col or not isinstance(depth_col, str):
            raise ValueError("depth_col必须是非空字符串")

        if not curve_cols or not isinstance(curve_cols, list):
            raise ValueError("curve_cols必须是非空列表")

        # 处理可选的type_cols参数
        if type_cols is None:
            type_cols = []
        elif not isinstance(type_cols, list):
            raise ValueError("type_cols必须是列表或None")

        # 2. 列名存在性验证
        required_cols = [depth_col] + curve_cols + type_cols
        missing_cols = set(required_cols) - set(data.columns)

        if missing_cols:
            raise ValueError(f"数据中缺少以下必要列: {missing_cols}")

        # 3. 数据有效性检查
        if data.empty:
            raise ValueError("输入数据不能为空DataFrame")

        if data[depth_col].isna().all():
            raise ValueError("深度列数据全部为空，无法确定深度范围")

        logger.info("输入参数验证通过: 深度列='%s', 曲线列=%s, 分类列=%s",
                    depth_col, curve_cols, type_cols)

    def _validate_fmi_data(self, fmi_dict: Optional[Dict[str, Any]]) -> None:
        """
        验证FMI图像数据的结构和完整性

        FMI数据字典预期结构:
        {
            'depth': np.ndarray,      # 深度数据，形状为(n_depth,)
            'image_data': List[np.ndarray],  # 图像数据列表，每个形状为(n_depth, width)或(n_depth, width, channels)
            'title': Optional[List[str]]  # 图像标题列表（可选）
        }

        Args:
            fmi_dict: FMI数据字典

        Raises:
            ValueError: 当FMI数据结构不符合要求时抛出
        """
        if fmi_dict is None:
            logger.info("未提供FMI数据，跳过FMI图像显示")
            return

        # 检查必要键是否存在
        required_keys = ['depth', 'image_data']
        for key in required_keys:
            if key not in fmi_dict:
                raise ValueError(f"FMI字典缺少必要键: {key}")

        # 验证image_data类型和结构
        if not isinstance(fmi_dict['image_data'], list):
            raise ValueError("FMI字典的image_data必须是列表")

        depth_data = fmi_dict['depth']
        image_data_list = fmi_dict['image_data']

        if len(image_data_list) == 0:
            raise ValueError("FMI图像数据列表不能为空")

        # 验证每个图像数据的维度和深度匹配
        for i, image_data in enumerate(image_data_list):
            if not isinstance(image_data, np.ndarray):
                raise ValueError(f"FMI图像数据[{i}]必须是numpy数组")

            if len(depth_data) != image_data.shape[0]:
                raise ValueError(
                    f"FMI图像数据[{i}]的深度维度不匹配: "
                    f"深度长度={len(depth_data)}, 图像深度维度={image_data.shape[0]}"
                )

        # 处理标题数据
        if 'title' not in fmi_dict or fmi_dict['title'] is None:
            # 自动生成标题
            fmi_dict['title'] = [f'FMI_{i}' for i in range(len(image_data_list))]
        else:
            title_list = fmi_dict['title']
            # 如果标题数量不足，补充默认标题
            if len(title_list) < len(image_data_list):
                logger.warning("FMI标题数量不足，自动补充默认标题")
                for i in range(len(title_list), len(image_data_list)):
                    title_list.append(f'FMI_{i}')

        logger.info("FMI数据验证通过，包含%d个图像，深度范围: %.2f-%.2f",
                    len(image_data_list), depth_data.min(), depth_data.max())

    def _setup_depth_limits(self, depth_limit_config: Optional[List[float]]) -> None:
        """
        设置深度显示范围限制

        处理逻辑：
        1. 如果提供了深度限制配置，则使用配置范围与数据实际范围的交集
        2. 如果没有提供配置，则使用数据的完整深度范围
        3. 根据最终范围过滤数据

        Args:
            depth_limit_config: 深度限制配置 [min_depth, max_depth]

        Raises:
            ValueError: 当深度范围配置无效时抛出
        """
        # 计算数据的完整深度范围
        full_depth_min = self.data[self.depth_col].min()
        full_depth_max = self.data[self.depth_col].max()

        logger.debug("数据完整深度范围: %.2f - %.2f", full_depth_min, full_depth_max)

        if depth_limit_config is not None:
            # 验证深度限制配置
            if len(depth_limit_config) != 2:
                raise ValueError("depth_limit_config必须是包含2个元素的列表 [min_depth, max_depth]")

            config_min, config_max = depth_limit_config

            if config_min >= config_max:
                raise ValueError(f"深度范围配置无效: min={config_min} >= max={config_max}")

            # 计算实际使用的深度范围（配置范围与数据范围的交集）
            self.depth_min = max(full_depth_min, config_min)
            self.depth_max = min(full_depth_max, config_max)

            if self.depth_min >= self.depth_max:
                raise ValueError(
                    f"计算后的深度范围无效: min={self.depth_min}, max={self.depth_max}. "
                    f"请检查深度限制配置是否与数据范围重叠"
                )

            logger.info("应用深度限制: %.2f - %.2f (配置: %.2f - %.2f)",
                        self.depth_min, self.depth_max, config_min, config_max)
        else:
            # 使用完整数据范围
            self.depth_min = full_depth_min
            self.depth_max = full_depth_max
            logger.info("使用完整数据深度范围: %.2f - %.2f", self.depth_min, self.depth_max)

        # 根据深度范围过滤数据
        original_size = len(self.data)
        mask = (self.data[self.depth_col] >= self.depth_min) & (self.data[self.depth_col] <= self.depth_max)
        self.data = self.data[mask].reset_index(drop=True)

        filtered_size = len(self.data)
        logger.info("数据过滤完成: %d -> %d 个数据点 (保留%.1f%%)",
                    original_size, filtered_size, (filtered_size / original_size) * 100)

    def _setup_lithology_width_config(self) -> None:
        """
        设置岩性分类数据的显示宽度配置

        为每个岩性分类类型分配不同的显示宽度，便于在图中区分不同的分类。
        宽度分配策略：按分类值排序后均匀分配宽度比例。
        """
        if not self.type_cols:
            self.litho_width_config = {}
            logger.debug("无分类数据，跳过岩性宽度配置")
            return

        # 收集所有分类数据中的唯一值
        all_type_values = []
        for col in self.type_cols:
            valid_data = self.data[col].dropna()
            all_type_values.extend(valid_data.unique())

        if not all_type_values:
            self.litho_width_config = {}
            logger.warning("分类数据中未找到有效值，跳过岩性宽度配置")
            return

        # 获取唯一的分类值并排序
        unique_types = np.unique(all_type_values)
        type_count = len(unique_types)

        # 为每个分类类型分配宽度比例
        for i, litho_type in enumerate(sorted(unique_types)):
            litho_int = int(litho_type)
            if type_count == 1:
                # 只有一个分类时使用全宽度
                self.litho_width_config[litho_int] = 1.0
            else:
                # 多个分类时按顺序分配递增的宽度
                self.litho_width_config[litho_int] = (i + 1) / type_count

        logger.info("岩性分类宽度配置完成: %d个分类类型，配置: %s",
                    type_count, self.litho_width_config)

    # ==================== 性能优化层方法 ====================
    def _get_cached_data(self, key: str, depth_range: Tuple[float, float]) -> Optional[Any]:
        """
        从缓存中获取数据（如果存在且有效）

        缓存键格式: "{key}_{start_depth:.2f}_{end_depth:.2f}"
        使用LRU策略管理缓存，最近使用的数据会被移动到缓存末尾

        Args:
            key: 缓存键前缀，标识数据类型
            depth_range: 深度范围 (start, end)

        Returns:
            缓存的数据，如果未命中则返回None
        """
        if not self.PERFORMANCE_CONFIG['cache_enabled']:
            return None

        # 生成缓存键
        cache_key = f"{key}_{depth_range[0]:.2f}_{depth_range[1]:.2f}"

        if cache_key in self._data_cache:
            # 缓存命中：移动数据到缓存末尾（标记为最近使用）
            self._data_cache.move_to_end(cache_key)
            self._cache_hits += 1
            logger.debug("缓存命中: %s", cache_key)
            return self._data_cache[cache_key]

        # 缓存未命中
        self._cache_misses += 1
        logger.debug("缓存未命中: %s", cache_key)
        return None

    def _set_cached_data(self, key: str, depth_range: Tuple[float, float], data: Any) -> None:
        """
        将数据设置到缓存中

        如果缓存达到最大大小，会移除最久未使用的数据（LRU策略）

        Args:
            key: 缓存键前缀
            depth_range: 深度范围
            data: 要缓存的数据
        """
        if not self.PERFORMANCE_CONFIG['cache_enabled']:
            return

        cache_key = f"{key}_{depth_range[0]:.2f}_{depth_range[1]:.2f}"

        # 添加数据到缓存
        self._data_cache[cache_key] = data
        self._data_cache.move_to_end(cache_key)  # 标记为最近使用

        # 检查缓存大小，如果超出限制则移除最旧的数据
        if len(self._data_cache) > self.PERFORMANCE_CONFIG['max_cache_size']:
            oldest_key = next(iter(self._data_cache))  # 获取最旧的键（第一个）
            del self._data_cache[oldest_key]
            logger.debug("缓存清理: 移除最旧缓存项 %s (当前大小: %d)",
                         oldest_key, len(self._data_cache))

    def _get_visible_data(self, top_depth: float, bottom_depth: float) -> pd.DataFrame:
        """
        获取当前显示深度范围内的数据，使用缓存优化

        缓存策略：不仅返回请求范围的数据，还会缓存一个更大的范围（包含预渲染区域），
        以减少后续数据查询次数。

        Args:
            top_depth: 显示区域顶部深度
            bottom_depth: 显示区域底部深度

        Returns:
            深度范围内的数据子集
        """
        # 首先检查缓存中是否已有足够的数据
        if (self._cache_valid and self._visible_data_cache is not None and
                len(self._visible_data_cache) > 0):

            cache_top = self._visible_data_cache[self.depth_col].min()
            cache_bottom = self._visible_data_cache[self.depth_col].max()

            # 如果缓存数据完全包含请求范围，则直接返回子集
            if cache_top <= top_depth and cache_bottom >= bottom_depth:
                visible_data = self._visible_data_cache[
                    (self._visible_data_cache[self.depth_col] >= top_depth) &
                    (self._visible_data_cache[self.depth_col] <= bottom_depth)
                    ]
                logger.debug("使用缓存数据: 请求[%.2f-%.2f], 缓存[%.2f-%.2f]",
                             top_depth, bottom_depth, cache_top, cache_bottom)
                return visible_data

        # 缓存不满足要求，需要重新查询数据
        logger.debug("缓存未命中，查询数据库: [%.2f-%.2f]", top_depth, bottom_depth)
        visible_data = self.data[
            (self.data[self.depth_col] >= top_depth) &
            (self.data[self.depth_col] <= bottom_depth)
            ]

        # 更新缓存：缓存一个更大的范围以减少后续查询
        cache_margin = self.window_size * 0.5  # 缓存范围扩展量
        cache_top = max(self.depth_min, top_depth - cache_margin)
        cache_bottom = min(self.depth_max, bottom_depth + cache_margin)

        self._visible_data_cache = self.data[
            (self.data[self.depth_col] >= cache_top) &
            (self.data[self.depth_col] <= cache_bottom)
            ]
        self._cache_valid = True

        logger.debug("更新数据缓存: [%.2f-%.2f] (扩展%.1fm)",
                     cache_top, cache_bottom, cache_margin)

        return visible_data

    def _prerender_visible_data(self, current_top: float, current_bottom: float) -> None:
        """
        预渲染当前窗口前后范围的数据，提高滚动响应速度

        预渲染策略：提前加载和准备当前可视区域前后一定范围的数据，
        当用户滚动时可以直接使用预渲染数据，减少计算延迟。

        Args:
            current_top: 当前显示区域顶部深度
            current_bottom: 当前显示区域底部深度
        """
        if not self.PERFORMANCE_CONFIG['lazy_rendering']:
            return

        # 计算预渲染范围（当前范围前后扩展一定比例）
        margin = self.window_size * self.PERFORMANCE_CONFIG['pre_render_margin']
        prerender_top = max(self.depth_min, current_top - margin)
        prerender_bottom = min(self.depth_max, current_bottom + margin)

        # 只有预渲染范围发生变化时才重新预渲染
        if (prerender_top, prerender_bottom) != self._prerendered_range:
            self._prerendered_data = self._get_visible_data(prerender_top, prerender_bottom)
            self._prerendered_range = (prerender_top, prerender_bottom)
            logger.debug("预渲染数据范围: %.1f-%.1f (扩展%.1f%%)",
                         prerender_top, prerender_bottom, self.PERFORMANCE_CONFIG['pre_render_margin'] * 100)

    def _get_optimized_visible_data(self, top_depth: float, bottom_depth: float) -> pd.DataFrame:
        """
        优化后的可见数据获取，优先使用预渲染数据

        性能优化流程：
        1. 首先检查预渲染数据是否包含请求范围
        2. 如果包含，直接返回预渲染数据的子集
        3. 如果不包含，回退到常规数据查询

        Args:
            top_depth: 请求顶部深度
            bottom_depth: 请求底部深度

        Returns:
            深度范围内的优化数据
        """
        # 检查预渲染数据是否可用且包含请求范围
        if (self._prerendered_data is not None and
                self._prerendered_range[0] <= top_depth and
                self._prerendered_range[1] >= bottom_depth):
            visible_data = self._prerendered_data[
                (self._prerendered_data[self.depth_col] >= top_depth) &
                (self._prerendered_data[self.depth_col] <= bottom_depth)
                ]
            logger.debug("使用预渲染数据: 请求[%.2f-%.2f], 预渲染[%.2f-%.2f]",
                         top_depth, bottom_depth, self._prerendered_range[0], self._prerendered_range[1])
            return visible_data

        # 回退到常规数据获取
        logger.debug("预渲染数据不适用，使用常规查询")
        return self._get_visible_data(top_depth, bottom_depth)

    # ==================== 界面管理层方法 ====================

    def _calculate_subplot_count(self) -> int:
        """
        计算需要的子图总数

        计算逻辑：
        子图总数 = 曲线面板数 + 分类面板数 + FMI面板数

        Returns:
            需要的子图总数
        """
        n_plots = len(self.curve_cols) + len(self.type_cols)

        if self.fmi_dict and self.fmi_dict.get('image_data'):
            n_plots += len(self.fmi_dict['image_data'])

        logger.info("子图数量计算: 曲线%d + 分类%d + FMI%d = 总计%d",
                    len(self.curve_cols), len(self.type_cols),
                    len(self.fmi_dict['image_data']) if self.fmi_dict else 0, n_plots)
        return n_plots

    def _setup_figure_layout(self, figure: Optional[plt.Figure], n_plots: int,
                             has_legend: bool, figsize: Tuple[float, float] = (12, 10)) -> None:
        """
        设置图形布局和子图排列

        布局策略：
        1. 创建或重用图形对象
        2. 设置子图排列（共享Y轴）
        3. 调整边距和间距

        Args:
            figure: 可重用的图形对象（如果为None则创建新图形）
            n_plots: 子图数量
            has_legend: 是否显示图例（影响底部边距）

        Raises:
            ValueError: 当子图数量为0时抛出
        """
        if n_plots == 0:
            raise ValueError("没有可显示的子图内容，请检查数据配置")

        # 创建或重用图形对象
        if figure is None:
            self.fig, self.axs = plt.subplots(
                1, n_plots,
                figsize=figsize,  # 使用传入的figsize参数
                sharey=True,  # 所有子图共享Y轴（深度轴）
                gridspec_kw={'wspace': 0.0}  # 子图间无间距
            )
            logger.info("创建新图形: %d个子图", n_plots)
        else:
            self.fig = figure
            self.fig.clear()  # 清除原有内容
            self.axs = self.fig.subplots(1, n_plots, sharey=True, gridspec_kw={'wspace': 0.0})
            logger.info("重用现有图形: %d个子图", n_plots)

        # 确保axs总是列表形式（单子图时matplotlib返回单个Axes对象）
        if n_plots == 1:
            self.axs = [self.axs]

        # 根据是否显示图例计算底部边距
        bottom_margin = (self.LAYOUT_CONFIG['legend_bottom_margin'] if has_legend else self.LAYOUT_CONFIG['no_legend_bottom_margin'])

        # 应用图形布局调整
        plt.subplots_adjust(
            left=self.LAYOUT_CONFIG['left_margin'],
            right=self.LAYOUT_CONFIG['right_margin'],
            bottom=bottom_margin,
            top=self.LAYOUT_CONFIG['top_margin'],
            wspace=0.0,                                             # 水平间距为0
        )

        logger.info("图形布局设置完成: 边距[L=%.3f, R=%.3f, B=%.3f, T=%.3f]",
                    self.LAYOUT_CONFIG['left_margin'], self.LAYOUT_CONFIG['right_margin'],
                    bottom_margin, self.LAYOUT_CONFIG['top_margin'])

    def _create_window_size_slider(self, has_legend: bool) -> None:
        """
        创建窗口大小控制滑动条

        滑动条功能：
        - 控制显示窗口的深度范围大小
        - 垂直方向放置在图右侧
        - 实时响应用户调整

        Args:
            has_legend: 是否显示图例（影响滑动条位置计算）
        """
        # 计算滑动条位置和尺寸
        bottom_margin = (self.LAYOUT_CONFIG['legend_bottom_margin'] if has_legend else self.LAYOUT_CONFIG['no_legend_bottom_margin'])
        slider_height = self.LAYOUT_CONFIG['top_margin'] - bottom_margin

        slider_ax = plt.axes([
            self.LAYOUT_CONFIG['right_margin'],  # x位置（右侧）
            bottom_margin,  # y位置（底部）
            self.LAYOUT_CONFIG['slider_width'],  # 宽度
            slider_height  # 高度
        ])

        # 计算滑动条的数值范围
        depth_range = self.depth_max - self.depth_min
        min_window_size = depth_range * self.LAYOUT_CONFIG['min_window_size_ratio']
        max_window_size = depth_range

        # 设置初始窗口大小（总范围的50%）
        initial_window_size = depth_range * 0.5
        self.window_size = initial_window_size

        # 创建滑动条对象
        self.window_size_slider = Slider(
            ax=slider_ax,
            label='',  # 空标签，使用自定义文本
            valmin=min_window_size,
            valmax=max_window_size,
            valinit=initial_window_size,
            valstep=(max_window_size - min_window_size) / 100,  # 100个步长
            orientation='vertical'  # 垂直方向
        )

        # 添加滑动条标签文本
        slider_ax.text(
            x=0.5, y=0.5,
            s='窗口大小(m)，点击或拖动以调整可视化器窗长大小',
            rotation=270,  # 垂直文本
            ha='center', va='center',
            transform=slider_ax.transAxes,  # 使用坐标轴相对坐标
            fontsize=8
        )

        logger.info("窗口大小滑动条创建完成: 范围[%.1f-%.1f], 初始值%.1f",
                    min_window_size, max_window_size, initial_window_size)

    def _create_title_box(self, ax: plt.Axes, title: str, color: str, index: int) -> None:
        """
        为每个子图创建统一的标题框

        标题框设计：
        - 位于子图顶部中央
        - 半透明背景提高可读性
        - 统一样式保持界面一致性

        Args:
            ax: 目标坐标轴对象
            title: 标题文本
            color: 标题颜色
            index: 子图索引（用于日志）
        """
        # 获取坐标轴在图形中的位置
        orig_pos = ax.get_position()

        # 计算标题框位置（在坐标轴上方）
        title_bbox = [
            orig_pos.x0 + self.LAYOUT_CONFIG['title_margin'],  # x起始
            orig_pos.y0 + orig_pos.height + self.LAYOUT_CONFIG['title_margin'],  # y起始
            orig_pos.width - 2 * self.LAYOUT_CONFIG['title_margin'],  # 宽度
            self.LAYOUT_CONFIG['title_height']  # 高度
        ]

        # 创建标题背景矩形
        title_rect = Rectangle(
            (title_bbox[0], title_bbox[1]),  # 左下角位置
            title_bbox[2], title_bbox[3],  # 宽度和高度
            transform=self.fig.transFigure,  # 使用图形坐标
            facecolor='#f5f5f5',  # 浅灰色背景
            edgecolor='#aaaaaa',  # 灰色边框
            linewidth=1,
            clip_on=False,  # 不裁剪
            zorder=10  # 较高层级确保显示在最前
        )
        self.fig.add_artist(title_rect)

        # 创建标题文本
        title_text = Text(
            title_bbox[0] + title_bbox[2] / 2,  # x中心
            title_bbox[1] + title_bbox[3] / 2,  # y中心
            title,
            fontsize=12,
            fontweight='bold',
            color=color,
            ha='center', va='center',  # 居中显示
            transform=self.fig.transFigure,
            clip_on=False,
            zorder=11  # 在背景矩形之上
        )
        self.fig.add_artist(title_text)
        logger.debug("为子图%d创建标题框: '%s'", index, title)

    # ==================== 绘图管理层方法 ====================
    def _calculate_curve_display_limits(self, curve_data: pd.Series) -> Tuple[float, float]:
        """
        智能计算曲线的显示范围（X轴范围）

        计算策略：
        1. 过滤异常值（如-999等填充值）
        2. 计算有效数据的范围
        3. 添加适当边距确保数据完全显示
        4. 处理特殊情况下（如常数数据）的范围计算
        Args:
            curve_data: 曲线数据序列
        Returns:
            (min_value, max_value) 显示范围
        """
        # 过滤异常值（常见的无效数据标记）
        valid_mask = (curve_data > -999) & (curve_data < 999) & ~curve_data.isna()
        valid_data = curve_data[valid_mask]

        if valid_data.empty:
            # 没有有效数据时使用原始范围
            min_val, max_val = curve_data.min(), curve_data.max()
            logger.warning("曲线数据异常，使用原始范围: %.2f-%.2f", min_val, max_val)
        else:
            min_val, max_val = valid_data.min(), valid_data.max()

        # 处理特殊情况：数据范围过小或为常数
        if abs(max_val - min_val) < 1e-10:
            # 数据基本为常数，添加相对边距
            margin = abs(min_val) * 0.1 if min_val != 0 else 1.0
            min_val -= margin
            max_val += margin
            logger.debug("常数数据检测，添加边距: %.2f", margin)
        else:
            # 正常数据，添加5%的边距
            data_range = max_val - min_val
            if min_val >= 0:
                # 非负数据，确保从0或接近0开始
                min_val = max(0, min_val - data_range * 0.05)
                max_val += data_range * 0.05
            else:
                # 包含负值的数据，两侧添加边距
                margin = data_range * 0.05
                min_val -= margin
                max_val += margin

        # 最终范围验证
        if min_val >= max_val:
            min_val, max_val = max_val - 1, min_val + 1
            logger.warning("显示范围无效，使用安全范围: %.2f-%.2f", min_val, max_val)

        logger.debug("曲线显示范围计算: %.2f-%.2f", min_val, max_val)
        return min_val, max_val

    def _plot_fmi_panel(self, ax: plt.Axes, image_data: np.ndarray,
                        title: str, index: int) -> None:
        """
        绘制单个FMI图像面板

        支持多种图像格式：
        - 2D灰度图像: (depth, width)
        - 3D彩色图像: (depth, width, channels)，支持1/3/4通道

        Args:
            ax: 目标坐标轴
            image_data: FMI图像数据
            title: 面板标题
            index: 面板索引
        """
        self._create_title_box(ax, title, '#222222', index)

        fmi_depth = self.fmi_dict['depth']

        # 根据图像维度选择适当的显示方法
        if len(image_data.shape) == 2:
            # 2D灰度图像
            img = ax.imshow(
                image_data,
                aspect='auto',  # 自动调整宽高比
                # cmap='gray',  # 灰度色彩映射
                cmap='hot',
                extent=[0, image_data.shape[1], fmi_depth[-1], fmi_depth[0]],           # 显示范围
            )
        elif len(image_data.shape) == 3:
            # 3D彩色/多通道图像
            if image_data.shape[2] in [1, 3, 4]:
                if image_data.shape[2] == 1:
                    # 单通道，压缩为2D
                    display_data = image_data[:, :, 0]
                else:
                    # 3或4通道（RGB/RGBA）
                    display_data = image_data

                img = ax.imshow(
                    display_data,
                    aspect='auto',
                    extent=[0, image_data.shape[1], fmi_depth[-1], fmi_depth[0]]
                )
            else:
                raise ValueError(f"不支持的图像通道数: {image_data.shape[2]}")
        else:
            raise ValueError(f"不支持的图像维度: {image_data.shape}")

        # 存储图像对象用于后续更新
        self.fmi_images.append(img)

        # # 设置坐标轴属性
        # ax.set_title(title, fontsize=10, pad=5)
        # ax.set_xlabel('通道', fontsize=8)
        # ax.tick_params(axis='x', labelsize=6)
        # 设置隐藏X方向的坐标轴属性
        ax.set_title(title, fontsize=10, pad=5)
        # 隐藏所有x轴相关元素
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_xlabel('')  # 清空x轴标签

        # 只在第一个FMI面板显示Y轴刻度标签
        if index == 0:  # 第一个FMI面板
            ax.set_ylabel('深度 (m)', fontsize=10)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_ylim([self.depth_max, self.depth_min])  # 设置深度范围
        else:
            ax.tick_params(left=False, labelleft=False)  # 隐藏Y轴刻度

        ax.invert_yaxis()  # 深度向下递增
        logger.debug("FMI面板%d绘制完成: '%s', 形状%s", index, title, image_data.shape)

    def _plot_all_fmi_panels(self) -> None:
        """
        绘制所有FMI图像面板

        绘制顺序：FMI面板显示在所有曲线和分类面板的前面（最左侧）
        """
        self.fmi_axes = []
        self.fmi_images = []

        if not self.fmi_dict:
            logger.debug("无FMI数据，跳过FMI面板绘制")
            return

        # FMI面板放在最前面（最左侧的子图）
        image_data_list = self.fmi_dict['image_data']
        title_list = self.fmi_dict['title']

        for i, (image_data, title) in enumerate(zip(image_data_list, title_list)):
            ax = self.axs[i]  # 前几个子图用于FMI
            self.fmi_axes.append(ax)
            self._plot_fmi_panel(ax, image_data, title, i)

        logger.info("FMI面板绘制完成: %d个面板", len(image_data_list))

    def _plot_curve_panel(self, ax: plt.Axes, curve_col: str,
                          color: str, index: int) -> None:
        """
        绘制单个曲线面板

        Args:
            ax: 目标坐标轴
            curve_col: 曲线列名
            color: 曲线颜色
            index: 面板索引
        """
        self._create_title_box(ax, curve_col, color, index)

        # 绘制曲线（使用.values提高性能）
        line, = ax.plot(
            self.data[curve_col].values,  # X轴数据（使用numpy数组提高性能）
            self.data[self.depth_col].values,  # Y轴数据（深度）
            color=color,
            linewidth=1.0,
            linestyle='-',
            label=curve_col
        )
        self.plots.append(line)

        # 设置X轴显示范围
        min_val, max_val = self._calculate_curve_display_limits(self.data[curve_col])
        ax.set_xlim(min_val, max_val)

        # 设置坐标轴属性
        ax.invert_yaxis()  # 深度向下递增
        ax.grid(True, alpha=0.3)  # 显示网格

        # # 只在第一个曲线面板显示Y轴刻度标签
        # 调整Y轴刻度显示逻辑
        if not self.fmi_axes:  # 如果没有FMI面板
            # 在第一个曲线面板显示Y轴刻度
            if index == 0:  # 第一个曲线面板
                ax.set_ylabel('深度 (m)', fontsize=10)
                ax.tick_params(axis='y', labelsize=8)
                ax.set_ylim([self.depth_max, self.depth_min])       # 设置深度范围
            else:
                ax.tick_params(left=False, labelleft=False)         # 隐藏Y轴刻度
        else:
            # 有FMI面板时，曲线面板都不显示Y轴刻度
            ax.tick_params(left=False, labelleft=False)             # 隐藏Y轴刻度

        ax.set_title(curve_col, fontsize=10, pad=5)
        logger.debug("曲线面板%d绘制完成: '%s'", index, curve_col)

    def _plot_all_curves(self, colors: List[str]) -> None:
        """
        绘制所有曲线面板

        Args:
            colors: 曲线颜色列表
        """
        self.plots = []

        # 曲线面板放在FMI面板后面
        start_index = len(self.fmi_axes)

        for i, col in enumerate(self.curve_cols):
            ax_index = start_index + i
            ax = self.axs[ax_index]
            color = colors[i % len(colors)]  # 循环使用颜色
            self._plot_curve_panel(ax, col, color, ax_index)

        logger.info("曲线面板绘制完成: %d条曲线", len(self.curve_cols))

    def _batch_render_classification(self, ax: plt.Axes, class_col: str,
                                     visible_data: pd.DataFrame) -> None:
        """
        批量渲染分类数据（性能优化版本）

        使用PolyCollection批量绘制矩形，显著提高渲染性能，
        特别适用于大量分类数据的显示。

        Args:
            ax: 目标坐标轴
            class_col: 分类数据列名
            visible_data: 可见范围内的数据
        """
        if visible_data.empty:
            logger.debug("分类数据为空，跳过批量渲染")
            return

        # 按分类值分组数据
        class_groups = visible_data.groupby(class_col)
        vertices_list = []  # 存储所有矩形的顶点
        colors_list = []  # 存储所有矩形的颜色

        for class_val, group in class_groups:
            # 跳过无效分类值
            if pd.isna(class_val) or class_val < 0:
                continue

            class_int = int(class_val)
            # 获取该分类的显示宽度
            xmax = self.litho_width_config.get(class_int, 0.1)
            color = self.DEFAULT_CURVE_COLORS[class_int % len(self.DEFAULT_CURVE_COLORS)]

            # 为每个数据点计算矩形顶点
            for depth in group[self.depth_col]:
                y_bottom = depth - self.resolution / 2
                y_top = depth + self.resolution / 2

                # 矩形顶点（左下→右下→右上→左上）
                vertices = [
                    [0, y_bottom],  # 左下
                    [xmax, y_bottom],  # 右下
                    [xmax, y_top],  # 右上
                    [0, y_top]  # 左上
                ]
                vertices_list.append(vertices)
                colors_list.append(color)

        # 使用PolyCollection批量绘制所有矩形
        if vertices_list:
            poly_collection = PolyCollection(
                vertices_list,
                facecolors=colors_list,
                edgecolors='none',  # 无边框
                linewidths=0,
                closed=True
            )
            ax.add_collection(poly_collection)

        logger.debug("分类数据批量渲染完成: %d个数据点，%d个矩形",
                     len(visible_data), len(vertices_list))

    def _plot_classification_panel(self, ax: plt.Axes, class_col: str, index: int) -> None:
        """
        绘制单个分类数据面板（初始绘制，非优化版本）

        Args:
            ax: 目标坐标轴
            class_col: 分类数据列名
            index: 面板索引
        """
        self._create_title_box(ax, class_col, '#222222', index)

        # 为每个数据点绘制水平矩形
        for depth, class_val in zip(self.data[self.depth_col], self.data[class_col]):
            if pd.isna(class_val) or class_val < 0:
                continue

            class_int = int(class_val)
            xmax = self.litho_width_config.get(class_int, 0.1)

            # 绘制水平矩形代表分类
            ax.axhspan(
                ymin=depth - self.resolution / 2,  # 矩形底部
                ymax=depth + self.resolution / 2,  # 矩形顶部
                xmin=0, xmax=xmax,  # 矩形宽度
                facecolor=self.DEFAULT_CURVE_COLORS[class_int % len(self.DEFAULT_CURVE_COLORS)],
                edgecolor='none',
                linewidth=0,
                clip_on=False
            )

        # 设置面板属性
        ax.set_xticks([])  # 隐藏X轴刻度
        ax.set_title(class_col, fontsize=10, pad=5)
        ax.tick_params(left=False, labelleft=False)  # 隐藏Y轴刻度
        ax.invert_yaxis()  # 深度向下递增

        logger.debug("分类面板%d初始绘制完成: '%s'", index, class_col)

    def _plot_all_classification_panels(self) -> None:
        """
        绘制所有分类数据面板
        """
        self.class_axes = []

        if not self.type_cols:
            logger.debug("无分类数据，跳过分类面板绘制")
            return

        # 分类面板放在曲线面板后面
        base_index = len(self.fmi_axes) + len(self.curve_cols)

        for i, col in enumerate(self.type_cols):
            ax_idx = base_index + i
            ax = self.axs[ax_idx]
            self.class_axes.append(ax)
            self._plot_classification_panel(ax, col, ax_idx)

        logger.info("分类面板绘制完成: %d个面板", len(self.type_cols))

    def _optimize_fmi_rendering(self) -> None:
        """
        FMI图像渲染优化预处理

        优化措施：
        1. 图像数据标准化（归一化到0-255）
        2. 数据类型转换（提高渲染效率）
        3. 图像质量优化
        """
        if not self.fmi_dict:
            return

        for i, image_data in enumerate(self.fmi_dict['image_data']):
            # 图像预处理优化：转换为uint8类型提高渲染效率
            if image_data.dtype != np.uint8:
                if image_data.max() > image_data.min():
                    # 归一化到0-255范围
                    normalized = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255
                    self.fmi_dict['image_data'][i] = normalized.astype(np.uint8)
                    logger.debug("FMI图像%d标准化完成: %s -> uint8", i, image_data.dtype)
                else:
                    # 常数图像处理
                    self.fmi_dict['image_data'][i] = np.full_like(image_data, 128, dtype=np.uint8)
                    logger.warning("FMI图像%d为常数图像，使用中间灰度值", i)

    def _create_legend_panel(self, legend_dict: Dict[int, str]) -> None:
        """
        创建分类图例面板

        Args:
            legend_dict: 分类值到标签的映射字典
        """
        if not legend_dict:
            logger.debug("无图例数据，跳过图例创建")
            return

        n_items = len(legend_dict)
        if n_items == 0:
            return

        # 计算图例尺寸和位置（位于图形底部中央）
        legend_height = 0.02
        legend_width = min(0.8, n_items * 0.15)  # 动态宽度，最大80%

        # 创建图例坐标轴（隐藏边框）
        legend_ax = plt.axes([0.5 - legend_width / 2, 0.01, legend_width, legend_height],
                             frameon=False)
        legend_ax.set_axis_off()

        # 准备图例句柄和标签
        handles = []
        labels = []
        sorted_keys = sorted(legend_dict.keys())

        for key in sorted_keys:
            # 创建颜色块
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
            ncol=min(n_items, 6),  # 最多6列
            frameon=True,
            framealpha=0.9,  # 半透明背景
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
        frame.set_facecolor('#f8f8f8')  # 浅灰色背景
        frame.set_edgecolor('#666666')  # 灰色边框
        frame.set_linewidth(0.5)

        logger.info("图例面板创建完成: %d个分类项", n_items)

    # ==================== 交互控制层方法 ====================

    def _on_window_size_change(self, val: float) -> None:
        """
        窗口大小滑动条变化事件处理

        Args:
            val: 新的窗口大小值
        """
        old_window_size = self.window_size
        self.window_size = val

        # 调整深度位置确保显示范围有效
        max_valid_position = self.depth_max - self.window_size
        if self.depth_position > max_valid_position:
            self.depth_position = max_valid_position

        logger.debug("窗口大小变化: %.1f -> %.1fm", old_window_size, self.window_size)
        self._update_display()

    def _on_mouse_scroll(self, event) -> None:
        """
        鼠标滚轮事件处理 - 深度导航

        支持功能：
        - 向上滚动：向浅层（小深度）移动
        - 向下滚动：向深层（大深度）移动

        Args:
            event: 鼠标事件对象
        """
        # 只处理在子图内的滚轮事件
        if event.inaxes not in self.axs:
            return

        # 计算滚动步长（窗口大小的10%）
        step = self.window_size * self.LAYOUT_CONFIG['scroll_step_ratio']

        # 根据滚动方向调整深度位置
        if event.button == 'up':
            new_position = self.depth_position - step
            direction = "向上(浅层)"
        elif event.button == 'down':
            new_position = self.depth_position + step
            direction = "向下(深层)"
        else:
            return  # 忽略其他按钮事件

        # 限制深度位置在有效范围内
        self.depth_position = np.clip(
            new_position,
            self.depth_min,
            self.depth_max - self.window_size
        )

        logger.debug("滚轮滚动%s: 位置%.1fm, 步长%.1fm", direction, self.depth_position, step)
        self._update_display()

    def _update_display(self) -> None:
        """
        更新所有显示内容（核心刷新函数）

        执行流程：
        1. 预渲染下一帧数据
        2. 更新坐标轴范围
        3. 更新各组件显示
        4. 触发图形重绘
        5. 记录性能数据
        """
        start_time = time.time()

        # 计算当前显示范围
        top_depth = self.depth_position
        bottom_depth = self.depth_position + self.window_size

        # 预渲染优化
        if self.PERFORMANCE_CONFIG['lazy_rendering']:
            self._prerender_visible_data(top_depth, bottom_depth)

        # 更新所有坐标轴的Y轴范围
        for ax in self.axs:
            ax.set_ylim(bottom_depth, top_depth)  # 注意：bottom_depth在下，top_depth在上

        # 更新各组件显示
        self._update_classification_panels(top_depth, bottom_depth)
        self._update_fmi_display(top_depth, bottom_depth)
        self._update_depth_indicator(top_depth, bottom_depth)

        # 触发图形重绘（非阻塞方式）
        self.fig.canvas.draw_idle()

        # 性能监控和记录
        render_time = (time.time() - start_time) * 1000  # 转换为毫秒
        self._render_times.append(render_time)

        # 保留最近10次的渲染时间用于性能分析
        if len(self._render_times) > 10:
            self._render_times = self._render_times[-10:]

        # 计算平均渲染时间和缓存命中率
        avg_time = np.mean(self._render_times) if self._render_times else 0
        total_cache_attempts = self._cache_hits + self._cache_misses

        if total_cache_attempts > 0:
            hit_rate = self._cache_hits / total_cache_attempts * 100
            logger.debug("渲染完成: %.1fms, 缓存命中率: %.1f%%", avg_time, hit_rate)
        else:
            logger.debug("渲染完成: %.1fms, 暂无缓存数据", avg_time)

    def _update_classification_panels(self, top_depth: float, bottom_depth: float) -> None:
        """
        更新分类面板显示内容

        Args:
            top_depth: 显示区域顶部深度
            bottom_depth: 显示区域底部深度
        """
        if not self.class_axes:
            return

        # 获取可见范围内的数据（使用优化版本）
        visible_data = self._get_optimized_visible_data(top_depth, bottom_depth)

        # 更新每个分类面板
        for i, (ax, col) in enumerate(zip(self.class_axes, self.type_cols)):
            ax.clear()  # 清除原有内容

            # 重新设置坐标轴属性
            ax.set_xticks([])
            ax.set_title(col, fontsize=10, pad=5)
            ax.tick_params(left=False, labelleft=False)
            ax.invert_yaxis()
            ax.set_ylim(bottom_depth, top_depth)  # 设置显示范围

            # 使用批量渲染优化性能
            self._batch_render_classification(ax, col, visible_data)

        logger.debug("分类面板更新完成: %d个面板", len(self.class_axes))

    def _update_fmi_display(self, top_depth: float, bottom_depth: float) -> None:
        """
        更新FMI图像显示

        Args:
            top_depth: 显示区域顶部深度
            bottom_depth: 显示区域底部深度
        """
        if not self.fmi_dict or not self.fmi_images:
            return

        fmi_depth = self.fmi_dict['depth']

        # 查找在显示范围内的深度索引
        visible_indices = (fmi_depth >= top_depth) & (fmi_depth <= bottom_depth)

        if not np.any(visible_indices):
            logger.debug("当前显示范围内无FMI数据")
            return

        # 更新每个FMI图像
        for i, (img, image_data) in enumerate(zip(self.fmi_images, self.fmi_dict['image_data'])):
            visible_data = image_data[visible_indices]

            if len(visible_data) > 0:
                # 更新图像数据
                if len(visible_data.shape) == 2:
                    img.set_data(visible_data)
                elif len(visible_data.shape) == 3:
                    if visible_data.shape[2] == 1:
                        img.set_data(visible_data[:, :, 0])
                    else:
                        img.set_data(visible_data)

                # 更新图像显示范围
                visible_depths = fmi_depth[visible_indices]
                if len(visible_depths) > 0:
                    img.set_extent([0, visible_data.shape[1], visible_depths[-1], visible_depths[0]])

        logger.debug("FMI显示更新完成: %d个图像", len(self.fmi_images))

    def _update_depth_indicator(self, top_depth: float, bottom_depth: float) -> None:
        """
        更新深度位置指示器文本

        Args:
            top_depth: 当前顶部深度
            bottom_depth: 当前底部深度
        """
        indicator_text = (f"当前深度: {top_depth:.1f} - {bottom_depth:.1f} m | "
                          f"窗口: {self.window_size:.1f} m | "
                          f"限制范围: {self.depth_min:.1f} - {self.depth_max:.1f} m")

        if hasattr(self, '_depth_indicator'):
            # 更新现有指示器文本
            self._depth_indicator.set_text(indicator_text)
        else:
            # 创建新的深度指示器
            self._depth_indicator = self.fig.text(
                0.99, 0.01,  # 右下角位置 (x=0.98, y=0.02)
                indicator_text,
                ha='right', va='bottom',  # 右对齐，底对齐
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.1', edgecolor='gray')
            )

    # ==================== 主入口函数 ====================

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
        """
        主可视化函数 - 测井数据交互式显示

        完整的可视化流程：
        1. 参数验证和预处理
        2. 数据初始化和配置
        3. 图形界面设置
        4. 各组件绘制
        5. 交互功能绑定
        6. 初始显示

        Args:
            data: 包含测井数据的DataFrame
            depth_col: 深度数据列名，默认'Depth'
            curve_cols: 要显示的曲线数据列名列表
            type_cols: 分类数据列名列表
            figsize: 图形大小 (宽度, 高度)
            colors: 曲线颜色列表
            legend_dict: 分类图例映射 {分类值: 标签}
            fmi_dict: FMI图像数据字典
            depth_limit_config: 深度显示限制 [最小深度, 最大深度]
            figure: 可重用的图形对象

        Raises:
            Exception: 可视化过程中的任何错误
        """
        try:
            logger.info("开始测井数据可视化流程")
            start_time = time.time()

            # === 1. 参数预处理和验证 ===
            if curve_cols is None:
                curve_cols = ['L1', 'L2', 'L3', 'L4']
            if type_cols is None:
                type_cols = []
            if colors is None:
                colors = self.DEFAULT_CURVE_COLORS
            if legend_dict is None:
                legend_dict = {}

            logger.info("参数预处理完成: 曲线%d条, 分类%d个, 颜色%d种", len(curve_cols), len(type_cols), len(colors))

            # 输入参数验证
            self._validate_input_parameters(data, depth_col, curve_cols, type_cols)
            self._validate_fmi_data(fmi_dict)

            # === 2. 数据初始化 ===
            self.data = data.sort_values(depth_col).reset_index(drop=True)
            self.depth_col = depth_col
            self.curve_cols = curve_cols
            self.type_cols = type_cols
            self.fmi_dict = fmi_dict
            self.depth_limit_config = depth_limit_config

            # 数据预处理
            self._setup_depth_limits(depth_limit_config)

            # 计算显示参数
            depth_range = self.depth_max - self.depth_min
            self.window_size = depth_range * 0.5  # 初始窗口大小为总范围的50%
            self.depth_position = self.depth_min  # 从最小深度开始显示
            self.resolution = get_resolution_by_depth(self.data[depth_col].dropna().values)

            logger.info("数据显示参数: 范围%.1fm, 窗口%.1fm, 位置%.1fm, 分辨率%.3fm",
                        depth_range, self.window_size, self.depth_position, self.resolution)

            # === 3. 图形界面设置 ===
            n_plots = self._calculate_subplot_count()
            has_legend = bool(legend_dict)

            self._setup_figure_layout(figure, n_plots, has_legend, figsize)
            self._create_window_size_slider(has_legend)

            # === 4. 配置和优化 ===
            self._setup_lithology_width_config()
            self._optimize_fmi_rendering()

            # === 5. 绘制各组件 ===
            self._plot_all_fmi_panels()  # FMI面板（最前面）
            self._plot_all_curves(colors)  # 曲线面板
            self._plot_all_classification_panels()  # 分类面板

            # === 6. 交互功能绑定 ===
            self.window_size_slider.on_changed(self._on_window_size_change)
            self.fig.canvas.mpl_connect('scroll_event', self._on_mouse_scroll)

            # === 7. 辅助组件 ===
            self._create_legend_panel(legend_dict)

            # === 8. 性能优化初始化 ===
            self._prerender_visible_data(self.depth_position, self.depth_position + self.window_size)

            # === 9. 初始显示 ===
            self._update_display()

            total_time = time.time() - start_time
            logger.info("可视化初始化完成，总耗时: %.2fs", total_time)

            # 显示图形
            plt.show()

        except Exception as e:
            logger.error("测井可视化失败: %s", str(e))
            # 清理资源
            if self.fig:
                plt.close(self.fig)
            raise

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息

        Returns:
            包含性能统计信息的字典:
            - total_renders: 总渲染次数
            - avg_render_time: 平均渲染时间(ms)
            - cache_hit_rate: 缓存命中率(%)
            - cache_size: 当前缓存大小
            - imaging_curves_count: 成像曲线数量
            - regular_curves_count: 常规曲线数量
        """
        total_renders = len(self._render_times)
        avg_render_time = np.mean(self._render_times) if self._render_times else 0

        total_cache_attempts = self._cache_hits + self._cache_misses
        cache_hit_rate = (self._cache_hits / total_cache_attempts * 100
                          if total_cache_attempts > 0 else 0)

        stats = {
            'total_renders': total_renders,
            'avg_render_time': avg_render_time,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._data_cache),
            'imaging_curves_count': len(self.fmi_axes) if self.fmi_dict else 0,
            'regular_curves_count': len(self.curve_cols)
        }

        logger.info("性能统计: 渲染%d次, 平均%.1fms, 缓存命中率%.1f%%, 缓存大小%d",
                    total_renders, avg_render_time, cache_hit_rate, len(self._data_cache))
        return stats

    def enable_performance_mode(self, enabled: bool = True) -> None:
        """
        动态启用/禁用性能模式

        Args:
            enabled: 是否启用性能模式
        """
        self.PERFORMANCE_CONFIG['cache_enabled'] = enabled
        self.PERFORMANCE_CONFIG['lazy_rendering'] = enabled

        if not enabled:
            # 禁用性能模式时清空缓存
            self._data_cache.clear()
            self._visible_data_cache = None
            self._prerendered_data = None

        logger.info("性能模式%s", "启用" if enabled else "禁用")

    def clear_cache(self) -> None:
        """清空所有缓存数据"""
        self._data_cache.clear()
        self._visible_data_cache = None
        self._prerendered_data = None
        self._cache_valid = False
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info("缓存已清空")

    def close(self) -> None:
        """关闭可视化器，释放资源"""
        if self.fig:
            plt.close(self.fig)
            self.fig = None

        self.clear_cache()
        logger.info("可视化器已关闭")

# ==================== 兼容性接口函数 ====================
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
    """
    测井数据可视化兼容接口函数（静态方法）

    提供与函数式调用兼容的接口，便于现有代码迁移。
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