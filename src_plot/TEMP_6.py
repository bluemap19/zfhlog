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

# 配置日志系统：记录程序运行状态和调试信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WellLogVisualizer:
    """
    测井数据可视化类 - 专业级可视化工具

    核心功能：
    - 多类型数据显示：常规测井曲线、岩性分类数据、FMI图像数据
    - 交互式导航：滑动条控制窗口大小，鼠标滚轮滚动深度
    - 智能缓存：LRU缓存机制提升大数据量渲染性能
    - 自适应布局：根据数据类型自动调整面板布局

    设计原则：
    - 模块化设计：各功能模块职责单一，便于维护
    - 性能优先：缓存优化和批量渲染减少计算开销
    - 异常安全：完善的错误处理和资源清理
    """

    # 默认曲线颜色序列：10种高对比度颜色，确保不同曲线清晰可辨
    DEFAULT_CURVE_COLORS = [
        '#FF0000',  # 红色
        '#00FF00',  # 绿色
        '#0000FF',  # 蓝色
        '#00FFFF',  # 青色
        '#FF00FF',  # 洋红
        '#8000FF',  # 紫色
        '#00FF80',  # 春绿色
        '#FF0080',  # 深粉色
        '#FFA500',  # 橙色
        '#FFFF00'  # 黄色
    ]

    # 多曲线道颜色序列（用于同一道内的多条曲线）
    DEFAULT_MULTI_CURVE_COLORS = [
        '#FF0000',  # 红色
        '#000000',  # 黑色
        '#0000FF',  # 蓝色
        '#800080',  # 紫色
        '#FFA500',  # 橙色
        '#FFFF00'  # 黄色
    ]

    # 布局配置常量：定义图形各元素的相对位置和尺寸
    LAYOUT_CONFIG = {
        'left_margin': 0.05,  # 左侧边距占图形宽度的比例
        'right_margin': 0.98,  # 右侧边距（为滑动条留空间）
        'top_margin': 0.96,  # 顶部边距
        'legend_bottom_margin': 0.065,  # 有图例时的底部边距
        'no_legend_bottom_margin': 0.05,  # 无图例时的底部边距
        'title_margin': 0.001,  # 标题框与子图的间距
        'title_height': 0.04,  # 标题框高度
        'slider_width': 0.02,  # 滑动条宽度
        'fmi_panel_width': 0.1,  # FMI面板宽度（未直接使用，保留兼容性）
        'min_window_size_ratio': 0.01,  # 最小窗口大小相对于总深度范围的比例
        'scroll_step_ratio': 0.1  # 滚轮滚动步长相对于窗口大小的比例
    }

    def __init__(self, performance_config: Dict[str, Any] = None):
        """
        初始化测井数据可视化器
        参数说明：
        performance_config: 性能配置字典，可自定义缓存参数
            - cache_enabled: 是否启用缓存
            - max_cache_size: 最大缓存条目数
        """
        # 性能配置：控制缓存和行为优化参数
        self.performance_config = {
            'cache_enabled': True,  # 缓存开关
            'max_cache_size': 500,  # 最大缓存数据块数量
        }

        # 如果传入自定义性能配置，更新默认配置
        if performance_config:
            self.performance_config.update(performance_config)

        # 数据相关属性：存储输入数据和配置信息
        self.data: Optional[pd.DataFrame] = None  # 原始测井数据
        self.depth_col: Optional[str] = None  # 深度列名
        self.curve_cols: Optional[List[Any]] = None  # 曲线列名列表
        self.type_cols: Optional[List[str]] = None  # 分类列名列表
        self.fmi_dict: Optional[Dict[str, Any]] = None  # FMI图像数据字典
        self.depth_limit_config: Optional[List[float]] = None  # 深度范围限制配置

        # 显示状态属性：记录当前视图的状态
        self.depth_min: float = 0.0  # 数据最小深度
        self.depth_max: float = 0.0  # 数据最大深度
        self.depth_position: float = 0.0  # 当前视图顶部深度位置
        self.window_size: float = 0.0  # 当前显示窗口的深度范围
        self.resolution: float = 0.0  # 深度采样分辨率（点间距）
        self.litho_width_config: Dict[int, float] = {}  # 岩性类型对应的显示宽度配置

        # 图形对象属性：存储 matplotlib 图形组件
        self.fig: Optional[plt.Figure] = None  # 主图形对象
        self.axs: Optional[List[plt.Axes]] = None  # 子图轴对象列表
        self.window_size_slider: Optional[Slider] = None  # 窗口大小滑动条
        self.plots: List[Any] = []  # 曲线绘图对象列表
        self.class_axes: List[plt.Axes] = []  # 分类数据子图轴列表
        self.fmi_axes: List[plt.Axes] = []  # FMI图像子图轴列表
        self.fmi_images: List[Any] = []     # FMI图像对象列表，画图用的

        # 缓存属性：使用有序字典实现 LRU 缓存策略
        self._data_cache: OrderedDict = OrderedDict()  # 数据缓存：键为深度范围，值为数据块

        # 缓存统计：记录命中率用于性能监控
        self._cache_stats = {'hits': 0, 'misses': 0, 'total_renders': 0}

        # 记录初始化完成日志
        cache_status = "启用" if self.performance_config['cache_enabled'] else "禁用"
        logger.info("WellLogVisualizer初始化完成，缓存%s", cache_status)

    def _validate_input_parameters(self, data: pd.DataFrame, depth_col: str,
                                   curve_cols: List[Any], type_cols: List[str]) -> None:
        """
        验证输入参数的有效性 - 确保数据格式正确且必要列存在

        参数验证流程：
        1. 检查数据类型是否正确
        2. 检查必要参数是否为空
        3. 验证所有指定列是否存在于数据中
        4. 检查数据基本完整性（非空、深度列有效）
        """
        # 1. 验证数据框类型
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data参数必须是pandas DataFrame类型")

        # 2. 验证深度列参数
        if not depth_col or not isinstance(depth_col, str):
            raise ValueError("depth_col必须是非空字符串")

        # 3. 验证曲线列参数
        if not curve_cols or not isinstance(curve_cols, list):
            raise ValueError("curve_cols必须是非空列表")

        # 4. 处理可选的分类列参数
        if type_cols is None:
            type_cols = []  # 转换为空列表便于后续处理
        elif not isinstance(type_cols, list):
            raise ValueError("type_cols必须是列表或None")

        # 修改：展平曲线列结构以检查所有必要列
        required_cols = [depth_col]
        flat_curve_cols = []
        for item in curve_cols:
            if isinstance(item, list):
                # 嵌套列表：多曲线道
                flat_curve_cols.extend(item)
                required_cols.extend(item)
            elif isinstance(item, str):
                # 单曲线
                flat_curve_cols.append(item)
                required_cols.append(item)
            else:
                raise ValueError("curve_cols中的元素必须是字符串或字符串列表")

        # 检查列名存在性 检查所有指定列是否存在于数据中
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"数据中缺少以下必要列: {missing_cols}")

        # 6. 检查数据基本完整性
        if data.empty:
            raise ValueError("输入数据不能为空DataFrame")

        if data[depth_col].isna().all():
            raise ValueError("深度列数据全部为空")

        logger.info("输入参数验证通过，曲线结构: %s", curve_cols)


    def _validate_fmi_data(self, fmi_dict: Optional[Dict[str, Any]]) -> None:
        """
        验证FMI图像数据的结构和完整性

        FMI数据要求结构：
        {
            'depth': 1D数组，与图像数据深度维度匹配,
            'image_data': [图像1, 图像2, ...],  # 每个图像为2D或3D数组
            'title': ['标题1', '标题2', ...]    # 可选，自动生成
        }
        """
        if fmi_dict is None:
            logger.info("无FMI数据")
            return

        # 检查必要键是否存在
        required_keys = ['depth', 'image_data']
        for key in required_keys:
            if key not in fmi_dict:
                raise ValueError(f"FMI字典缺少必要键: {key}")

        # 验证图像数据格式
        if not isinstance(fmi_dict['image_data'], list) or len(fmi_dict['image_data']) == 0:
            raise ValueError("FMI图像数据必须是非空列表")

        depth_data = fmi_dict['depth']
        # 验证每个图像数据的格式和维度匹配
        for i, image_data in enumerate(fmi_dict['image_data']):
            if not isinstance(image_data, np.ndarray):
                raise ValueError(f"FMI图像数据[{i}]必须是numpy数组")
            if len(depth_data) != image_data.shape[0]:
                raise ValueError(f"FMI图像数据[{i}]深度维度不匹配")

        # 自动生成标题（如果未提供）
        if 'title' not in fmi_dict or fmi_dict['title'] is None:
            fmi_dict['title'] = [f'FMI_{i+1}' for i in range(len(fmi_dict['image_data']))]

        logger.info("FMI数据验证通过，包含%d个图像", len(fmi_dict['image_data']))

    def _setup_depth_limits(self, depth_limit_config: Optional[List[float]]) -> None:
        """
        设置深度显示范围，根据配置过滤数据

        处理逻辑：
        1. 如果没有配置限制，使用完整数据范围
        2. 如果有配置，确保配置有效且与数据范围有重叠
        3. 根据最终范围过滤数据
        """
        # 计算数据的完整深度范围
        full_depth_min = self.data[self.depth_col].min()
        full_depth_max = self.data[self.depth_col].max()

        if depth_limit_config is not None:
            # 验证深度限制配置格式
            if len(depth_limit_config) != 2:
                raise ValueError("depth_limit_config必须是包含2个元素的列表")

            config_min, config_max = depth_limit_config
            if config_min >= config_max:
                raise ValueError(f"深度范围配置无效: min={config_min} >= max={config_max}")

            # 计算实际显示范围（配置范围与数据范围的交集）
            self.depth_min = max(full_depth_min, config_min)
            self.depth_max = min(full_depth_max, config_max)

            # 检查是否有有效范围
            if self.depth_min >= self.depth_max:
                raise ValueError("深度范围配置与数据范围无重叠")
        else:
            # 无配置时使用完整数据范围
            self.depth_min = full_depth_min
            self.depth_max = full_depth_max

        # 根据深度范围过滤数据，只过滤常规测井数据，不对图像测井数据进行过滤
        original_size = len(self.data)
        mask = (self.data[self.depth_col] >= self.depth_min) & (self.data[self.depth_col] <= self.depth_max)
        self.data = self.data[mask].reset_index(drop=True)

        # 记录过滤结果
        logger.info("深度范围设置: %.2f - %.2f, 数据点: %d -> %d",
                    self.depth_min, self.depth_max, original_size, len(self.data))

    def _setup_lithology_width_config(self) -> None:
        """
        设置岩性分类的显示宽度配置

        为每种岩性类型分配不同的显示宽度，便于视觉区分
        宽度计算：按类型排序后均匀分配 (i+1)/N
        """
        if not self.type_cols:
            self.litho_width_config = {}
            return

        # 收集所有分类列中的唯一值
        all_type_values = []
        for col in self.type_cols:
            valid_data = self.data[col].dropna()  # 忽略空值
            all_type_values.extend(valid_data.unique())

        if not all_type_values:
            self.litho_width_config = {}
            return

        # 获取所有唯一类型并排序
        unique_types = np.unique(all_type_values)

        # 为每种类型分配宽度：类型值越大，显示宽度越大
        for i, litho_type in enumerate(sorted(unique_types)):
            litho_int = int(litho_type)
            # 宽度按顺序均匀分配：第一种类型宽度=1/N，第二种=2/N，以此类推
            self.litho_width_config[litho_int] = (i + 1) / len(unique_types)

        logger.info("岩性宽度配置: %s", self.litho_width_config)

    def _get_cached_data(self, depth_range: Tuple[float, float]) -> Optional[pd.DataFrame]:
        """
        从缓存获取数据 - 核心缓存方法

        LRU缓存策略：
        - 命中时：将数据移到字典末尾（标记为最近使用）
        - 未命中时：返回None，由调用方处理

        参数：
        depth_range: 请求的深度范围 (start, end)

        返回：
        缓存数据或None（未命中时）
        """
        # 检查缓存是否启用
        if not self.performance_config['cache_enabled']:
            return None

        # # 生成缓存键：使用深度范围的前两位小数确保匹配精度
        cache_key = f"{depth_range[0]:.2f}_{depth_range[1]:.2f}"

        if cache_key in self._data_cache:
            # 缓存命中：移动数据到末尾（LRU策略）
            self._data_cache.move_to_end(cache_key)
            self._cache_stats['hits'] += 1
            logger.debug("缓存命中: %s (请求范围: %.2f-%.2f)", cache_key, depth_range[0], depth_range[1])
            return self._data_cache[cache_key]

        # 缓存未命中
        self._cache_stats['misses'] += 1
        return None

    def _set_cached_data(self, depth_range: Tuple[float, float], data: pd.DataFrame) -> None:
        """
        设置缓存数据

        缓存管理：
        - 添加新数据到缓存
        - 维护缓存大小，超过限制时移除最旧数据
        - 使用LRU策略确保常用数据保留
        """
        if not self.performance_config['cache_enabled']:
            return

        # 使用与获取时相同的精度
        cache_key = f"{depth_range[0]:.2f}_{depth_range[1]:.2f}"

        # 添加数据到缓存并标记为最近使用
        self._data_cache[cache_key] = data
        self._data_cache.move_to_end(cache_key)

        # 检查缓存大小，移除最旧数据（如果超过限制）
        if len(self._data_cache) > self.performance_config['max_cache_size']:
            oldest_key = next(iter(self._data_cache))  # 获取最旧的键（第一个）
            del self._data_cache[oldest_key]
            logger.debug("缓存清理: 移除%s", oldest_key)

    def _get_visible_data(self, top_depth: float, bottom_depth: float) -> pd.DataFrame:
        """
        获取可见深度范围内的数据（带缓存优化）

        缓存策略：
        1. 首先尝试从缓存获取精确范围的数据
        2. 如果未命中，查询数据库并缓存扩展范围的数据
        3. 扩展缓存提高后续滚动操作的命中率

        参数：
        top_depth: 可见区域顶部深度
        bottom_depth: 可见区域底部深度
        """
        # 首先尝试从缓存获取精确范围数据
        cached_data = self._get_cached_data((top_depth, bottom_depth))
        if cached_data is not None:
            return cached_data

        # 缓存未命中，从原始数据查询
        logger.debug("缓存未命中，查询数据: [%.2f-%.2f]", top_depth, bottom_depth)
        visible_data = self.data[
            (self.data[self.depth_col] >= top_depth) &
            (self.data[self.depth_col] <= bottom_depth)
            ]

        # 缓存扩展范围数据
        self._set_cached_data((top_depth, bottom_depth), visible_data)

        return visible_data

    def _calculate_subplot_count(self) -> int:
        """计算子图总数：FMI面板 + 曲线面板 + 分类面板"""
        n_curve_panels = len(self.curve_cols) if self.curve_cols else 0
        n_type_panels = len(self.type_cols) if self.type_cols else 0
        n_fmi_panels = len(self.fmi_dict['image_data']) if self.fmi_dict and self.fmi_dict.get('image_data') else 0

        return n_curve_panels + n_type_panels + n_fmi_panels

    def _setup_figure_layout(self, figure: Optional[plt.Figure], n_plots: int,
                             has_legend: bool, figsize: Tuple[float, float]) -> None:
        """
        设置图形布局和子图排列

        布局逻辑：
        - 创建指定数量的子图，共享Y轴（深度轴）
        - 调整边距为图例和滑动条留空间
        - 子图间无间距（wspace=0）确保紧凑布局
        """
        if n_plots == 0:
            raise ValueError("没有可显示的子图内容")

        # 创建或重用图形对象
        if figure is None:
            # 创建新图形：1行n_plots列，共享Y轴，子图间无间距
            self.fig, self.axs = plt.subplots(1, n_plots, figsize=figsize, sharey=True, gridspec_kw={'wspace': 0.0})
        else:
            # 重用现有图形：清除内容后重新创建子图
            self.fig = figure
            self.fig.clear()
            self.axs = self.fig.subplots(1, n_plots, sharey=True, gridspec_kw={'wspace': 0.0})

        # 确保axs为列表（单子图时subplots返回单个Axes对象）
        if n_plots == 1:
            self.axs = [self.axs]

        # 根据是否有图例选择底部边距
        bottom_margin = (self.LAYOUT_CONFIG['legend_bottom_margin'] if has_legend else self.LAYOUT_CONFIG['no_legend_bottom_margin'])

        # 调整图形布局参数
        plt.subplots_adjust(
            left=self.LAYOUT_CONFIG['left_margin'],  # 左侧边距
            right=self.LAYOUT_CONFIG['right_margin'],  # 右侧边距（为滑动条留空间）
            bottom=bottom_margin,  # 底部边距
            top=self.LAYOUT_CONFIG['top_margin'],  # 顶部边距
            wspace=0.0  # 子图间水平间距为0（紧密排列）
        )

    def _create_window_size_slider(self, has_legend: bool) -> None:
        """
        创建窗口大小滑动条

        滑动条功能：
        - 控制显示窗口的深度范围
        - 垂直方向，位于图形右侧
        - 范围从最小显示比例到完整深度范围
        """
        # 计算滑动条位置和尺寸
        bottom_margin = (self.LAYOUT_CONFIG['legend_bottom_margin'] if has_legend else self.LAYOUT_CONFIG['no_legend_bottom_margin'])
        slider_height = self.LAYOUT_CONFIG['top_margin'] - bottom_margin

        # 创建滑动条轴对象（右侧垂直条）
        slider_ax = plt.axes([
            self.LAYOUT_CONFIG['right_margin'],  # x位置：右侧边距处
            bottom_margin,  # y位置：底部边距处
            self.LAYOUT_CONFIG['slider_width'],  # 宽度
            slider_height  # 高度：从底部到顶部
        ])

        # 计算滑动条数值范围
        depth_range = self.depth_max - self.depth_min
        min_window_size = depth_range * self.LAYOUT_CONFIG['min_window_size_ratio']  # 最小窗口大小
        max_window_size = depth_range  # 最大窗口大小（完整范围）
        initial_window_size = depth_range * 0.5  # 初始窗口大小（一半范围）
        self.window_size = initial_window_size  # 设置当前窗口大小

        # 创建滑动条对象
        self.window_size_slider = Slider(
            ax=slider_ax,
            label='',  # 空标签（使用文本标签代替）
            valmin=min_window_size,
            valmax=max_window_size,
            valinit=initial_window_size,
            orientation='vertical'  # 垂直方向
        )

        # 添加滑动条文本标签（旋转270度）
        slider_ax.text(0.5, 0.5, '窗口大小(m)', rotation=270, ha='center', va='center',
                       transform=slider_ax.transAxes, fontsize=8)

    def _create_title_box(self, ax: plt.Axes, title: Any, color: str, index: int) -> None:
        """
        为子图创建标题框

        标题框设计：
        - 位于子图顶部中央
        - 浅灰色背景，细边框
        - 粗体文本，指定颜色
        - 高zorder确保显示在最上层
        """
        # 获取子图在图形中的位置（相对坐标）
        orig_pos = ax.get_position()

        # 计算标题框位置和尺寸（基于子图位置）
        title_bbox = [
            orig_pos.x0 + self.LAYOUT_CONFIG['title_margin'],  # x起点
            orig_pos.y0 + orig_pos.height + self.LAYOUT_CONFIG['title_margin'],  # y起点（子图上方）
            orig_pos.width - 2 * self.LAYOUT_CONFIG['title_margin'],  # 宽度
            self.LAYOUT_CONFIG['title_height']  # 高度
        ]

        # 创建标题背景矩形
        title_rect = Rectangle(
            (title_bbox[0], title_bbox[1]), title_bbox[2], title_bbox[3],
            transform=self.fig.transFigure,  # 使用图形坐标变换
            facecolor='#f5f5f5',  # 浅灰色背景
            edgecolor='#aaaaaa',  # 灰色边框
            linewidth=1,  # 边框宽度
            clip_on=False,  # 不裁剪（允许显示在子图外）
            zorder=10  # 高层级确保显示在前面
        )
        self.fig.add_artist(title_rect)


        # 修改点：检查title类型，支持列表和字符串
        if isinstance(title, list):
            # 多部分标题：每个部分使用不同颜色
            n_parts = len(title)
            part_width = title_bbox[2] / n_parts  # 每个标题部分的宽度

            for i, part_text in enumerate(title):
                # 计算每个部分的中心位置
                x_center = title_bbox[0] + (i + 0.5) * part_width
                y_center = title_bbox[1] + title_bbox[3] / 2

                # 从多曲线颜色序列中获取颜色（循环使用）
                part_color = self.DEFAULT_MULTI_CURVE_COLORS[i % len(self.DEFAULT_MULTI_CURVE_COLORS)]

                # 创建单个标题部分文本
                text_obj = Text(
                    x_center, y_center,
                    part_text,
                    fontsize=12, fontweight='bold', color=part_color,
                    ha='center', va='center',
                    transform=self.fig.transFigure,
                    clip_on=False,
                    zorder=11
                )
                self.fig.add_artist(text_obj)
        else:
            # 单标题：使用传入的颜色
            title_text = Text(
                title_bbox[0] + title_bbox[2] / 2,
                title_bbox[1] + title_bbox[3] / 2,
                title,
                fontsize=12, fontweight='bold', color=color,
                ha='center', va='center',
                transform=self.fig.transFigure,
                clip_on=False,
                zorder=11
            )
            self.fig.add_artist(title_text)

    def _calculate_curve_display_limits(self, curve_data: pd.Series) -> Tuple[float, float]:
        """
        计算曲线的显示范围（X轴范围）

        范围计算策略：
        1. 过滤异常值（-999到999之外的值）
        2. 处理常数数据：添加相对边距
        3. 处理非常数数据：添加5%边距
        4. 确保非负数据的显示范围从0或正值开始
        """
        # 过滤异常值：排除极端值和空值
        valid_mask = (curve_data > -999) & (curve_data < 999) & ~curve_data.isna()
        valid_data = curve_data[valid_mask]

        if valid_data.empty:
            # 无有效数据时使用原始数据范围
            min_val, max_val = curve_data.min(), curve_data.max()
        else:
            # 使用有效数据范围
            min_val, max_val = valid_data.min(), valid_data.max()

        # 处理常数数据（变化范围极小）
        if abs(max_val - min_val) < 1e-10:
            # 添加相对边距：非零值用10%边距，零值用固定边距
            margin = abs(min_val) * 0.1 if min_val != 0 else 1.0
            min_val -= margin
            max_val += margin
        else:
            # 添加5%边距
            data_range = max_val - min_val
            margin = data_range * 0.05
            # 非负数据确保从0或正值开始显示
            min_val = max(0, min_val - margin) if min_val >= 0 else min_val - margin
            max_val += margin

        return min_val, max_val

    def _plot_fmi_panel(self, ax: plt.Axes, image_data: np.ndarray, title: str, index: int) -> None:
        """
        绘制FMI图像面板

        支持图像格式：
        - 2D灰度图像：使用热力图色彩映射
        - 3D彩色图像：直接显示RGB或RGBA
        - 3D单通道：转换为2D显示
        """
        # 创建标题框
        self._create_title_box(ax, title, '#222222', index)
        fmi_depth = self.fmi_dict['depth']

        # 根据图像维度选择显示方法
        if len(image_data.shape) == 2:
            # 2D图像：使用热力图色彩映射
            img = ax.imshow(image_data, aspect='auto', cmap='hot', extent=[0, image_data.shape[1], fmi_depth[-1], fmi_depth[0]])
        elif len(image_data.shape) == 3 and image_data.shape[2] in [1, 3, 4]:
            # 3D图像：单通道转换为2D，多通道直接显示
            display_data = image_data if image_data.shape[2] != 1 else image_data[:, :, 0]
            img = ax.imshow(display_data, aspect='auto', extent=[0, image_data.shape[1], fmi_depth[-1], fmi_depth[0]])
        else:
            raise ValueError(f"不支持的图像维度: {image_data.shape}")

        # 保存图像对象用于后续更新
        self.fmi_images.append(img)

        # 设置坐标轴：隐藏X轴标签
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_xlabel('')

        # Y轴显示逻辑：只有第一个面板显示Y轴标签
        if index == 0:  # 第一个面板显示Y轴
            ax.set_ylabel('深度 (m)', fontsize=10)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_ylim([self.depth_max, self.depth_min])  # 设置Y轴范围
        else:
            ax.tick_params(left=False, labelleft=False)  # 其他面板隐藏Y轴

        # 反转Y轴：深度值从上到下递增
        ax.invert_yaxis()

    def _plot_all_fmi_panels(self) -> None:
        """绘制所有FMI图像面板"""
        self.fmi_axes = []  # 清空FMI轴列表
        self.fmi_images = []  # 清空图像对象列表

        if not self.fmi_dict:
            return  # 无FMI数据时直接返回

        # 为每个FMI图像创建面板
        for i, (image_data, title) in enumerate(zip(self.fmi_dict['image_data'], self.fmi_dict['title'])):
            ax = self.axs[i]  # 使用前几个子图
            self.fmi_axes.append(ax)  # 记录FMI轴
            self._plot_fmi_panel(ax, image_data, title, i)

    def _plot_curve_panel(self, ax: plt.Axes, curve_item: Any, color: str, index: int) -> None:
        """
        绘制单个曲线面板

        曲线显示特性：
        - 深度在Y轴，曲线值在X轴
        - 自动计算合适的显示范围
        - 添加网格线提高可读性
        - 反转Y轴符合地质显示习惯
        """
        if isinstance(curve_item, list):
            # 修改：多曲线道 - 绘制多条曲线
            self._create_title_box(ax, curve_item, color, index)

            # 为每条曲线分配颜色
            for i, curve_col in enumerate(curve_item):
                curve_color = self.DEFAULT_MULTI_CURVE_COLORS[i % len(self.DEFAULT_MULTI_CURVE_COLORS)]
                line, = ax.plot(self.data[curve_col].values, self.data[self.depth_col].values,
                                color=curve_color, linewidth=1.0, linestyle='-', label=curve_col)
                self.plots.append(line)

            # 计算多曲线的联合显示范围
            all_min, all_max = float('inf'), float('-inf')
            for curve_col in curve_item:
                min_val, max_val = self._calculate_curve_display_limits(self.data[curve_col])
                all_min = min(all_min, min_val)
                all_max = max(all_max, max_val)

            ax.set_xlim(all_min, all_max)
        elif isinstance(curve_item, str):
            # 单曲线道 - 原有逻辑 创建标题框
            self._create_title_box(ax, curve_item, color, index)
            # 绘制曲线：X=曲线值，Y=深度
            line, = ax.plot(self.data[curve_item].values, self.data[self.depth_col].values,
                            color=color, linewidth=1.0, linestyle='-', label=curve_item)
            self.plots.append(line)         # 保存绘图对象
            # 设置X轴显示范围
            min_val, max_val = self._calculate_curve_display_limits(self.data[curve_item])
            ax.set_xlim(min_val, max_val)
        else:
            raise ValueError(f"不支持的曲线项类型: {type(curve_item)}")

        # 设置坐标轴属性
        ax.invert_yaxis()  # 反转Y轴：深度从上到下增加
        ax.grid(True, alpha=0.3)  # 添加半透明网格线

        # Y轴显示逻辑：无FMI时的第一个曲线面板显示Y轴
        if not self.fmi_axes and index == 0:  # 无FMI时的第一个曲线面板
            ax.set_ylabel('深度 (m)', fontsize=10)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_ylim([self.depth_max, self.depth_min])  # 设置Y轴范围
        else:
            ax.tick_params(left=False, labelleft=False)  # 其他面板隐藏Y轴

    def _plot_all_curves(self, colors) -> None:
        """绘制所有曲线面板"""
        self.plots = []  # 清空绘图对象列表

        # 计算曲线面板的起始索引（在FMI面板之后）
        start_index = len(self.fmi_axes)

        # 为每条曲线创建面板
        for i, item in enumerate(self.curve_cols):
            ax_index = start_index + i
            color = colors[i % len(colors)]  # 循环使用颜色
            self._plot_curve_panel(self.axs[ax_index], item, color, ax_index)


    def _batch_render_classification(self, ax: plt.Axes, class_col: str, visible_data: pd.DataFrame) -> None:
        """
        批量渲染分类数据（性能优化版本）

        优化策略：
        - 使用PolyCollection批量绘制多边形，比逐个绘制矩形性能更高
        - 按分类值分组，相同颜色的矩形批量处理
        - 减少matplotlib绘图调用次数
        """
        if visible_data.empty:
            return  # 无可见数据时直接返回

        # 按分类值分组数据
        class_groups = visible_data.groupby(class_col)
        vertices_list = []  # 存储所有矩形的顶点
        colors_list = []  # 存储对应的颜色

        # 为每个分类值创建矩形
        for class_val, group in class_groups:
            if pd.isna(class_val) or class_val < 0:
                continue  # 跳过无效值

            class_int = int(class_val)
            # 获取该分类的显示宽度
            xmax = self.litho_width_config.get(class_int, 0.1)

            # 根据分类值选择颜色
            color = self.DEFAULT_CURVE_COLORS[class_int % len(self.DEFAULT_CURVE_COLORS)]

            # 为每个深度点创建矩形
            for depth in group[self.depth_col]:
                # 计算矩形的上下边界（基于深度分辨率）
                y_bottom = depth - self.resolution / 2
                y_top = depth + self.resolution / 2

                # 定义矩形的四个顶点（左下→右下→右上→左上）
                vertices = [[0, y_bottom], [xmax, y_bottom], [xmax, y_top], [0, y_top]]
                vertices_list.append(vertices)
                colors_list.append(color)

        # 如果有矩形数据，批量绘制
        if vertices_list:
            poly_collection = PolyCollection(vertices_list,
                                             facecolors=colors_list,  # 填充颜色
                                             edgecolors='none',  # 无边框
                                             linewidths=0)  # 边框宽度0
            ax.add_collection(poly_collection)  # 添加到轴

    def _plot_classification_panel(self, ax: plt.Axes, class_col: str, index: int) -> None:
        """
        绘制分类数据面板（初始绘制，非优化版本）
        用于初始显示，后续更新使用优化的批量渲染版本
        """
        self._create_title_box(ax, class_col, '#222222', index)

        # 为每个测井道绘制矩形
        for depth, class_val in zip(self.data[self.depth_col], self.data[class_col]):
            if pd.isna(class_val) or class_val < 0:
                continue  # 跳过无效值

            class_int = int(class_val)
            # 获取显示宽度
            xmax = self.litho_width_config.get(class_int, 0.1)

            # 绘制水平矩形条
            ax.axhspan(ymin=depth - self.resolution / 2,  # 下边界
                       ymax=depth + self.resolution / 2,  # 上边界
                       xmin=0, xmax=xmax,  # 左右边界
                       facecolor=self.DEFAULT_CURVE_COLORS[class_int % len(self.DEFAULT_CURVE_COLORS)],
                       edgecolor='none')  # 无边框

        # 设置坐标轴属性
        ax.set_xticks([])  # 隐藏X轴刻度
        ax.tick_params(left=False, labelleft=False)  # 隐藏Y轴
        ax.invert_yaxis()  # 反转Y轴

    def _plot_all_classification_panels(self) -> None:
        """绘制所有分类数据面板"""
        self.class_axes = []  # 清空分类轴列表

        if not self.type_cols:
            return  # 无分类数据时直接返回

        # 计算分类面板的起始索引（在FMI和曲线面板之后）
        base_index = len(self.fmi_axes) + len(self.curve_cols)

        # 为每个分类列创建面板
        for i, col in enumerate(self.type_cols):
            ax_idx = base_index + i  # 计算子图索引
            ax = self.axs[ax_idx]
            self.class_axes.append(ax)  # 记录分类轴
            self._plot_classification_panel(ax, col, ax_idx)

    def _optimize_fmi_rendering(self) -> None:
        """
        FMI图像渲染优化：预处理图像数据提高显示性能

        优化内容：
        - 将浮点图像数据归一化到0-255并转换为uint8
        - 减少图像数据传输和内存占用
        - 提高imshow函数的渲染效率
        """
        if not self.fmi_dict:
            return  # 无FMI数据时直接返回

        # 处理每个FMI图像
        for i, image_data in enumerate(self.fmi_dict['image_data']):
            if image_data.dtype != np.uint8:
                # 非uint8类型需要转换
                if image_data.max() > image_data.min():
                    # 归一化到0-255范围
                    normalized = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255
                    self.fmi_dict['image_data'][i] = normalized.astype(np.uint8)
                else:
                    # 常数图像：填充中间灰度值
                    self.fmi_dict['image_data'][i] = np.full_like(image_data, 128, dtype=np.uint8)

    def _create_legend_panel(self, legend_dict: Dict[int, str]) -> None:
        """
        创建图例面板

        图例设计：
        - 位于图形底部中央
        - 多列布局，自动调整列数
        - 半透明背景，细边框
        - 使用岩性对应的颜色显示
        """
        if not legend_dict:
            return  # 无图例数据时直接返回

        n_items = len(legend_dict)
        # 计算图例尺寸：宽度基于项目数，固定高度
        legend_height, legend_width = 0.02, min(0.8, n_items * 0.15)

        # 创建图例轴对象（底部中央，无边框）
        legend_ax = plt.axes([0.5 - legend_width / 2, 0.01, legend_width, legend_height], frameon=False)
        legend_ax.set_axis_off()  # 隐藏坐标轴

        # 准备图例句柄和标签
        handles, labels = [], []
        for key in sorted(legend_dict.keys()):
            # 创建颜色块作为图例句柄
            patch = Rectangle((0, 0), 1, 1,
                              facecolor=self.DEFAULT_CURVE_COLORS[key % len(self.DEFAULT_CURVE_COLORS)],
                              edgecolor='black', linewidth=0.5)
            handles.append(patch)
            labels.append(legend_dict[key])

        # 创建图例：中央位置，自动列数，半透明背景
        legend = legend_ax.legend(handles, labels, loc='center', ncol=min(n_items, 6),
                                  frameon=True, framealpha=0.9, fontsize=9)

        # 设置图例框样式
        frame = legend.get_frame()
        frame.set_facecolor('#f8f8f8')  # 浅灰色背景
        frame.set_edgecolor('#666666')  # 深灰色边框

    def _on_window_size_change(self, val: float) -> None:
        """
        窗口大小变化事件处理

        处理逻辑：
        1. 更新窗口大小
        2. 调整深度位置确保不超出有效范围
        3. 触发显示更新
        """
        self.window_size = val  # 更新窗口大小

        # 计算最大有效深度位置（确保窗口不超出数据范围）
        max_valid_position = self.depth_max - self.window_size
        if self.depth_position > max_valid_position:
            self.depth_position = max_valid_position  # 调整位置

        self._update_display()  # 更新显示

    def _on_mouse_scroll(self, event) -> None:
        """
        鼠标滚轮事件处理：实现深度滚动

        滚动逻辑：
        - 向上滚动：向浅部（小深度）滚动
        - 向下滚动：向深部（大深度）滚动
        - 步长为窗口大小的10%
        """
        # 检查事件是否发生在子图内
        if event.inaxes not in self.axs:
            return

        # 计算滚动步长（窗口大小的10%）
        step = self.window_size * self.LAYOUT_CONFIG['scroll_step_ratio']

        # 根据滚轮方向计算新位置
        if event.button == 'up':
            new_position = self.depth_position - step  # 向上滚动：向浅部
        elif event.button == 'down':
            new_position = self.depth_position + step  # 向下滚动：向深部
        else:
            return  # 非滚轮事件忽略

        # 限制位置在有效范围内
        self.depth_position = np.clip(new_position, self.depth_min, self.depth_max - self.window_size)
        self._update_display()  # 更新显示

    def _update_display(self) -> None:
        """
        更新显示内容（核心刷新函数）

        刷新流程：
        1. 计算当前显示范围
        2. 更新所有子图的Y轴范围
        3. 更新分类面板内容
        4. 更新FMI图像显示
        5. 更新深度信息显示
        6. 重绘图形
        7. 记录性能数据
        """
        start_time = time.time()  # 记录开始时间
        self._cache_stats['total_renders'] += 1  # 增加渲染计数

        # 计算当前显示的深度范围
        top_depth = self.depth_position
        bottom_depth = self.depth_position + self.window_size

        # 更新所有子图的Y轴范围
        for ax in self.axs:
            ax.set_ylim(bottom_depth, top_depth)  # 注意：matplotlib中Y轴从上到下

        # 更新各显示组件
        self._update_classification_panels(top_depth, bottom_depth)  # 分类面板
        self._update_fmi_display(top_depth, bottom_depth)  # FMI图像
        # self._update_fmi_display(top_depth, bottom_depth)  # FMI图像
        self._update_depth_indicator(top_depth, bottom_depth)  # 深度指示器

        # 请求图形重绘（非阻塞方式）
        self.fig.canvas.draw_idle()

        # 计算并记录渲染时间
        render_time = (time.time() - start_time) * 1000
        logger.debug("渲染完成: %.1fms", render_time)

    def _update_classification_panels(self, top_depth: float, bottom_depth: float) -> None:
        """
        更新分类面板显示

        更新策略：
        1. 清除现有内容
        2. 获取可见范围内的数据
        3. 使用批量渲染方法重新绘制
        """
        if not self.class_axes:
            return  # 无分类面板时直接返回

        # 获取可见深度范围内的数据（使用缓存优化）
        visible_data = self._get_visible_data(top_depth, bottom_depth)

        # 更新每个分类面板
        for i, (ax, col) in enumerate(zip(self.class_axes, self.type_cols)):
            ax.clear()  # 清除现有内容

            # 重新设置坐标轴属性
            ax.set_xticks([])  # 隐藏X轴
            ax.tick_params(left=False, labelleft=False)  # 隐藏Y轴
            ax.invert_yaxis()  # 反转Y轴
            ax.set_ylim(bottom_depth, top_depth)  # 设置Y轴范围

            # 使用批量渲染方法绘制分类数据
            self._batch_render_classification(ax, col, visible_data)

    def _update_fmi_display(self, top_depth: float, bottom_depth: float) -> None:
        """
        更新FMI图像显示：只更新可见范围内的图像数据

        优化策略：
        - 不重新创建图像对象，只更新数据
        - 减少内存分配和图像传输开销
        """
        if not self.fmi_dict or not self.fmi_images:
            return  # 无FMI数据时直接返回

        fmi_depth = self.fmi_dict['depth']
        # 计算当前可见的深度索引
        visible_indices = (fmi_depth >= top_depth) & (fmi_depth <= bottom_depth)

        if not np.any(visible_indices):
            return  # 无可见数据时直接返回

        # 更新每个FMI图像
        for img, image_data in zip(self.fmi_images, self.fmi_dict['image_data']):
            visible_data = image_data[visible_indices]  # 提取可见部分数据

            if len(visible_data) > 0:
                # 根据图像维度设置数据
                if len(visible_data.shape) == 2:
                    img.set_data(visible_data)  # 2D图像直接设置数据
                elif len(visible_data.shape) == 3:
                    # 3D图像：单通道转换为2D显示
                    display_data = visible_data if visible_data.shape[2] != 1 else visible_data[:, :, 0]
                    img.set_data(display_data)

                # 更新图像显示范围
                visible_depths = fmi_depth[visible_indices]
                if len(visible_depths) > 0:
                    img.set_extent([0, visible_data.shape[1], visible_depths[-1], visible_depths[0]])

    def _update_depth_indicator(self, top_depth: float, bottom_depth: float) -> None:
        """
        更新深度指示器：显示当前视图的深度范围和窗口大小

        位置：图形右下角，半透明背景确保可读性
        """
        # 生成指示器文本
        indicator_text = (f"深度: {top_depth:.1f} - {bottom_depth:.1f} m | "
                          f"窗口: {self.window_size:.1f} m")

        if hasattr(self, '_depth_indicator'):
            # 更新现有文本对象
            self._depth_indicator.set_text(indicator_text)
        else:
            # 创建新文本对象（首次调用时）
            self._depth_indicator = self.fig.text(
                0.99, 0.01, indicator_text,  # 位置：右下角
                ha='right', va='bottom', fontsize=8,  # 对齐和字体
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.1')  # 半透明背景框
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
        """
        主可视化函数：完整的测井数据可视化流程

        执行流程：
        1. 参数预处理和验证
        2. 数据初始化和预处理
        3. 图形设置和布局
        4. 数据绘制和优化
        5. 交互功能设置
        6. 初始显示和性能监控
        """
        try:
            logger.info("开始测井数据可视化")
            start_time = time.time()  # 记录开始时间

            # ========== 参数预处理 ==========
            if curve_cols is None:
                if data is not None:
                    curve_cols = data.columns[1:]       # 默认曲线列
            if type_cols is None:
                type_cols = []                      # 默认无分类列
            if colors is None:
                colors = self.DEFAULT_CURVE_COLORS  # 默认颜色序列
            if legend_dict is None:
                legend_dict = {}  # 默认空图例

            # ========== 参数验证 ==========
            self._validate_input_parameters(data, depth_col, curve_cols, type_cols)
            self._validate_fmi_data(fmi_dict)

            # ========== 数据初始化 ==========
            self.data = data.sort_values(depth_col).reset_index(drop=True)  # 按深度排序
            self.depth_col = depth_col
            self.curve_cols = curve_cols
            self.type_cols = type_cols
            self.fmi_dict = fmi_dict
            self.depth_limit_config = depth_limit_config

            # ========== 数据预处理 ==========
            self._setup_depth_limits(depth_limit_config)  # 设置深度范围
            self._setup_lithology_width_config()  # 设置岩性宽度配置

            # ========== 计算显示参数 ==========
            depth_range = self.depth_max - self.depth_min
            self.window_size = depth_range * 0.4  # 初始窗口大小：一半范围
            self.depth_position = self.depth_min  # 初始位置：最浅部

            # 计算深度分辨率（采样间隔）
            self.resolution = get_resolution_by_depth(self.data[depth_col].dropna().values)

            # ========== 图形设置 ==========
            n_plots = self._calculate_subplot_count()  # 计算子图总数
            has_legend = bool(legend_dict)  # 检查是否有图例
            self._setup_figure_layout(figure, n_plots, has_legend, figsize)  # 设置布局
            self._create_window_size_slider(has_legend)  # 创建滑动条

            # ========== 优化和绘制 ==========
            self._optimize_fmi_rendering()  # FMI图像优化
            self._plot_all_fmi_panels()  # 绘制FMI面板
            self._plot_all_curves(colors)  # 绘制曲线面板
            self._plot_all_classification_panels()  # 绘制分类面板

            # ========== 交互功能 ==========
            self.window_size_slider.on_changed(self._on_window_size_change)  # 滑动条回调
            self.fig.canvas.mpl_connect('scroll_event', self._on_mouse_scroll)  # 滚轮事件
            self._create_legend_panel(legend_dict)  # 创建图例

            # ========== 初始显示 ==========
            self._update_display()  # 执行首次显示更新

            # ========== 性能统计 ==========
            total_time = time.time() - start_time
            logger.info("可视化完成，耗时: %.2fs", total_time)
            plt.show()  # 显示图形

        except Exception as e:
            # ========== 异常处理 ==========
            logger.error("可视化失败: %s", str(e))
            if self.fig:
                plt.close(self.fig)  # 关闭图形释放资源
            raise  # 重新抛出异常

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息

        返回内容：
        - 渲染次数、缓存命中/未命中次数
        - 缓存命中率、当前缓存大小
        - 各类曲线数量统计
        """
        total_attempts = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = (self._cache_stats['hits'] / total_attempts * 100) if total_attempts > 0 else 0

        stats = {
            'total_renders': self._cache_stats['total_renders'],  # 总渲染次数
            'cache_hits': self._cache_stats['hits'],  # 缓存命中次数
            'cache_misses': self._cache_stats['misses'],  # 缓存未命中次数
            'cache_hit_rate': hit_rate,  # 缓存命中率
            'cache_size': len(self._data_cache),  # 当前缓存大小
            'imaging_curves_count': len(self.fmi_axes) if self.fmi_dict else 0,  # FMI曲线数
            'regular_curves_count': len(self.curve_cols)  # 常规曲线数
        }

        # 记录性能统计日志
        logger.info("性能统计: 渲染%d次, 缓存命中率%.1f%%, 缓存大小%d",
                    stats['total_renders'], hit_rate, stats['cache_size'])
        return stats

    def close(self) -> None:
        """关闭可视化器：释放图形资源和缓存"""
        if self.fig:
            plt.close(self.fig)  # 关闭matplotlib图形
            self.fig = None
            logger.info("可视化器已关闭")

        """清空缓存：释放内存并重置统计"""
        self._data_cache.clear()  # 清空缓存字典
        self._cache_stats = {'hits': 0, 'misses': 0, 'total_renders': 0}  # 重置统计
        logger.info("缓存已清空")



# ==================== 使用示例和测试代码 ====================
if __name__ == "__main__":
    """
    示例用法：生成模拟测井数据并演示可视化功能
    """
    # 生成示例测试数据
    print("生成示例测井数据...")
    n_points = 2000  # 数据点数量

    # 创建示例测井数据（模拟真实测井数据特征）
    sample_data = pd.DataFrame({
        'Depth': np.linspace(300, 400, n_points),  # 深度：300-400米均匀分布
        'GR': np.random.normal(60, 15, n_points),  # 伽马射线：正态分布
        'GR_X': np.random.normal(60, 15, n_points),  # 方向X伽马射线
        'GR_Y': np.random.normal(60, 15, n_points),  # 方向Y伽马射线
        'RT': np.random.lognormal(2, 0.3, n_points),  # 电阻率：对数正态分布
        'RX': np.random.lognormal(2, 0.3, n_points),  # 电阻率
        'NPHI': np.random.uniform(0.15, 0.45, n_points),  # 中子孔隙度：均匀分布
        'RHOB': np.random.uniform(2.3, 2.7, n_points),  # 体积密度：均匀分布
        'LITHOLOGY': np.random.choice([0, 1, 2, 3], n_points, p=[0.7, 0.1, 0.1, 0.1]),  # 岩性分类
        'FACIES': np.random.choice([0, 1, 2], n_points, p=[0.5, 0.3, 0.2])  # 相分类
    })

    # 准备FMI示例数据（截取中间部分避免边界效应）
    depth_fmi = sample_data.Depth.values[200:-200]
    print(f"FMI深度数据形状: {depth_fmi.shape}, 起始深度: {depth_fmi[0]}, 结束深度: {depth_fmi[-1]}")

    # 生成示例FMI图像数据（小图像扩展到完整深度范围）
    FMI_RAND = np.random.randint(0, 256, size=[20, 8], dtype=np.uint8)

    # 使用不同插值方法创建FMI图像
    fmi_dynamic = cv2.resize(FMI_RAND, (256, depth_fmi.shape[0]), interpolation=cv2.INTER_NEAREST)
    fmi_static = cv2.resize(FMI_RAND, (256, depth_fmi.shape[0]), interpolation=cv2.INTER_CUBIC)
    print(f"FMI动态图像形状: {fmi_dynamic.shape}, FMI静态图像形状: {fmi_static.shape}")

    # 使用类接口进行可视化
    print("创建可视化器...")
    visualizer = WellLogVisualizer()

    try:
        # 启用详细日志级别
        logging.getLogger().setLevel(logging.INFO)

        # 执行可视化
        visualizer.visualize(
            data=sample_data,
            depth_col='Depth',
            curve_cols=[['GR', 'GR_X', 'GR_Y'], ['RT', 'RX'], 'NPHI', 'RHOB'],  # 选择显示的曲线
            type_cols=['LITHOLOGY', 'FACIES'],  # 分类数据
            legend_dict={  # 图例定义
                0: '砂岩',
                1: '页岩',
                2: '石灰岩',
                3: '白云岩'
            },
            fmi_dict={  # FMI图像数据
                'depth': depth_fmi,
                'image_data': [fmi_dynamic, fmi_static],
                'title': ['FMI动态', 'FMI静态']
            },
            # fmi_dict=None,  # 注释掉此行可测试无FMI的情况
            # depth_limit_config=[320, 380],  # 可取消注释测试深度限制
            figsize=(12, 8)  # 图形尺寸
        )

        # 显示性能统计
        stats = visualizer.get_performance_stats()
        print("性能统计:", stats)

    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback

        traceback.print_exc()  # 打印完整错误堆栈
    finally:
        # 清理资源
        visualizer.close()
