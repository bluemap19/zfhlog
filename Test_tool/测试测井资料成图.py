import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QScrollArea)
# 设置随机种子保证可重复性
np.random.seed(2023)

def get_units(self, col):
    """获取单位说明（补充完整单位）"""
    units = {
        'GR': 'Gamma Ray (API)',
        'SP': 'Spontaneous Potential (mV)',
        'CAL': 'Caliper (inch)',
        'DT24': 'Sonic Travel Time (μs/ft)',
        'AC': 'Sonic Travel Time (μs/ft)',
        'CNL': 'Neutron Porosity (%)', 
        'DEN': 'Bulk Density (g/cm³)'
    }
    return units.get(col, '')

def generate_well_logs(samples=2000):
    """
    生成模拟测井数据DataFrame
    参数：
        samples : 数据点数量（默认2000个）
    返回：
        pd.DataFrame : 包含模拟测井数据的DataFrame
    """
    # 生成深度列（1000米开始，0.5米间隔）
    depth = np.linspace(1000, 1000 + 0.5 * (samples - 1), samples)

    # 预先生成DEN列（避免后续引用时未定义）
    DEN = np.concatenate([
        np.random.normal(2.3, 0.05, samples // 4),
        np.random.normal(2.6, 0.08, samples // 4),
        np.random.normal(2.45, 0.1, samples // 4),
        np.random.normal(2.7, 0.12, samples - 3 * (samples // 4))
    ])

    # 生成测井数据列（添加趋势项和噪声）
    data = {
        'DEPTH': depth,
        # 自然伽马（API）: 带周期性变化的正态分布
        'GR': np.clip(50 + 40 * np.sin(np.linspace(0, 10 * np.pi, samples))
                      + 15 * np.random.randn(samples), 0, 150),

        # 自然电位（mV）: 分段变化的三角波
        'SP': np.where(depth < 1200,
                       np.random.uniform(-80, 20, samples),
                       np.random.uniform(-20, 100, samples)) * 0.8,

        # 井径（英寸）: 随深度逐渐增大
        'CAL': np.clip(8 + 0.002 * (depth - 1000)
                       + 1.5 * np.random.randn(samples), 6, 16),

        # 密度（g/cm³）: 已预先生成
        'DEN': DEN,

        # 中子孔隙度（%）: 使用预先生成的DEN变量
        'CNL': np.clip(30 - 10 * (DEN - 2.4) + 5 * np.random.randn(samples), 0, 40),

        # 声波时差（μs/ft）: 同样使用DEN变量
        'AC': np.clip(100 - 30 * (DEN - 2.4) + 15 * np.random.randn(samples), 50, 150)
    }

    # 生成分类列
    df = pd.DataFrame(data)

    # 原始类型（假设5种岩性）
    df['Type'] = np.random.choice([0, 1, 2, 3, 4, 5], size=samples, p=[0.2, 0.1, 0.2, 0.25, 0.15, 0.1])

    # 各算法分类结果（添加相关性）
    df['NLP'] = np.where(df['Type'].isin([2, 4]),
                         np.random.binomial(1, 0.8, samples),
                         np.random.binomial(1, 0.3, samples))
    df['KNN'] = (df['DEN'] > 2.5).astype(int) ^ (df['CNL'] < 25).astype(int)
    df['SVM'] = np.random.binomial(1, 0.6 - 0.1 * df['Type'], samples)
    df['Native Bayes'] = (df['NLP'] | df['SVM']) & (df['KNN'] ^ (df['Type'] % 2))
    df['Random Forest'] = (df['NLP'] | df['SVM']) & (df['KNN'] ^ (df['Type'] % 2))
    df['GBM'] = (df['NLP'] | df['SVM']) & (df['KNN'] ^ (df['Type'] % 2))
    # , 'Type', 'NLP', 'KNN', 'SVM', 'Native Bayes', 'Random Forest', 'GBM'
    # 调整列顺序
    return df[['DEPTH', 'AC', 'CNL', 'DEN', 'GR', 'SP', 'CAL', 'Type', 'NLP', 'KNN', 'SVM', 'Native Bayes', 'Random Forest', 'GBM']]


# # 生成数据
# well_df = generate_well_logs()
#
# # 数据预览
# print("数据维度:", well_df.shape)
# print("\n前5行数据预览:")
# print(well_df.head())




class WellLogViewer(QMainWindow):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.initUI()

    def initUI(self):
        self.setWindowTitle('测井数据综合展示系统')
        self.setGeometry(100, 100, 1600, 900)

        # 创建主容器
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 创建Matplotlib画布
        self.fig = plt.Figure(figsize=(16, 12), dpi=100)
        self.canvas = FigureCanvas(self.fig)

        # 添加导航工具栏
        toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(toolbar)

        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.canvas)
        layout.addWidget(scroll)

        # 绘制测井道
        self.plot_well_logs()

    def plot_well_logs(self):
        """绘制所有测井道和分类道"""
        # 清除旧图
        self.fig.clf()

        # 设置子图布局
        n_tracks = len(self.df.columns) - 1  # 减去深度列
        axes = []

        # 创建共享Y轴的子图
        for i in range(n_tracks):
            if i == 0:
                ax = self.fig.add_subplot(1, n_tracks, i + 1)
                ax.set_ylabel("DEPTH (m)", fontsize=10)
            else:
                ax = self.fig.add_subplot(1, n_tracks, i + 1, sharey=axes[0])

            axes.append(ax)

        # 设置紧凑布局
        self.fig.subplots_adjust(left=0.03, right=0.97,
                                 wspace=0,  # 道间距设置为0
                                 bottom=0.08, top=0.92)

        # 绘制每个道的数据
        for i, col in enumerate(self.df.columns[1:]):  # 跳过深度列
            ax = axes[i]

            # 设置道标题
            ax.set_title(col, fontsize=12, pad=8)

            # 设置X轴范围
            if col in ['AC', 'CNL', 'DEN', 'GR', 'SP', 'CAL']:
                self.plot_curve(ax, col)
            else:
                self.plot_classification(ax, col)

            # 隐藏Y轴标签（除第一道外）
            if i != 0:
                plt.setp(ax.get_yticklabels(), visible=False)

            # 设置网格线
            ax.grid(True, linestyle='--', alpha=0.5)

        self.canvas.draw()

    # def plot_curve(self, ax, col):
    #     """绘制测井曲线道"""
    #     ax.plot(self.df[col], self.df['DEPTH'],
    #             linewidth=1.0, color=self.get_curve_color(col))
    #
    #     # 设置X轴范围
    #     min_val = self.df[col].min() * 0.9
    #     max_val = self.df[col].max() * 1.1
    #     ax.set_xlim(left=min_val, right=max_val)
    #
    #     # 设置X轴标签
    #     ax.xaxis.set_label_position('top')
    #     ax.set_xlabel(self.get_units(col), fontsize=9)
    def plot_curve(self, ax, col):
        """绘制测井曲线道（修改图头部分）"""
        # 获取数据统计信息
        data_min = self.df[col].min()  # 实际最小值
        data_max = self.df[col].max()  # 实际最大值
        unit = self.get_units(col)  # 获取单位

        # 生成多行标题文本
        title_text = f"{col}\n{unit}\n[{data_min:.1f}-{data_max:.1f}]"

        # 设置图头样式
        ax.set_title(title_text,
                     fontsize=9,  # 缩小字体
                     pad=4,  # 减小标题与图的间距
                     loc='center',  # 居中显示
                     color='#333333',  # 深灰色字体
                     fontweight='bold')

        # 绘制曲线（保持原有逻辑）
        ax.plot(self.df[col], self.df['DEPTH'],
                linewidth=1.0,
                color=self.get_curve_color(col))

        # 设置X轴范围（保持原有逻辑）
        min_val = self.df[col].min() * 0.9
        max_val = self.df[col].max() * 1.1
        ax.set_xlim(left=min_val, right=max_val)

        # 隐藏X轴标签（因信息已显示在标题中）
        ax.set_xticklabels([])
        ax.set_xlabel('')  # 清空原有单位标签

    # def plot_classification(self, ax, col):
    #     """绘制分类结果道"""
    #     # 生成颜色映射
    #     unique_classes = self.df[col].unique()
    #     cmap = plt.get_cmap('tab10', len(unique_classes))
    #
    #     # 绘制分类结果
    #     depth = self.df['DEPTH'].values
    #     y = np.zeros_like(depth)
    #
    #     for i, cls in enumerate(sorted(unique_classes)):
    #         mask = self.df[col] == cls
    #         ax.fill_betweenx(depth[mask], y[mask], i + 1,
    #                          color=cmap(i), alpha=0.6, label=str(cls))
    #
    #     ax.set_xlim(0, len(unique_classes) + 1)
    #     ax.set_xticks([])
    #
    #     # 添加图例
    #     ax.legend(loc='upper right', fontsize=8,
    #               framealpha=0.8, title='Class')
    def plot_classification(self, ax, col):
        """优化后的分类结果道绘制方法"""
        # 获取唯一类别并排序
        unique_classes = sorted(self.df[col].unique())
        num_classes = len(unique_classes)

        # 设置颜色映射
        cmap = plt.get_cmap('tab10', num_classes)

        # 设置水平刻度范围
        class_width = 1.0  # 每个类别的宽度
        ax.set_xlim(0, num_classes * class_width)

        # 清除垂直网格线
        ax.grid(False)

        # 绘制每个类别的填充区域
        depth = self.df['DEPTH'].values
        for i, cls in enumerate(unique_classes):
            # 计算水平位置
            x_start = i * class_width
            x_end = (i + 1) * class_width

            # 创建掩码
            mask = self.df[col] == cls

            # 绘制填充区域
            ax.fill_betweenx(depth[mask],
                             x_start, x_end,
                             color=cmap(i),
                             alpha=0.6,
                             label=f'Class {cls}',
                             edgecolor='none')

        # 设置刻度
        ax.set_xticks([(i + 0.5) * class_width for i in range(num_classes)])
        ax.set_xticklabels([str(c) for c in unique_classes], fontsize=8)

        # # 添加图例
        # ax.legend(loc='lower right',
        #           fontsize=7,
        #           title='Categories',
        #           title_fontsize=8,
        #           framealpha=0.9,
        #           ncol=2 if num_classes > 5 else 1)

        # 添加网格线
        ax.grid(axis='x', linestyle=':', alpha=0.5)

    def get_curve_color(self, col):
        """获取曲线颜色"""
        colors = {
            'GR': '#FF4500',  # 橙色
            'SP': '#1E90FF',  # 蓝色
            'CAL': '#2E8B57',  # 绿色
            'DEN': '#8B0000',  # 红色
            'CNL': '#8A2BE2',  # 紫色
            'DT24': '#FF8C00'  # 橙色
        }
        return colors.get(col, '#000000')

    def get_units(self, col):
        """获取单位说明"""
        units = {
            'GR': 'API',
            'SP': 'mV',
            'CAL': 'inch',
            'DEN': 'g/cm³',
            'CNL': '%',
            'DT24': 'μs/ft'
        }
        return f"({units.get(col, '')})"


if __name__ == '__main__':
    # 生成示例数据（使用之前生成的DataFrame）
    # df = pd.read_csv('simulated_well_logs.csv')

    # 测试数据
    samples = 500
    test_df = generate_well_logs()
    print(test_df.describe())

    app = QApplication(sys.argv)
    viewer = WellLogViewer(test_df)
    viewer.show()
    sys.exit(app.exec_())


