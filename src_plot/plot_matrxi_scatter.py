from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from seaborn import color_palette
from typing import Optional
from src_random_data.create_random_data import get_random_logging_dataframe
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 这个函数主要功能是，绘制不同类别数据，输入属性的散布图
# 即，可视化输入属性对分类类别的影响能力
def plot_matrxi_scatter(df: pd.DataFrame = pd.DataFrame(),
                        input_names: list = [],
                        target_col='',
                        plot_string='',
                        target_col_dict: dict = {},
                        figure: Optional[plt.Figure] = None):
    """
    进行数据散步图绘制，制作不同类别数据分布
    :param df: 数据体，只能是dataframe
    :param input_names: 输入属性列名称，对应输入属性列
    :param target_col: 分类列名称，对应类别分类列信息
    :param plot_string: 图形名称title
    :param target_col_dict: 类别是int分类类别型，那就需要替换成 字符型，这个是替换字典信息
    :param figure: 画图类，在哪个 实体plt上进行画图
    :return:
    """


    # 数据预处理增强
    data = df[input_names].copy()
    # 判断target_col列是否是数值类型的数据
    if pd.api.types.is_numeric_dtype(df[target_col]):
        # 如果是空字典，初始化一个
        if not target_col_dict:
            target_col_dict = {0: 'type1', 1: 'type2', 2: 'type3', 3: 'type4',
                               4: 'type5', 5:'type6', 6: 'type7', 7: 'type8',
                               8: 'type9', 9: 'type10', 10: 'type11'}
            data[target_col] = df[target_col].astype(int)
            data[target_col] = data[target_col].map(target_col_dict)
            target_col_dict_temp = target_col_dict.copy()
        else:       # 不是空字典的话，需要判断键值对是不是反的
            target_col_dict_temp = target_col_dict.copy()
            # 判断 元素 0  是否在字典target_col_dict的键keys中，在的话说明键值对没有反，不在的话说明反了，要把反转的添加进去
            if 0 not in list(target_col_dict.keys()):
                dict_temp = {v: k for k, v in target_col_dict.items()}
                target_col_dict_temp = dict_temp
                print(target_col_dict_temp)
            # 先把类别数据转换成int格式
            data[target_col] = df[target_col].astype(int)
            # 再通过字典进行映射
            data[target_col] = data[target_col].map(target_col_dict_temp)
    # target_col列不是数值类型，也要把目标列加上
    else:
        data[target_col] = df[target_col]
        target_col_dict_temp = {}
        # 如果是空字典，初始化一个
        if not target_col_dict:
            all_type = list(np.unique(data[target_col]))
            for i in range(len(all_type)):
                target_col_dict_temp[i] = all_type[i]

    # 新建地质色谱（基于网页3和8的离散色谱方案）
    geology_colors = [
        '#4B0082',  # 中GR长英黏土质（靛青）
        '#32CD32',  # 中低GR长英质（亮绿）
        '#FF4500',  # 富有机质长英质（橙红）
        '#8B4513',  # 富有机质黏土质（赭石）
        '#00BFFF',  # 高GR富凝灰长英质（深天蓝）
        '#A9A9A9'  # 其他类型（灰色）
    ]
    palette = color_palette(geology_colors, n_colors=len(target_col_dict_temp))



    """
    data：DataFrame格式
    hue:将绘图的不同面映射为不同的颜色
    palette:调色板
    vars:使用data中的变量，否则使用一个数值型数据类型的每一列。
    height标量，可选，每个刻面的高度（以英寸为单位）指定每个子图的高度（单位为英寸）。
    aspect：标量，可选,aspect 和 height 的乘积得出每个刻面的宽度（以英寸为单位）,指定每个子图的宽高比。
    despine：布尔值，可选,从图中移除顶部和右侧脊柱。
    dropna：布尔值，可选,在绘图之前删除数据中的缺失值。
    """
    # AI看这里，请问我这样初始化我的画布对不对？如果不对的话，我应该怎么改？
    # 创建画布（使用传入的figure或新建）    # 创建PairGrid对象
    if figure is None:
        plt.figure(dpi=240)  # 显式设置画布尺寸
        # fig = plt.gcf()  # 获取当前图形对象
    else:
        fig = figure
        fig.clear()

    # height参数:控制每个子图的高度（以英寸为单位）
    # aspect参数:控制每个子图的宽高比（宽度/高度）

    g = sns.PairGrid(
        data,
        vars=input_names,
        hue=target_col,
        # palette='husl',  # 更专业的色系方案
        palette=palette,  # 应用自定义颜色列表
        hue_order=sorted(data[target_col].unique()),  # 固定类别顺序
        diag_sharey=False,
        height=2,
        aspect=1,
        # corner=True  # 隐藏重复的右上三角（当上下三角类型相同时）
    )

    # 上三角优化（KDE图）
    g.map_upper(
        sns.kdeplot,
        alpha=0.7,  # 透明度增强
        levels=15,  # 等高线密度
        fill=False,  # 填充颜色
        thresh=0.05,  # 显示阈值
    )

    # 下三角优化（散点图）
    g.map_lower(
        sns.scatterplot,
        s=18,  # 点大小（根据数据量调整）
        alpha=0.8,  # 透明度平衡
        edgecolor='w',  # 边缘色（增强对比）
        linewidth=0.5,  # 边缘线宽
        palette='husl',  # 与整体色系一致
        zorder=2  # 图层顺序
    )

    # 对角线优化（KDE图）
    g.map_diag(
        sns.kdeplot,
        fill=False,  # 填充颜色
        alpha=0.7,  # 填充透明度
        linewidth=1.8,  # 主轮廓线宽
        linestyle='-',  # 虚线样式
    )

    # 样式增强组件
    g.add_legend(
        title=target_col,  # 图例标题
        frameon=True,  # 图例外框
        # bbox_to_anchor=(1.01, 0.89),  # 图例位置（右上侧，外部）
        fontsize=10  # 标签字号
    )

    # 坐标轴统一美化
    for ax in (a for a in g.axes.flatten() if a is not None):  # 过滤空坐标轴
        # 设置四周边框线
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(True)  # 显示所有边框
            ax.spines[spine].set_color('#404040')  # 深灰色边框
            ax.spines[spine].set_linewidth(0.8)  # 边框线宽
        ax.grid(
            True,
            linestyle=':',  # 虚线网格
            alpha=0.3,  # 网格透明度
            color='gray',  # 网格颜色
            zorder=0  # 确保网格在数据层下方
        )
        ax.xaxis.label.set_size(12)  # X轴标签字号
        ax.yaxis.label.set_size(12)  # Y轴标签字号
        # 刻度线专业设置（重点修改部分）
        ax.tick_params(
            which='both',  # 同时控制主/次刻度
            direction='in',  # 关键参数：刻度线朝内
            length=3,  # 适当缩短刻度线长度
            width=0.8,  # 保持线宽与边框协调
            color='#606060',  # 刻度线颜色与边框统一
            labelsize=9,
            rotation=45,
            grid_alpha=0.6,
            # top=True,  # 显示顶部刻度
            # right=True,  # 显示右侧刻度
            # bottom=True,  # 确保底部刻度可见
            # left=True  # 确保左侧刻度可见
        )

    # 标题优化（添加编号和注释）
    plt.suptitle('{}矩阵分析（下三角：分类散点图 | 上三角：KDE等高线图）'.format(plot_string),
                 y=1.02,
                 fontsize=12,)


    # 添加版权标识（可选）
    # 在版权信息前插入时间戳生成代码
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 生成ISO格式时间戳[1,5](@ref)
    plt.gcf().text(
        0.75, 0.01,
        f'© ZFH UPC&CAS Data Visualization {timestamp_str}',
        fontdict={
            'fontname': 'DejaVu Sans',  # 优先使用支持符号的字体
            'fontsize': 8,
            'color': 'gray',
            'alpha': 0.5
        },
        # fontsize=8,
        # color='gray',
        # alpha=0.5
    )
    plt.show()


if __name__ == '__main__':
    logging_data = get_random_logging_dataframe(curve_name=['#DEPTH', 'SP', 'AC', 'CNL', 'DEN', 'Type'], logging_resolution=0.1, dep_start=100, dep_end=500)
    print(logging_data.shape)           # ----->(4000, 6)
    plot_matrxi_scatter(logging_data, ['AC', 'CNL', 'DEN', 'SP'], 'Type', target_col_dict={'中GR长英黏土质': 0, '中低GR长英质': 1, '富有机质长英质': 2, '富有机质黏土质': 3, '高GR富凝灰长英质': 4, 'asdadsadas':5})
