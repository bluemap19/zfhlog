import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
import os
import warnings
# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)

def set_ultimate_chinese_font():
    """
    终极中文解决方案：确保所有中文正常显示
    优先使用系统字体，失败则回退到默认字体
    """
    try:
        # Windows 系统中的常见中文字体文件路径
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
            'C:/Windows/Fonts/simkai.ttf',  # 楷体
        ]

        # Linux/MacOS 中的常见中文字体路径
        if os.name != 'nt':
            font_paths = [
                '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                '/System/Library/Fonts/PingFang.ttc',
                '/Library/Fonts/Arial Unicode.ttf'
            ]

        # 查找第一个可用的字体文件
        font_file = None
        for path in font_paths:
            if os.path.exists(path):
                font_file = path
                print(f"✅ 找到字体文件: {path}")
                break

        if font_file:
            # 添加字体属性
            font_prop = fm.FontProperties(fname=font_file)
            font_name = font_prop.get_name()

            # 核心设置
            plt.rcParams['font.family'] = [font_name, 'sans-serif']
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial Unicode MS']

            # 用于检查的示例文本
            test_text = "测试中文字体: 窗长 均 类"
            fig, ax = plt.subplots(figsize=(2, 1))
            ax.text(0.5, 0.5, test_text, ha='center', va='center', fontproperties=font_prop)
            plt.savefig('chinese_test.png', dpi=100)
            plt.close(fig)
            print(f"✅ 中文测试图已保存，请检查 'chinese_test.png'")

        # 设置通用的中文相关参数
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 返回用于特定元素的字体属性
        return font_prop if font_file else None

    except Exception as e:
        print(f"⚠ 设置字体时出错: {str(e)}")
        # 安全后备方案
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
        return None



def create_acc_heatmap(data_df, curve_columns, index_plot='窗长',
                       label_plot={'label':'Heatmap of different windows length', 'x':'Windows Length', 'y':'Type', 'heatmap_feature':'影响因子'},
                       save_plot=False):
    """
    创建丰富的热力图可视化

    参数:
    data_df : DataFrame
        包含窗长和各类别准确率的数据框
    curve_columns : list
        包含各类别准确率列名的列表
    """
    # 设置样式# 1. 设置字体 - 使用终极方案
    chinese_font_prop = set_ultimate_chinese_font()

    # 2. 准备数据
    heatmap_data = data_df.set_index(index_plot)[curve_columns]

    # 3. 创建基础图形
    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(
        heatmap_data.T,
        annot=True,
        fmt=".2f",
        # cmap="RdYlGn",
        cmap="bwr",
        vmin=0.0,
        vmax=1.0,
        linewidths=.5,
        cbar_kws={'label': label_plot['heatmap_feature']},
        annot_kws={'size': 12}
    )

    # 4. 显式设置所有文本的字体属性
    if chinese_font_prop:
        # 标题
        ax.set_title(label_plot['label'], fontproperties=chinese_font_prop, pad=20, fontsize=16)

        # 轴标签
        ax.set_xlabel(label_plot['x'], fontproperties=chinese_font_prop, labelpad=15, fontsize=14)
        ax.set_ylabel(label_plot['y'], fontproperties=chinese_font_prop, labelpad=15, fontsize=14)

        # X轴刻度
        for label in ax.get_xticklabels():
            label.set_fontproperties(chinese_font_prop)
            label.set_rotation(45)
            label.set_ha('right')

        # Y轴刻度
        for label in ax.get_yticklabels():
            label.set_fontproperties(chinese_font_prop)
            label.set_rotation(45)
            label.set_ha('right')

        # 颜色条标签
        cbar = ax.collections[0].colorbar
        cbar.set_label(label_plot['heatmap_feature'], fontproperties=chinese_font_prop)
        for label in cbar.ax.get_yticklabels():
            label.set_fontproperties(chinese_font_prop)

    # 5. 优化布局和保存
    plt.tight_layout()

    if save_plot:
        # 尝试保存为PDF（支持矢量字体）
        try:
            plt.savefig('heatmap_chinese.pdf', format='pdf', bbox_inches='tight')
            print("✅ 热力图已保存为PDF: heatmap_chinese.pdf")
        except:
            # 保存为PNG（位图）
            plt.savefig('heatmap_chinese.png', dpi=300, bbox_inches='tight')
            print("✅ 热力图已保存为PNG: heatmap_chinese.png")

    plt.show()
    return ax


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    # 创建示例数据
    data = {
        '窗长': [200, 220, 240, 260, 280, 300],
        '类别0': [0.85, 0.88, 0.90, 0.89, 0.87, 0.84],
        '类别1': [0.78, 0.82, 0.85, 0.83, 0.80, 0.78],
        '类别2': [0.92, 0.91, 0.93, 0.94, 0.92, 0.90],
        '平均': [0.85, 0.87, 0.89, 0.88, 0.86, 0.84]
    }
    df = pd.DataFrame(data)

    # 定义标签
    plot_labels = {
        'label': '不同窗长下的精度热力图',
        'x': '窗长参数',
        'y': '精度类别',
        'heatmap_feature':'准确率'
    }

    # 创建热力图
    create_acc_heatmap(
        data_df=df,
        curve_columns=['类别0', '类别1', '类别2', '平均'],
        label_plot=plot_labels,
        save_plot=False
    )