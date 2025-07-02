import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src_file_op.dir_operation import search_files_by_criteria
from src_plot.plot_heatmap import create_acc_heatmap

if __name__ == '__main__':
    path_folder = r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV'
    path_file_list = search_files_by_criteria(search_root=path_folder, name_keywords=['IMP', 'ALL'], file_extensions=['csv', 'xlsx'])
    Curve_IMP_List = [
        'STAT_CON', 'STAT_DIS', 'STAT_HOM', 'STAT_ENG', 'STAT_COR', 'STAT_ASM', 'STAT_ENT', 'STAT_XY_CON',
        'STAT_XY_DIS', 'STAT_XY_HOM', 'STAT_XY_ENG', 'STAT_XY_COR', 'STAT_XY_ASM', 'STAT_XY_ENT',
        'DYNA_CON', 'DYNA_DIS', 'DYNA_HOM', 'DYNA_ENG', 'DYNA_COR', 'DYNA_ASM', 'DYNA_ENT', 'DYNA_XY_CON',
        'DYNA_XY_DIS', 'DYNA_XY_HOM', 'DYNA_XY_ENG', 'DYNA_XY_COR', 'DYNA_XY_ASM', 'DYNA_XY_ENT'
    ]
    print(path_file_list)

    IMP_MATRIX = pd.read_excel(path_file_list[0], sheet_name=0)
    print(IMP_MATRIX.head())

    # 这里是整体上看一下ACC在 窗长-随机森林参数 上的分布特征
    # 0. 按窗长分组计算均值
    window_stats = IMP_MATRIX.groupby('窗长')[Curve_IMP_List].mean().reset_index()
    print(window_stats)
    # 0. 按 随机森林参数 分组计算所有Curve_ACC_List选项的均值
    tree_stats = IMP_MATRIX.groupby('随机森林参数')[Curve_IMP_List].mean().reset_index()
    print(tree_stats)

    # 1. 过滤数据：窗长220-280且随机森林参数3-9
    filtered_df = IMP_MATRIX[
        (IMP_MATRIX['窗长'] >= 80) &
        (IMP_MATRIX['窗长'] <= 150) &
        (IMP_MATRIX['随机森林参数'] >= 3) &
        (IMP_MATRIX['随机森林参数'] <= 9)
        ]

    # print(filtered_df.head())
    Curve_target_names = [
        # 'STAT_CON', 'STAT_ENT', 'STAT_HOM', 'DYNA_DIS', 'STAT_XY_HOM', 'DYNA_HOM',     # 静态数据筛选出来的 6个
        # 'STAT_XY_COR', 'STAT_ENG', 'DYNA_COR', 'DYNA_XY_ENT', 'DYNA_XY_DIS', 'DYNA_ENG'      # 动态数据筛选出来的 6个
        # 'STAT_CON', 'STAT_ENT', 'STAT_HOM', 'STAT_XY_CON', 'DYNA_DIS', 'STAT_XY_HOM',      # 整体数据筛选出来的 18个
        # 'STAT_DIS', 'DYNA_HOM', 'STAT_XY_COR', 'STAT_ENG',
        # 'STAT_COR', 'DYNA_CON', 'DYNA_ENG', 'STAT_XY_ENG', 'STAT_XY_ASM'

        # 'STAT_ENT', 'STAT_DIS', 'STAT_CON', 'STAT_XY_HOM', 'STAT_HOM', 'STAT_XY_CON',
        # 'DYNA_DIS', 'STAT_ENG', 'DYNA_CON', 'STAT_XY_COR', 'DYNA_HOM', 'STAT_XY_ENG',
        # 'STAT_COR', 'DYNA_ENG', 'STAT_XY_ASM'
    ]
        # 'STAT_XY_HOM', 'STAT_HOM', 'STAT_ENT', 'STAT_ENG', 'STAT_CON', 'STAT_XY_COR',     # 静态数据筛选出来的 6个
        # 'DYNA_DIS', 'DYNA_HOM', 'DYNA_COR', 'DYNA_XY_DIS', 'DYNA_ENG', 'DYNA_XY_ENT'      # 动态数据筛选出来的 6个
    # 'STAT_CON', 'STAT_DIS', 'STAT_HOM', 'STAT_ENG', 'STAT_COR', 'STAT_ASM',  # 整体数据筛选出来的 18个
    # 'STAT_ENT', 'STAT_XY_CON', 'STAT_XY_DIS', 'STAT_XY_HOM', 'STAT_XY_ENG', 'STAT_XY_COR',
    # 'STAT_XY_ASM', 'STAT_XY_ENT', 'DYNA_CON', 'DYNA_DIS', 'DYNA_HOM', 'DYNA_ENG'

    # 2. 计算所需特征的均值
    # 按 窗长 分组计算均值
    window_grouped = filtered_df.groupby('窗长')[Curve_target_names].mean().reset_index()
    # 按 随机森林参数 分组计算均值
    param_grouped = filtered_df.groupby('随机森林参数')[Curve_target_names].mean().reset_index()
    # 3. 计算 准确率 整体矩阵的均值
    overall_mean = filtered_df[Curve_target_names].mean()
    print(window_grouped)
    print(param_grouped)
    print(overall_mean)

    # 1. 将平均值Series数据体 overall_mean 转换为DataFrame类型并排序
    df = overall_mean.to_frame(name='均值').reset_index()
    df.rename(columns={'index': '特征'}, inplace=True)
    # 按均值降序排序
    df_sorted = df.sort_values(by='均值', ascending=False)
    # 添加排名列
    df_sorted['排名'] = range(1, len(df_sorted) + 1)
    print(df_sorted)

    # 计算全局极值
    global_min = window_grouped[Curve_target_names].min().min()  # 获取所有数据的最小值[4,6](@ref)
    global_max = window_grouped[Curve_target_names].max().max()  # 获取所有数据的最大值[4,6](@ref)
    # 全局归一化
    df_normalized = window_grouped.copy()
    df_normalized[Curve_target_names] = (window_grouped[Curve_target_names] - global_min) / (global_max - global_min)
    # print(df.describe())


    # 4.绘图, 创建热力图
    create_acc_heatmap(df_normalized, Curve_target_names,
                       label_plot={'label': 'Heatmap of different windows length', 'x': 'Windows Length', 'y': 'Feature', 'heatmap_feature':'Influence Factor'}
                       )


    # # 文件路径设置
    # path_folder = r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV'
    # # 搜索目标文件路径
    # path_file_list = search_files_by_criteria(search_root=path_folder, name_keywords=['ACC', 'ALL'], file_extensions=['csv', 'xlsx'])
    # # 准确率ACC文件读取
    # ACC_MATRIX = pd.read_excel(path_file_list[0], sheet_name=0)
    # # 曲线信息设置
    # Curve_ACC_List = [
    #     '类别0', '类别1', '类别2', '类别3', '类别4', '平均'
    # ]
    # print(ACC_MATRIX.head())
    #
    # # 这里是整体上看一下ACC在 窗长-随机森林参数 上的分布特征
    # # 0. 按 窗长 分组计算所有Curve_ACC_List选项的均值
    # window_stats = ACC_MATRIX.groupby('窗长')[Curve_ACC_List].mean().reset_index()
    # print(window_stats)
    # # 0. 按 随机森林参数 分组计算所有Curve_ACC_List选项的均值
    # tree_stats = ACC_MATRIX.groupby('随机森林参数')[Curve_ACC_List].mean().reset_index()
    # print(tree_stats)
    #
    # # 1. 过滤数据：窗长220-280且随机森林参数3-9
    # filtered_df = ACC_MATRIX[
    #     (ACC_MATRIX['窗长'] >= 220) &
    #     (ACC_MATRIX['窗长'] <= 280) &
    #     (ACC_MATRIX['随机森林参数'] >= 3) &
    #     (ACC_MATRIX['随机森林参数'] <= 9)
    #     ]
    #
    # # 2. 计算所需特征的均值
    # # 按 窗长 分组计算均值
    # window_grouped = filtered_df.groupby('窗长')[Curve_ACC_List].mean().reset_index()
    # # 按 随机森林参数 分组计算均值
    # param_grouped = filtered_df.groupby('随机森林参数')[Curve_ACC_List].mean().reset_index()
    # # 3. 计算 准确率 整体矩阵的均值
    # overall_mean = filtered_df[Curve_ACC_List].mean()
    # print(window_grouped)
    # print(param_grouped)
    # print(overall_mean)
    # # 4.绘图, 创建热力图
    # create_acc_heatmap(window_grouped, Curve_ACC_List,
    #                    label_plot={'label': 'Heatmap of different windows length', 'x': 'Windows Length', 'y': 'Type'}
    #                    )
    exit(0)








