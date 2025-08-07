import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src_file_op.dir_operation import get_all_file_paths
from src_file_op.xlsx_file_read import get_data_from_pathlist

if __name__ == '__main__':

    path_folder = r'C:\Users\ZFH\Documents\simu-result'
    path_file_list = get_all_file_paths(root_dir=path_folder)
    Curve_List_imp = [
                'STAT_CON', 'STAT_DIS', 'STAT_HOM', 'STAT_ENG', 'STAT_COR', 'STAT_ASM', 'STAT_ENT', 'STAT_XY_CON',
                'STAT_XY_DIS', 'STAT_XY_HOM', 'STAT_XY_ENG', 'STAT_XY_COR', 'STAT_XY_ASM', 'STAT_XY_ENT',
                'DYNA_CON', 'DYNA_DIS', 'DYNA_HOM', 'DYNA_ENG', 'DYNA_COR', 'DYNA_ASM', 'DYNA_ENT', 'DYNA_XY_CON',
                'DYNA_XY_DIS', 'DYNA_XY_HOM', 'DYNA_XY_ENG', 'DYNA_XY_COR', 'DYNA_XY_ASM', 'DYNA_XY_ENT'
    ]
    Config_logging = {'file_name':'influence_win', 'curve_name':Curve_List_imp, 'sheet_name':0}
    data_importance_feature = get_data_from_pathlist(path_file_list, Config_logging)
    print(data_importance_feature.shape, data_importance_feature.describe())

    Curve_List_acc = ['Type1', 'Type2', 'Type3', 'Type4', 'Type5', 'Mean']
    Config_logging = {'file_name':'influence_win', 'curve_name':Curve_List_acc, 'sheet_name':1}
    data_acc = get_data_from_pathlist(path_file_list, Config_logging)
    print(data_acc.shape, data_acc[:5, :7], data_acc[:, -1].shape)

    columns = []
    for i in range(data_acc.shape[0]):
        columns.append(str(int(data_acc[i, 0])))
    print(columns)


    # 方法1：通过reshape调整维度
    ACC_final_reshaped = data_acc[:, -1].reshape(-1, 1)  # 变为 (3,1)
    # 逐元素相乘
    data_importance_feature = data_importance_feature * ACC_final_reshaped  # 或 np.multiply(a1, a2_reshaped)
    # print(data_importance_feature.shape, data_importance_feature[:5, :7])

    df = pd.DataFrame(data_importance_feature,
                      index=columns,
                      columns=Curve_List_imp)
    # 计算全局极值
    global_min = df.min().min()  # 获取所有数据的最小值[4,6](@ref)
    global_max = df.max().max()  # 获取所有数据的最大值[4,6](@ref)
    # 全局归一化
    df_normalized = (df - global_min) / (global_max - global_min)
    # print(df.describe())

    # 方法1：使用loc筛选行 + mean计算列均值
    means = df_normalized.loc[['20', '40', '60', '80', '100', '120', '140', '160', '180', '200', '220']].mean(axis=0)  # axis=0表示逐列计算[4,8](@ref)
    df_normalized.loc['ACC_Mean'] = means  # 添加新行


    # Step 1: 转置DataFrame以便按列排序[7](@ref)
    df_t = df_normalized.T
    print(df_t.shape)
    # Step 2: 按指定行值降序排序(@ref) 最后一行
    df_sorted = df_t.sort_values(by=['ACC_Mean'], ascending=False, axis=0)
    # Step 3: 提取前a个列索引（原DataFrame的列名）
    top_columns = df_sorted.index[:12].tolist()
    print("排序后列顺序:", df_sorted.index.tolist())
    print("最大前10列索引:", top_columns)


    # # 绘制增强型热力图
    # plt.rcdefaults()
    # plt.figure(figsize=(15, 9))
    # sns.heatmap(df_normalized.loc[['100', '110', '120', '130', '140', '150', '160', '170', '180', 'ACC_Mean'], top_columns],
    #             annot=True,          # 显示数值[4,8](@ref)
    #             fmt=".2f",           # 数值格式
    #             cmap='Spectral',     # 光谱色系
    #             linewidths=0.1,      # 单元格边框线宽[6](@ref)
    #             square=True)         # 单元格保持正方形
    # plt.title("Importance Heatmap by window length")
    # plt.show()


    # 根据行列信息,进行柱状图数据定位,提取与转换
    selected_data = df_normalized.loc[['40', '60', '80', '100', 'ACC_Mean'], top_columns]
    melted_data = selected_data.reset_index().melt(
        id_vars='index',
        var_name='Column',
        value_name='Value'
    )
    # print(melted_data)
    # 可视化配置
    plt.figure(figsize=(12, 9))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(
        x='Column',
        y='Value',
        hue='index',
        data=melted_data,
        palette="Blues_d",
        alpha=0.8
    )
    # 标签优化
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.0f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 5),
                    textcoords='offset points')
    ax.tick_params(axis='x', labelsize=14)   # x轴刻度字号
    ax.tick_params(axis='y', labelsize=14)   # y轴刻度字号
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.title('Texture Influence Column Map')
    plt.xlabel('Texture Abbreviation', fontsize=16)
    plt.ylabel('Influence', fontsize=16)
    plt.legend(title='Label', fontsize=14)
    plt.show()




