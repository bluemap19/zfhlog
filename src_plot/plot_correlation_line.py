import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch


# 1. 修复后的数据准备函数
def prepare_data(env_path, spec_path):
    # 读取环境因子数据（行为环境因子，列为样本）
    env_data = pd.read_csv(env_path, index_col=0)
    env_df = env_data.T  # 转置后：行为样本，列为环境因子

    # 读取物种丰度数据（行为样本，列为物种）
    spec_data = pd.read_csv(spec_path)
    group_info = spec_data['Group']
    spec_df = spec_data.drop('Group', axis=1).set_index(spec_data.columns[0])

    # 统一索引格式（处理可能的格式差异）
    env_df.index = env_df.index.astype(str).str.strip().str.upper()
    spec_df.index = spec_df.index.astype(str).str.strip().str.upper()

    # 确保样本顺序一致
    common_idx = env_df.index.intersection(spec_df.index)
    return env_df.loc[common_idx], spec_df.loc[common_idx], group_info


# 计算两个距离矩阵的 Mantel 统计量
def mantel(matrix1, matrix2, permutations=1000):
    flat1 = squareform(matrix1)
    flat2 = squareform(matrix2)
    r_obs, _ = pearsonr(flat1, flat2)
    count = 0
    for _ in range(permutations):
        np.random.shuffle(flat1)
        r_perm, _ = pearsonr(flat1, flat2)
        if abs(r_perm) >= abs(r_obs):
            count += 1
    p_value = (count + 1) / (permutations + 1)  # 避免p=0
    return r_obs, p_value


# 2. 修复后的Mantel检验函数
def mantel_test(env_df, spec_df, group_info):
    env_dist = squareform(pdist(env_df, metric='euclidean'))

    results = []
    for group_val in np.unique(group_info):
        # 获取当前分组的样本索引
        sample_indices = group_info[group_info == group_val].index
        group_spec = spec_df.loc[sample_indices]

        # 跳过样本不足的分组
        if len(group_spec) < 3:
            print(f"跳过分组 {group_val} (样本数不足)")
            continue

        spec_dist = squareform(pdist(group_spec, metric='braycurtis'))
        r_value, p_value = mantel(env_dist, spec_dist)
        results.append({
            'spec': f'Cluster{group_val}',
            'env': 'All',
            'r': r_value,
            'p': p_value
        })

    return pd.DataFrame(results)


# 3. 分类Mantel检验结果（保持不变）
def classify_mantel_results(mantel_res):
    mantel_res['r_class'] = pd.cut(mantel_res['r'],
                                   bins=[-np.inf, 0.2, 0.4, np.inf],
                                   labels=['< 0.2', '0.2 - 0.4', '>= 0.4'])
    mantel_res['p_class'] = pd.cut(mantel_res['p'],
                                   bins=[-np.inf, 0.01, 0.05, np.inf],
                                   labels=['< 0.01', '0.01 - 0.05', '>= 0.05'])
    return mantel_res


# 4. 创建组合热图
def create_combined_corrplot(env_df, mantel_res):
    # 计算环境因子之间的相关系数
    corr_matrix = env_df.corr()

    # 设置图形
    fig, ax = plt.subplots(figsize=(12, 10))

    # 创建自定义颜色映射（RdYlBu）
    cmap = LinearSegmentedColormap.from_list('rdylbu', [
        '#313695', '#4575b4', '#74add1', '#abd9e9',
        '#e0f3f8', '#ffffbf', '#fee090', '#fdae61',
        '#f46d43', '#d73027', '#a50026'
    ], N=256)

    # 绘制上三角热图（隐藏下三角）
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.tril_indices_from(mask)] = True
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                annot=True, fmt='.2f', square=True, cbar=True,
                cbar_kws={'shrink': 0.5, 'label': "Pearson's r"},
                linewidths=0.5, linecolor='white', ax=ax)

    # 添加网络连线
    add_network_connections(ax, corr_matrix, mantel_res)

    # 添加图例
    add_custom_legend(ax)

    plt.tight_layout()
    return fig, ax


# 5. 添加网络连线
def add_network_connections(ax, corr_matrix, mantel_res):
    # 设置颜色映射
    p_colors = {
        '< 0.01': '#eb7e60',
        '0.01 - 0.05': '#6ca3d4',
        '>= 0.05': 'grey'
    }

    # 设置线宽映射
    size_map = {
        '< 0.2': 1.0,
        '0.2 - 0.4': 2.0,
        '>= 0.4': 3.0
    }

    # 获取环境因子位置
    env_positions = {}
    n = len(corr_matrix.columns)
    for i, col in enumerate(corr_matrix.columns):
        env_positions[col] = (i + 0.5, n - i - 0.5)

    # 添加物种节点位置
    spec_positions = {}
    for i, row in enumerate(mantel_res.itertuples()):
        spec_positions[row.spec] = (n + 1 + i, n - i - 0.5)

    # 绘制连线
    for row in mantel_res.itertuples():
        color = p_colors[row.p_class]
        linewidth = size_map[row.r_class]

        # 创建曲线连接
        for env in corr_matrix.columns:
            con = ConnectionPatch(
                xyA=env_positions[env],
                xyB=spec_positions[row.spec],
                coordsA='data',
                coordsB='data',
                axesA=ax,
                axesB=ax,
                arrowstyle='-',
                color=color,
                linewidth=linewidth,
                alpha=0.8,
                connectionstyle=f"arc3,rad={0.2 * (1 + row.r)}"
            )
            ax.add_artist(con)

    # 添加物种节点
    for spec, pos in spec_positions.items():
        ax.scatter(*pos, s=200, c='#FFD700', edgecolor='black', zorder=10)
        ax.text(pos[0] + 0.3, pos[1], spec, fontsize=10, ha='left', va='center')


# 6. 添加自定义图例
def add_custom_legend(ax):
    # 创建Mantel r图例
    r_legend = [
        plt.Line2D([0], [0], color='black', lw=1, label='< 0.2'),
        plt.Line2D([0], [0], color='black', lw=2, label='0.2 - 0.4'),
        plt.Line2D([0], [0], color='black', lw=3, label='>= 0.4')
    ]

    # 创建Mantel p图例
    p_legend = [
        plt.Line2D([0], [0], color='#eb7e60', lw=2, label='< 0.01'),
        plt.Line2D([0], [0], color='#6ca3d4', lw=2, label='0.01 - 0.05'),
        plt.Line2D([0], [0], color='grey', lw=2, label='>= 0.05')
    ]

    # 添加图例
    leg1 = ax.legend(handles=r_legend, title="Mantel r",
                     loc='upper left', bbox_to_anchor=(1.01, 1))
    leg2 = ax.legend(handles=p_legend, title="Mantel p",
                     loc='upper left', bbox_to_anchor=(1.01, 0.85))

    # 手动添加图例到图形
    ax.add_artist(leg1)
    ax.add_artist(leg2)


# 7. 主函数（增加错误处理）
def main(env_path, spec_path, output_path='correlation_network_heatmap.png'):
    try:
        env_df, spec_df, group_info = prepare_data(env_path, spec_path)
        print("环境因子数据样本:", env_df.index.tolist())
        print("物种丰度数据样本:", spec_df.index.tolist())

        mantel_res = mantel_test(env_df, spec_df, group_info)
        mantel_res = classify_mantel_results(mantel_res)

        fig, ax = create_combined_corrplot(env_df, mantel_res)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    except Exception as e:
        print(f"错误发生: {str(e)}")
        import traceback
        traceback.print_exc()


# 8. 修复后的模拟数据生成
def generate_sample_data():
    # 环境因子数据（样本为列）
    env_data = pd.DataFrame(
        np.random.rand(5, 10) * 10,
        index=['pH', 'Temperature', 'Humidity', 'Salinity', 'Nutrient'],
        columns=[f'Sample{i}' for i in range(1, 11)]
    )
    env_data.to_csv("env_testdata.csv")

    # 物种丰度数据（样本为行）
    spec_data = pd.DataFrame(
        np.random.randint(0, 100, (10, 20)),
        index=[f'Sample{i}' for i in range(1, 10)],
        columns=[f'Sample{i}' for i in range(1, 10)]
    )
    spec_data['Group'] = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    spec_data.reset_index(inplace=True)  # 将索引转为列
    spec_data.rename(columns={'index': 'SampleID'}, inplace=True)
    spec_data.to_csv("spec_testdata.csv", index=False)


if __name__ == '__main__':
    generate_sample_data()
    main("env_testdata.csv", "spec_testdata.csv")