import numpy as np
from scipy.stats import shapiro, ttest_ind


def scale_gaussian(source_data, target_data, return_stats=False):
    """
    将源数据缩放到目标数据的分布范围，同时保持高斯分布特性

    参数:
    source_data (np.array): 要缩放的数据 (n2)
    target_data (np.array): 目标数据范围 (n1)
    return_stats (bool): 是否返回统计信息

    返回:
    scaled_data (np.array): 缩放后的数据
    stats (dict): 缩放前后的统计信息（仅当return_stats=True时返回）
    """
    # 计算目标数据的统计参数
    μ_target = np.median(target_data)  # 使用中位数更稳定
    σ_target = np.std(target_data)

    # 计算源数据的统计参数
    μ_source = np.median(source_data)
    σ_source = np.std(source_data)

    # 防止标准差为零
    if σ_source < 1e-10:
        σ_source = 1e-10

    # Z-score标准化
    z_scores = (source_data - μ_source) / σ_source

    # 缩放到目标分布
    scaled_data = z_scores * σ_target + μ_target

    if return_stats:
        # 计算缩放前后统计信息
        def get_range(data):
            """计算95%数据范围"""
            lower = np.percentile(data, 2.5)
            upper = np.percentile(data, 97.5)
            return (lower, upper)

        stats = {
            "source_mean": μ_source,
            "source_std": σ_source,
            "source_range": get_range(source_data),
            "target_mean": μ_target,
            "target_std": σ_target,
            "target_range": get_range(target_data),
            "scaled_mean": np.median(scaled_data),
            "scaled_std": np.std(scaled_data),
            "scaled_range": get_range(scaled_data),
        }
        return scaled_data, stats
    else:
        return scaled_data

import numpy as np


def scale_gaussian_by_quantiles(source_data, target_data, quantile=0.2):
    """
    将源数据缩放到目标数据的分布范围，使源数据的20%分位数与目标数据的20%分位数对齐

    参数:
    source_data (np.array): 要缩放的数据 (n2)
    target_data (np.array): 目标数据范围 (n1)
    quantile (float): 使用的分位数 (默认为0.2，即20%)
    return_stats (bool): 是否返回统计信息

    返回:
    scaled_data (np.array): 缩放后的数据
    stats (dict): 缩放前后的统计信息（仅当return_stats=True时返回）
    """

    μ_target = np.median(target_data)  # 使用中位数更稳定
    μ_source = np.median(source_data)
    source_data = source_data + (μ_target - μ_source)

    # 计算源数据和目标数据的20%分位数
    source_lower = np.percentile(source_data, quantile * 100)
    source_upper = np.percentile(source_data, (1 - 2*quantile) * 100)

    target_lower = np.percentile(target_data, quantile * 100)
    target_upper = np.percentile(target_data, (1 - 2*quantile) * 100)

    # 计算源数据和目标数据的范围
    source_range = source_upper - source_lower
    target_range = target_upper - target_lower

    # 防止范围为零
    if source_range < 1e-10:
        source_range = 1e-10
    if target_range < 1e-10:
        target_range = 1e-10

    # 计算缩放因子和偏移量
    scale_factor = target_range / source_range
    offset = target_lower - source_lower * scale_factor

    # 应用线性变换
    scaled_data = source_data * scale_factor + offset

    return scaled_data


# def scale_gaussian(source_data, target_data, trim_percent=0.01):
#     """
#     将源数据缩放到目标数据的分布范围，同时保持高斯分布特性
#     包含对输入数据的5%极值剔除处理
#
#     参数:
#     source_data (np.array): 要缩放的数据 (n2)
#     target_data (np.array): 目标数据范围 (n1)
#     return_stats (bool): 是否返回统计信息
#     trim_percent (float): 要剔除的极值比例 (默认5%)
#
#     返回:
#     scaled_data (np.array): 缩放后的数据 (保持原始顺序和长度)
#     stats (dict): 缩放前后的统计信息（仅当return_stats=True时返回）
#     """
#     # 保存原始数据的完整副本和索引
#     source_full = source_data.copy()
#     target_full = target_data.copy()
#
#     # 计算分位数位置
#     low_percentile = trim_percent * 100
#     high_percentile = 100 - trim_percent * 100
#
#     # 处理源数据：剔除最小5%和最大5%
#     source_lower = np.percentile(source_data, low_percentile)
#     source_upper = np.percentile(source_data, high_percentile)
#     source_filtered = source_data[(source_data >= source_lower) & (source_data <= source_upper)]
#
#     # 处理目标数据：剔除最小5%和最大5%
#     target_lower = np.percentile(target_data, low_percentile)
#     target_upper = np.percentile(target_data, high_percentile)
#     target_filtered = target_data[(target_data >= target_lower) & (target_data <= target_upper)]
#
#     # 检查筛选后数据是否足够
#     if len(source_filtered) < 2:
#         raise ValueError("筛选后的源数据太少，无法计算统计参数")
#
#     if len(target_filtered) < 2:
#         raise ValueError("筛选后的目标数据太少，无法计算统计参数")
#
#     # 计算筛选后源数据的统计参数
#     μ_source = np.median(source_filtered)
#     σ_source = np.std(source_filtered)
#
#     # 计算筛选后目标数据的统计参数
#     μ_target = np.median(target_filtered)
#     σ_target = np.std(target_filtered)
#
#     # 防止标准差为零
#     if σ_source < 1e-10:
#         σ_source = 1e-10
#     if σ_target < 1e-10:
#         σ_target = 1e-10
#
#     # 对整个源数据进行Z-score标准化
#     z_scores = (source_full - μ_source) / σ_source
#
#     # 缩放到目标分布
#     scaled_data = z_scores * σ_target + μ_target
#
#
#     return scaled_data



def scale_gaussian_by_config(source_data, target_data_config, return_stats=False):
    """
    将源数据缩放到目标数据的分布范围，同时保持高斯分布特性

    参数:
    source_data (np.array): 要缩放的数据 (n2)
    target_data (dict{μ_target: float, σ_target: float}): 目标数据范围 (n1)
    return_stats (bool): 是否返回统计信息

    返回:
    scaled_data (np.array): 缩放后的数据
    stats (dict): 缩放前后的统计信息（仅当return_stats=True时返回）
    """
    # 计算目标数据的统计参数
    μ_target = target_data_config['μ_target']
    σ_target = target_data_config['σ_target']

    # 计算源数据的统计参数
    μ_source = np.median(source_data)
    σ_source = np.std(source_data)

    # 防止标准差为零
    if σ_source < 1e-10:
        σ_source = 1e-10

    # Z-score标准化
    z_scores = (source_data - μ_source) / σ_source

    # 缩放到目标分布
    scaled_data = z_scores * σ_target + μ_target

    if return_stats:
        # 计算缩放前后统计信息
        def get_range(data):
            """计算95%数据范围"""
            lower = np.percentile(data, 2.5)
            upper = np.percentile(data, 97.5)
            return (lower, upper)

        stats = {
            "source_mean": μ_source,
            "source_std": σ_source,
            "target_mean": μ_target,
            "target_std": σ_target,
            "source_range": get_range(source_data),
            "scaled_mean": np.median(scaled_data),
            "scaled_std": np.std(scaled_data),
            "scaled_range": get_range(scaled_data),
        }
        return scaled_data, stats
    else:
        return scaled_data


def test_trimmed_gaussian_scaling():
    # 生成测试数据
    np.random.seed(42)

    # 源数据：高斯分布，有极端值
    source_clean = np.random.normal(10, 2, 1000)
    source_outliers = np.concatenate([source_clean, [-50], [50]])
    source_data = source_outliers.copy()

    # 目标数据：高斯分布，有极端值
    target_clean = np.random.normal(20, 3, 1000)
    target_outliers = np.concatenate([target_clean, [-100], [100]])
    target_data = target_outliers.copy()

    print("== 原始数据统计 ==")
    print(f"源数据最小值: {np.min(source_data):.2f}, 最大值: {np.max(source_data):.2f}")
    print(f"目标数据最小值: {np.min(target_data):.2f}, 最大值: {np.max(target_data):.2f}")
    print(f"源数据大小: {len(source_data)}, 目标数据大小: {len(target_data)}")

    # 执行带筛选的缩放
    scaled_data = scale_gaussian(
        source_data,
        target_data,
        return_stats=False
    )

    print(scaled_data)



if __name__ == "__main__":
    # 运行测试
    test_stats = test_trimmed_gaussian_scaling()

    # 保存统计信息
    import json

    with open('scaling_stats.json', 'w') as f:
        json.dump(test_stats, f, indent=4)

    print("测试完成! 结果已保存到 trimmed_gaussian_scaling_results.png 和 scaling_stats.json")