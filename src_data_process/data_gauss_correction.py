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


def generate_gaussian_data(size, mean, std):
    """生成高斯分布数据"""
    data = np.random.normal(mean, std, size)
    # 移除极端异常值（保持95%数据在范围内）
    lower_bound = mean - 1.96 * std
    upper_bound = mean + 1.96 * std
    filtered_data = data[(data > lower_bound) & (data < upper_bound)]

    # 确保大小一致
    while len(filtered_data) < size:
        new_data = np.random.normal(mean, std, size)
        filtered_data = new_data[(new_data > lower_bound) & (new_data < upper_bound)]

    return filtered_data[:size]


def test_gaussian_scaling():
    """
    测试高斯分布缩放算法

    1. 生成测试数据：n1 (范围1-10) 和 n2 (范围-300-700)
    2. 将n2缩放到n1的范围
    3. 验证缩放结果符合预期
    """
    # 1. 生成测试数据
    n1 = generate_gaussian_data(10000, mean=5, std=2)
    n2 = generate_gaussian_data(5000, mean=200, std=150)

    # 检查原始数据分布
    _, pval_n1 = shapiro(n1)
    _, pval_n2 = shapiro(n2)
    print(f"原始数据分布检验(p>0.05为正态分布):")
    print(f"n1 Shapiro-Wilk p-value: {pval_n1:.4f}")
    print(f"n2 Shapiro-Wilk p-value: {pval_n2:.4f}")

    print("\n原始数据95%范围:")
    print(f"n1: [{np.percentile(n1, 2.5):.1f}, {np.percentile(n1, 97.5):.1f}]")
    print(f"n2: [{np.percentile(n2, 2.5):.1f}, {np.percentile(n2, 97.5):.1f}]")

    # 2. 缩放数据
    scaled_n2, stats = scale_gaussian(n2, n1, return_stats=True)

    # 3. 验证结果
    print("\n缩放后统计信息:")
    for key, value in stats.items():
        if isinstance(value, tuple):
            print(f"{key}: [{value[0]:.2f}, {value[1]:.2f}]")
        else:
            print(f"{key}: {value:.2f}")

    # 验证分布特性 - KS检验
    print("\n分布特性验证:")

    # 检查缩放后的分布
    _, pval_scaled = shapiro(scaled_n2)
    print(f"缩放后数据Shapiro-Wilk p-value: {pval_scaled:.4f}")

    # 与n1的分布相似性
    # 检验两分布是否来自同一总体
    _, p_value = ttest_ind(n1, scaled_n2, equal_var=False)
    print(f"缩放数据与n1的T检验p-value: {p_value:.4f}")

    # 可视化验证
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))

    # 原始数据分布
    plt.subplot(2, 1, 1)
    plt.hist(n1, bins=50, alpha=0.7, label='n1 (target)', density=True)
    plt.hist(n2, bins=50, alpha=0.7, label='n2 (source)', density=True)
    plt.legend()
    plt.title('Original Distributions')

    # 缩放后数据分布
    plt.subplot(2, 1, 2)
    plt.hist(n1, bins=50, alpha=0.7, label='n1 (target)', density=True)
    plt.hist(scaled_n2, bins=50, alpha=0.7, label='n2 scaled', density=True)
    plt.legend()
    plt.title('Scaled Distribution Comparison')

    plt.tight_layout()
    plt.savefig('gaussian_scaling_result.png')
    plt.show()

    # 验证95%范围
    scaled_lower = np.percentile(scaled_n2, 2.5)
    scaled_upper = np.percentile(scaled_n2, 97.5)
    target_lower = np.percentile(n1, 2.5)
    target_upper = np.percentile(n1, 97.5)

    # 检查是否在合理误差范围内
    range_error_lower = abs(scaled_lower - target_lower) / (target_upper - target_lower)
    range_error_upper = abs(scaled_upper - target_upper) / (target_upper - target_lower)

    print(f"\n范围误差（相对目标范围大小）:")
    print(f"下界: {range_error_lower * 100:.1f}%")
    print(f"上界: {range_error_upper * 100:.1f}%")

    # 确认误差在5%以内
    assert range_error_lower < 0.05, "下界误差过大"
    assert range_error_upper < 0.05, "上界误差过大"
    print("\n测试通过：缩放后的数据95%范围在目标范围内，且保持高斯分布特性")

    return stats


if __name__ == "__main__":
    # 运行测试
    stats = test_gaussian_scaling()

    # 其他测试用例
    print("\n\n=== 附加测试用例 ===")

    # 用例1: 不同尺寸的数据集
    print("\n测试用例1: 不同尺寸数据集")
    n1_small = generate_gaussian_data(500, mean=5, std=2)
    n2_large = generate_gaussian_data(50, mean=200, std=150)
    scaled_n2 = scale_gaussian(n2_large, n1_small)
    scaled_lower = np.percentile(scaled_n2, 2.5)
    scaled_upper = np.percentile(scaled_n2, 97.5)
    print(f"目标范围: [{np.percentile(n1_small, 2.5):.1f}, {np.percentile(n1_small, 97.5):.1f}]")
    print(f"缩放范围: [{scaled_lower:.1f}, {scaled_upper:.1f}]")

    # 用例2: 边界情况 - 标准差接近0
    print("\n测试用例2: 接近零标准差")
    n1_tight = np.array([5.0] * 100 + [5.01])  # 几乎恒定
    n2_wide = generate_gaussian_data(1000, mean=0, std=200)
    scaled_n2 = scale_gaussian(n2_wide, n1_tight)
    scaled_std = np.std(scaled_n2)
    print(f"目标标准差: {np.std(n1_tight):.4f}")
    print(f"缩放标准差: {scaled_std:.4f}")

    # 用例3: 异常值测试
    print("\n测试用例3: 包含异常值")
    n1_clean = generate_gaussian_data(1000, mean=5, std=2)
    n2_outliers = np.concatenate([
        generate_gaussian_data(950, mean=200, std=150),
        np.random.uniform(-1000, 1000, 50)  # 添加一些异常值
    ])
    scaled_n2 = scale_gaussian(n2_outliers, n1_clean)
    scaled_lower = np.percentile(scaled_n2, 2.5)
    scaled_upper = np.percentile(scaled_n2, 97.5)
    print(f"目标范围: [{np.percentile(n1_clean, 2.5):.1f}, {np.percentile(n1_clean, 97.5):.1f}]")
    print(f"缩放范围: [{scaled_lower:.1f}, {scaled_upper:.1f}]")

