import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple


def generate_nmr_data(depth_range: Tuple[float, float], num_points: int = 20,
                      nmr_type: str = "T2") -> Dict[float, Dict[str, np.ndarray]]:
    """
    生成更真实的核磁共振(NMR)数据

    参数:
    - depth_range: 深度范围 (min, max)
    - num_points: 生成的NMR数据点数量
    - nmr_type: NMR类型 ("T2"或"T1")

    返回:
    - NMR字典: {深度: {'T2_values': array, 'amplitude': array, 'mean_T2': float}}
    """
    depth_min, depth_max = depth_range
    nmr_dict = {}

    # 生成T2时间范围 (对数尺度，单位: ms)
    t2_values = np.logspace(-1, 3, 64)  # 0.1ms 到 1000ms

    for i in range(num_points):
        # 在深度范围内均匀分布深度点
        depth = depth_min + (depth_max - depth_min) * i / max(1, num_points - 1)

        # 模拟不同地层的T2分布特征
        if depth < depth_min + (depth_max - depth_min) * 0.25:
            # 浅部: 泥岩特征 - 短T2为主
            mean_t2 = 5 + depth * 0.1  # 随深度轻微增加
            std_dev = 0.3
            amplitude_scale = 0.9
        elif depth < depth_min + (depth_max - depth_min) * 0.5:
            # 中部: 砂岩特征 - 中等T2
            mean_t2 = 15 + depth * 0.2
            std_dev = 0.6
            amplitude_scale = 1.5
        else:
            # 深部: 孔隙发育特征 - 长T2
            mean_t2 = 25 + depth * 0.3
            std_dev = 0.9
            amplitude_scale = 1.9

        # 生成T2分布 (对数正态分布)
        log_mean = np.log(mean_t2)
        amplitude = amplitude_scale * norm.pdf(np.log(t2_values), log_mean, std_dev)

        # 添加随机噪声使数据更真实
        noise = np.random.normal(0, 0.05 * amplitude.max(), size=amplitude.shape)
        amplitude = np.maximum(0, amplitude + noise)

        # 归一化振幅
        if amplitude.max() > 0:
            amplitude = amplitude / amplitude.max()

        nmr_dict[depth] = {
            'NMR_X': t2_values,
            'NMR_Y': amplitude,
            'mean_T2': mean_t2,
            'type': nmr_type
        }

    return nmr_dict