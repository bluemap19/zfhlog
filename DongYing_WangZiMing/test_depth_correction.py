import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter
import matplotlib as mpl

# 设置Matplotlib支持中文显示
def setup_chinese_support():
    """配置Matplotlib支持中文显示"""
    # 检查操作系统
    if os.name == 'nt':  # Windows系统
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
    else:  # Linux/Mac系统
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Heiti TC', 'STHeiti', 'SimHei']

    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

    # 设置字体大小
    plt.rcParams['font.size'] = 12

    # 尝试设置具体字体
    try:
        # Windows系统使用微软雅黑
        if os.name == 'nt':
            mpl.font_manager.fontManager.addfont('C:\\Windows\\Fonts\\msyh.ttc')
            plt.rcParams['font.family'] = 'Microsoft YaHei'
        # Mac系统使用苹方字体
        elif os.name == 'posix':
            mpl.font_manager.fontManager.addfont('/System/Library/Fonts/PingFang.ttc')
            plt.rcParams['font.family'] = 'PingFang SC'
    except:
        print("警告：无法设置特定字体，使用默认中文字体")


# 在程序开始时调用
setup_chinese_support()

# 曲线的深度校正
def denoising_depth_correction(df, depth_col='depth', base_col='baselog', process_col='processlog',
                               search_range=10.0, coarse_res=1.0, fine_res=0.1,
                               denoise_window_point=1, denoise_polyorder=2, plot=False):
    """
    带降噪的曲线深度校正接口

    参数:
    df: 包含深度、基准曲线和待校正曲线的DataFrame
    depth_col: 深度列名 (默认'depth')
    base_col: 基准曲线列名 (默认'baselog')
    process_col: 待校正曲线列名 (默认'processlog')
    search_range: 搜索范围 (米, 默认±10米)
    coarse_res: 粗搜索分辨率 (米, 默认1.0)
    fine_res: 精搜索分辨率 (米, 默认0.1)
    denoise_window: 降噪窗口大小 (默认15)
    denoise_polyorder: 降噪多项式阶数 (默认2)
    plot: 是否绘制结果 (默认False)

    返回:
    shift: 最佳深度偏移量 (米)
    corrected_df: 校正后的DataFrame
    """
    # 1. 数据准备
    if isinstance(depth_col, str):
        depth = df[depth_col].values
    elif isinstance(depth_col, int):
        depth = df.iloc[:, depth_col].values
    else:
        raise TypeError('depth_col输入参数类型错误')
    if isinstance(base_col, str):
        base = df[base_col].values
    elif isinstance(base_col, int):
        base = df.iloc[:, base_col].values
    else:
        raise TypeError('base_col 输入参数类型错误')
    if isinstance(process_col, str):
        process = df[process_col].values
    elif isinstance(process_col, int):
        process = df.iloc[:, process_col].values
    else:
        raise TypeError('process_col 输入参数类型错误')

    # 2. 曲线降噪
    denoised_base = denoise_curve(depth, base, denoise_window_point, denoise_polyorder)
    denoised_process = denoise_curve(depth, process, denoise_window_point, denoise_polyorder)

    # 3. 使用降噪曲线计算偏移量
    shift = calculate_shift(depth, denoised_base, denoised_process, search_range, coarse_res, fine_res)

    # 4. 应用校正到原始数据
    corrected_depth = depth + shift
    corrected_df = pd.DataFrame({
        depth_col: depth,
        base_col: base,
        f'denoised_{base_col}': denoised_base,
        f'denoised_{process_col}': denoised_process,
        f'corrected_{depth_col}': corrected_depth,
        f'corrected_{process_col}': process
    })

    # 5. 结果可视化
    if plot:
        plot_denoising_results(
            depth, base, denoised_base,
            depth, process, denoised_process,
            corrected_depth, process,
            shift
        )

    return shift, corrected_df


def denoise_curve(depth, curve, window_points=1, polyorder=2):
    """
    使用Savitzky-Golay滤波器降噪曲线

    参数:
    depth: 深度数组
    curve: 曲线值数组
    window_size: 滤波窗口大小，窗长 单位为m ，后续会换算成相对应的窗口长度的
    polyorder: 多项式阶数

    返回:
    降噪后的曲线
    """
    # 确保窗口大小为奇数
    if window_points % 2 == 0:
        window_points += 1

    # 应用Savitzky-Golay滤波器
    denoised_curve = savgol_filter(curve, window_length=window_points, polyorder=polyorder)

    return denoised_curve


def calculate_shift(depth, base, process, search_range, coarse_res, fine_res):
    """
    计算最佳深度偏移量
    """
    # 1. 粗搜索 - 低分辨率快速定位
    coarse_shift = coarse_search(depth, base, process, search_range, coarse_res)

    # 2. 精搜索 - 高分辨率精确定位
    fine_shift = fine_search(depth, base, process, coarse_shift, search_range / 2, fine_res)

    # 3. 优化验证
    final_shift = optimize_shift(depth, base, process, fine_shift, search_range / 10)

    return final_shift


def coarse_search(depth, base, process, search_range, resolution):
    """
    粗搜索阶段 - 低分辨率快速定位大致偏移范围
    """
    best_shift = 0
    best_corr = -np.inf

    # 创建搜索网格
    shifts = np.arange(-search_range, search_range + resolution, resolution)

    for shift in shifts:
        # 应用偏移
        shifted_depth = depth + shift

        # 插值到公共网格
        base_interp = interp1d(depth, base, kind='linear', bounds_error=False, fill_value='extrapolate')
        process_interp = interp1d(shifted_depth, process, kind='linear', bounds_error=False, fill_value='extrapolate')

        # 公共深度网格
        min_depth = max(depth.min(), shifted_depth.min())
        max_depth = min(depth.max(), shifted_depth.max())

        if min_depth >= max_depth:
            continue

        common_depth = np.arange(min_depth, max_depth, resolution)

        # 插值
        base_common = base_interp(common_depth)
        process_common = process_interp(common_depth)

        # 计算相关系数
        corr = np.corrcoef(base_common, process_common)[0, 1]

        # 更新最佳偏移
        if corr > best_corr:
            best_corr = corr
            best_shift = shift

    return best_shift


def fine_search(depth, base, process, center_shift, search_range, resolution):
    """
    精搜索阶段 - 高分辨率精确定位偏移量
    """
    best_shift = center_shift
    best_corr = -np.inf

    # 创建精细搜索网格
    shifts = np.arange(center_shift - search_range, center_shift + search_range + resolution, resolution)

    for shift in shifts:
        # 应用偏移
        shifted_depth = depth + shift

        # 插值到公共网格
        base_interp = interp1d(depth, base, kind='linear', bounds_error=False, fill_value='extrapolate')
        process_interp = interp1d(shifted_depth, process, kind='linear', bounds_error=False, fill_value='extrapolate')

        # 公共深度网格
        min_depth = max(depth.min(), shifted_depth.min())
        max_depth = min(depth.max(), shifted_depth.max())

        if min_depth >= max_depth:
            continue

        common_depth = np.arange(min_depth, max_depth, resolution)

        # 插值
        base_common = base_interp(common_depth)
        process_common = process_interp(common_depth)

        # 计算相关系数
        corr = np.corrcoef(base_common, process_common)[0, 1]

        # 更新最佳偏移
        if corr > best_corr:
            best_corr = corr
            best_shift = shift

    return best_shift


def optimize_shift(depth, base, process, initial_shift, search_range):
    """
    优化阶段 - 使用数值优化找到最佳偏移
    """

    # 定义目标函数 (最小化负相关系数)
    def objective(shift):
        shifted_depth = depth + shift

        # 公共深度网格
        min_depth = max(depth.min(), shifted_depth.min())
        max_depth = min(depth.max(), shifted_depth.max())

        if min_depth >= max_depth:
            return 10.0  # 返回大值表示差

        # 创建密集网格
        common_depth = np.linspace(min_depth, max_depth, 1000)

        # 插值
        base_interp = interp1d(depth, base, kind='linear', bounds_error=False, fill_value='extrapolate')
        process_interp = interp1d(shifted_depth, process, kind='linear', bounds_error=False, fill_value='extrapolate')

        base_common = base_interp(common_depth)
        process_common = process_interp(common_depth)

        # 使用相关系数的负值作为目标函数
        return -np.corrcoef(base_common, process_common)[0, 1]

    # 使用优化算法找到最小值
    result = minimize_scalar(
        objective,
        bounds=(initial_shift - search_range, initial_shift + search_range),
        method='bounded'
    )

    return result.x


def plot_denoising_results(base_depth, base_curve, denoised_base,
                           orig_process_depth, orig_process_curve, denoised_process,
                           corrected_depth, corrected_curve,
                           shift):
    """
    绘制带降噪的深度校正结果
    """
    plt.figure(figsize=(16, 12))

    # 原始曲线对比
    plt.subplot(221)
    plt.plot(base_depth, base_curve, 'b-', label='原始基准曲线')
    plt.plot(orig_process_depth, orig_process_curve, 'r-', alpha=0.7, label='原始待校正曲线')
    plt.title('原始曲线对比')
    plt.xlabel('深度 (m)')
    plt.ylabel('测井响应')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 降噪曲线对比
    plt.subplot(222)
    plt.plot(base_depth, denoised_base, 'b-', label='降噪基准曲线')
    plt.plot(orig_process_depth, denoised_process, 'r-', alpha=0.7, label='降噪待校正曲线')
    plt.title('降噪曲线对比')
    plt.xlabel('深度 (m)')
    plt.ylabel('测井响应')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 校正后曲线对比
    plt.subplot(223)
    plt.plot(base_depth, denoised_base, 'b-', label='降噪基准曲线')
    plt.plot(corrected_depth, denoised_process, 'g-', alpha=0.8, label=f'校正后曲线 (偏移: {shift:.4f}m)')
    plt.title(f'降噪曲线校正后对比 (偏移量: {shift:.4f}m)')
    plt.xlabel('深度 (m)')
    plt.ylabel('测井响应')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 原始曲线校正后对比
    plt.subplot(224)
    plt.plot(base_depth, base_curve, 'b-', label='原始基准曲线')
    plt.plot(corrected_depth, corrected_curve, 'g-', alpha=0.8, label=f'校正后原始曲线 (偏移: {shift:.4f}m)')
    plt.title(f'原始曲线校正后对比 (偏移量: {shift:.4f}m)')
    plt.xlabel('深度 (m)')
    plt.ylabel('测井响应')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.show()


# 测试函数
def test_denoising_depth_correction():
    """带降噪的深度校正测试"""
    # 创建测试数据
    depth = np.arange(1000, 2000, 0.1)
    base_curve = np.sin(depth * 0.01) + np.random.normal(0, 0.2, len(depth))

    # 创建偏移的processlog
    true_shift = 17.0  # 真实偏移量
    process_curve = np.sin((depth + true_shift) * 0.01) + np.random.normal(0, 0.2, len(depth))

    # 创建DataFrame
    df = pd.DataFrame({
        'depth': depth,
        'baselog': base_curve,
        'processlog': process_curve
    })

    # 应用带降噪的深度校正
    shift, corrected_df = denoising_depth_correction(
        df,
        depth_col='depth',
        base_col='baselog',
        process_col='processlog',
        search_range=10.0,
        coarse_res=1.0,
        fine_res=0.1,
        denoise_window=5,
        denoise_polyorder=2,
        plot=True
    )

    # 验证结果
    print(f"真实偏移量: {true_shift:.4f}m")
    print(f"检测偏移量: {shift:.4f}m")
    print(f"绝对误差: {abs(shift - true_shift):.4f}m")

    # 计算校正后深度误差
    corrected_depth = corrected_df['corrected_depth'].values
    depth_error = np.mean(np.abs(corrected_depth - (depth + true_shift)))
    print(f"平均深度误差: {depth_error:.4f}m")

    # # 保存结果
    # corrected_df.to_csv('denoising_corrected_log.csv', index=False)
    # print("校正后数据已保存至 denoising_corrected_log.csv")

    return shift, corrected_df


if __name__ == "__main__":
    # 运行测试
    shift, corrected_df = test_denoising_depth_correction()