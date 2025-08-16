import os

import numpy as np
import cv2
import matplotlib.pyplot as plt


# 设置中文支持
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
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


# 调用中文支持设置
setup_chinese_support()

# 图像转测井曲线,进行了相应的优化，会根据图像最大最小值分布，直接调整测井曲线响应效果
def image_to_trend_curve(image, plot=False):
    """
    将二维图像转换为一维趋势曲线

    参数:
    image: 输入图像 (numpy数组, 形状为 [height, width])
    curve_length: 输出曲线长度 (默认2845)
    smoothing_window: 平滑窗口大小 (默认15)
    plot: 是否绘制结果 (默认False)

    返回:
    trend_curve: 趋势曲线 (numpy数组, 长度为curve_length)
    """
    # 1. 图像预处理
    # 确保图像是单通道灰度图
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 获取图像尺寸
    height, width = image.shape

    # 2. 计算每列的平均值
    trend_curve_mean = np.mean(image, axis=1)
    trend_curve_max = np.max(image, axis=1)
    trend_curve_min = np.min(image, axis=1)
    trend_curve = np.zeros_like(trend_curve_mean)
    for i in range(height):
        if trend_curve_max[i] > 200 and trend_curve_mean[i] >= 68:
            trend_curve[i] = trend_curve_max[i]
        elif trend_curve_min[i] < 32 and trend_curve_mean[i] <= 48:
            trend_curve[i] = trend_curve_min[i]
        else:
            trend_curve[i] = trend_curve_mean[i]

    # 5. 结果可视化
    if plot:
        plt.figure(figsize=(6, 12))

        # 原始图像
        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title("原始图像 ({}×{})".format(height, width))
        plt.axis('off')

        # 趋势曲线
        plt.subplot(122)
        plt.plot(trend_curve_mean, np.linspace(1, image.shape[0], image.shape[0]), 'b-', linewidth=1.5)
        # plt.plot(trend_curve_max, np.linspace(1, image.shape[0], image.shape[0]), 'd-.', linewidth=1.5)
        # plt.plot(trend_curve_min, np.linspace(1, image.shape[0], image.shape[0]), 'y--', linewidth=1.5)
        plt.plot(trend_curve, np.linspace(1, image.shape[0], image.shape[0]), 'g-', linewidth=1.5)
        plt.xlabel("强度")
        plt.ylabel("位置")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    return trend_curve




# 测试函数
def test_image_to_curve():
    """测试图像到曲线转换功能"""
    # 创建测试图像 (64×2845)
    height, width = 2845, 64

    # 创建有趋势变化的图像
    image = np.zeros((height, width), dtype=np.uint8)

    # 添加趋势变化
    for x in range(height):
        # 基础趋势：正弦波
        base_value = 128 + 100 * np.sin(x * 0.01)

        # 添加噪声
        noise = np.random.normal(0, 20, width)

        # 设置列值
        image[x, :] = np.clip(base_value + noise, 0, 255)

    # 应用转换
    trend_curve = image_to_trend_curve(image, plot=True)

    # 分析结果
    print("曲线长度:", len(trend_curve))
    print("曲线范围: {:.2f} - {:.2f}".format(np.min(trend_curve), np.max(trend_curve)))

    return trend_curve


if __name__ == "__main__":
    # 运行测试
    curve = test_image_to_curve()

    # # 保存曲线数据
    # np.savetxt('image_trend_curve.csv', curve, delimiter=',', fmt='%.4f')
    # print("趋势曲线已保存为 'image_trend_curve.csv'")