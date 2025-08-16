import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 图像变化，根据图像的倾斜角度进行图像的拉伸
def shear_image(image, slope, padding_color=(0, 0, 0)):
    """
    根据公式 y = slope * x 对图像进行拉伸变换

    参数:
    image: 输入图像 (numpy数组，形状为 [height, width, channels])
    slope: 拉伸斜率 (正值向上拉伸，负值向下拉伸)
    padding_color: 边界填充颜色 (默认黑色)

    返回:
    transformed_image: 变换后的图像 (与输入同尺寸)
    """
    # 获取图像尺寸
    height, width, channels = image.shape

    # 创建输出图像，初始化为填充颜色
    transformed_image = np.zeros_like(image)
    transformed_image[:, :] = padding_color

    # 逐列处理图像
    for x in range(width):
        # 计算当前列的位移量 (y = slope * x)
        displacement = slope * x

        # 对位移取整 (亚像素变换需要插值)
        offset = int(round(displacement))

        # 计算源图像和目标图像的行范围
        src_start = max(0, -offset)
        src_end = min(height, height - offset)
        dst_start = max(0, offset)
        dst_end = min(height, height + offset)

        # 仅在有重叠区域时复制像素
        if src_start < src_end:
            # 从源图像复制像素到目标位置
            transformed_image[dst_start:dst_end, x, :] = image[src_start:src_end, x, :]

            # 处理顶部边界
            if offset > 0:
                transformed_image[0:offset, x, :] = padding_color

            # 处理底部边界
            if offset < 0:
                transformed_image[height + offset:, x, :] = padding_color

    return transformed_image


# 测试函数
def test_shear_transformation():
    """
    测试图像剪切变换功能

    修复了子图索引错误：原来的2x2网格只能容纳4个子图，
    但代码尝试创建5个子图（原始图像+4个变换图像），导致索引越界。

    解决方案：将布局改为3行2列网格（6个子图位置），
    但只使用前5个位置（原始图像+4个变换图像）
    """
    # 创建测试图像 (2000×256×3)
    height, width = 2000, 256
    channels = 3

    # 创建有视觉特征的测试图像
    image = np.zeros((height, width, channels), dtype=np.uint8)

    # 添加水平条纹
    for i in range(0, height, 50):
        image[i:i + 10, :, :] = [255, 0, 0]  # 红色条纹

    # 添加垂直线
    for j in range(0, width, 32):
        image[:, j:j + 2, :] = [0, 255, 0]  # 绿色垂直线

    # 添加文本信息
    cv2.putText(image, "Original Image", (50, 1000),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
    cv2.putText(image, f"{height} x {width}", (50, 1800),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)

    # 创建画布和子图 - 修复错误：使用3行2列网格（共6个子图位置）
    plt.figure(figsize=(18, 16), dpi=80)

    # 子图1：原始图像（位置1）
    plt.subplot(1, 5, 1)
    plt.imshow(image)
    plt.title(f"原始图像 ({height}×{width})")
    plt.axis('off')

    # 测试不同斜率下的变换效果
    slopes = [0.2, -0.2, 0.8, -1.0]  # 4个斜率值

    for i, slope in enumerate(slopes):
        # 应用变换
        transformed_image = shear_image(image, slope)

        # 计算图像位移示例
        displacement = slope * (width - 1)
        displacement_info = f"位移范围: {min(0, displacement)} - {max(0, displacement)} px"

        # 在子图中显示
        # 修复错误：使用正确的子图索引（从位置2开始）
        plt.subplot(1, 5, i+1)  # i=0 -> 位置2, i=1 -> 位置3, i=2 -> 位置4, i=3 -> 位置5
        plt.imshow(transformed_image)
        plt.title(f"斜率 slope={slope}\n{displacement_info}")
        plt.axis('off')

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 中文字体设置
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    plt.tight_layout()
    # plt.savefig('shear_transformation_examples.png', dpi=120)
    plt.show()

    return image


if __name__ == "__main__":
    # 生成并保存测试结果
    test_image = test_shear_transformation()

    # # 单独保存测试图像
    # cv2.imwrite('test_original.png', test_image)
    # print("测试结果已保存为 'shear_transformation_examples.png'")