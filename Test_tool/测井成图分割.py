import os
import cv2
# 使用Matplotlib作为备用显示方案
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

# path = r'C:\Users\ZFH\Desktop\logging.png'
path = r'C:\Users\ZFH\Desktop\logging-1.png'

# 验证文件存在性
if not os.path.exists(path):
    print(f"错误：文件 {path} 不存在")
    exit()

# 验证文件可读性
if not os.access(path, os.R_OK):
    print(f"错误：无读取权限 {path}")
    exit()

# 带异常捕获的读取方式
try:
    original_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if original_image is None:
        raise ValueError("OpenCV无法解码该文件")
except Exception as e:
    print(f"图像读取失败: {str(e)}")
    exit()


if original_image is not None:
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    # plt.title("测井图像 (Matplotlib显示)")
    plt.show()
else:
    print("无有效图像数据")

import cv2
import numpy as np


def detect_black_lines(image, threshold=0.7, channels=3):
    """专业级黑色区域检测函数
    参数：
        image: OpenCV读取的BGR(A)图像矩阵
        threshold: 黑色像素占比阈值（默认80%）
        channels: 用于检测的通道数（3=忽略Alpha通道）

    返回：
        (black_rows, black_cols) 满足条件的行列索引列表
    """
    if image is None or image.size == 0:
        raise ValueError("输入图像无效")

    # 转换图像为三维数组（兼容任意通道数）
    h, w = image.shape[:2]
    if len(image.shape) == 2:
        img = image[..., np.newaxis]  # 灰度图增加通道维度
    else:
        img = image.copy()

    # 创建黑白掩模（三通道全黑视为黑色像素）
    black_mask = np.all(img[:, :, :channels] == 0, axis=2)

    # 行检测（向量化计算）
    row_black_ratio = np.sum(black_mask, axis=1) / w
    black_rows = np.where(row_black_ratio >= threshold)[0].tolist()

    # 列检测（向量化计算）
    col_black_ratio = np.sum(black_mask, axis=0) / h
    black_cols = np.where(col_black_ratio >= threshold)[0].tolist()

    return black_rows, black_cols



if original_image is not None:
    try:
        rows, cols = detect_black_lines(original_image)

        print("黑色区域检测报告：")
        print(f"满足条件的行索引：{rows}")
        print(f"满足条件的列索引：{cols}")


        # # 可视化标记（可选）
        # debug_image = original_image.copy()
        # for r in rows:
        #     cv2.line(debug_image, (0, r), (w, r), (0, 255, 0), 1)
        # for c in cols:
        #     cv2.line(debug_image, (c, 0), (c, h), (0, 0, 255), 1)
        #
        # cv2.imshow('Detection Result', debug_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except Exception as e:
        print(f"分析错误：{str(e)}")
else:
    print("图像加载失败，请检查路径")

row_index = [42, 715]
col_index = [73, 152, 168, 247, 263, 342, 358, 437, 453, 532, 548, 627, 643, 722, 738, 817, 833, 912, 928, 1007, 1023, 1102, 1118, 1198, 1213, 1293]
pic_names = ['AC', 'CNL', 'DEN', 'GR', 'SP', 'CAL', 'Type', 'NLP', 'KNN', 'SVM', 'Native Bayes', 'Random Forest', 'GBM']
for i in range(13):
    col_s = col_index[i*2] + 1
    col_e = col_index[i*2+1]
    row_s = row_index[0] + 1
    row_e = row_index[1]
    # print(original_image.shape)
    pic_block = original_image[row_s:row_e, col_s:col_e]
    # print(pic_block.shape)
    # cv2.imshow('Detection Result', pic_block)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit(0)
    cv2.imwrite(f'{pic_names[i]}.png', pic_block)

# row_index = [56, 965]
# col_index = [58, 211, 363, 515, 668, 820, 972, 1125, 1277, 1429, 1582, 1734, 1886, 2039]
# for i in range(len(col_index)):
#     if i == 0:
#         continue
#
#     col_s = col_index[i-1]
#     col_e = col_index[i]
#     row_s = row_index[0]
#     row_e = row_index[1]
#
#     original_image[row_s+1:row_e, col_s+1: col_e, :] = 255
#
# cv2.imshow("original", original_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite("black_lines.png", original_image)