import cv2
import numpy as np
import os


def remove_black_text(image_path, output_path='processed_image.png'):
    """
    删除图片中的黑色文字区域并用白色填充

    参数:
    image_path (str): 输入图片路径
    output_path (str): 输出图片路径

    返回:
    bool: 处理成功返回True，否则False
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图片 {image_path}")
        return False

    # 转换为HSV颜色空间，便于识别黑色区域
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print(hsv.shape)

    # 定义黑色范围（HSV空间）
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([255, 40, 255])  # 低亮度值对应黑色
    # # # 根据用户取色器结果定义黑色范围
    # lower_black = np.array([70, 145, 0])  # H:75(150/2), S:153(60%), V:0
    # upper_black = np.array([90, 190, 50])  # H:85(170/2), S:179(70%), V:38(15%)

    # 创建黑色文字掩码
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # # 形态学操作增强文本区域识别
    # kernel = np.ones((3, 3), np.uint8)
    # enhanced_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)  # 闭合操作填充文本空隙
    # enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_OPEN, kernel)  # 开操作去除噪点

    # 应用掩码 - 将黑色文字区域替换为白色
    result = img.copy()
    result[black_mask == 255] = [255, 255, 255]  # BGR白色

    # 保存处理后的图片
    cv2.imwrite(output_path, result)
    print(f"成功处理图片保存至: {output_path}")

    return True


# # 使用示例
# if __name__ == "__main__":
#     input_image = "document_with_text.jpg"  # 替换为实际图片路径
#     remove_black_text(input_image)



def extract_seal(image_path, output_path='extracted_seal.png'):
    """
    从文档图片中提取红色印章，并保存为透明背景的PNG图片

    参数:
    image_path (str): 输入图片路径
    output_path (str): 输出PNG路径 (默认为'extracted_seal.png')

    返回:
    bool: 成功提取返回True，否则False
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图片 {image_path}")
        return False

    # 转换为HSV颜色空间，更好处理红色
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print(hsv.shape)
    print(hsv[110:130, 100:140, :])

    # 定义红色范围（包括深红和浅红）
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 60, 70])
    upper_red2 = np.array([185, 255, 255])

    # 创建红色区域掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 形态学操作增强红色区域
    kernel = np.ones((5, 5), np.uint8)
    enhanced_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_OPEN, kernel)

    # 寻找轮廓
    contours, _ = cv2.findContours(enhanced_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("错误: 未检测到印章区域")
        return False

    # 寻找最大轮廓（假设印章是最大的红色区域）
    largest_contour = max(contours, key=cv2.contourArea)

    # 创建印章掩码
    seal_mask = np.zeros_like(red_mask)
    cv2.drawContours(seal_mask, [largest_contour], 0, 255, -1)

    # 创建带有透明通道的RGBA图像
    bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgba = cv2.cvtColor(bgr, cv2.COLOR_RGB2RGBA)

    # 应用掩码设置非印章区域为透明
    rgba[:, :, 3] = seal_mask

    # 精确定位印章区域
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_seal = rgba[y:y + h, x:x + w]

    # 保存为PNG
    cv2.imwrite(output_path, cropped_seal)
    print(f"成功提取印章保存至: {output_path}")
    return True


# # 使用示例
# if __name__ == "__main__":
#     # input_image = r"C:\Users\ZFH\Desktop\zhang.png"  # 替换为实际图片路径
#     # remove_black_text(input_image)
#     input_image = r"processed_image.png"  # 替换为实际图片路径
#     remove_black_text(input_image, output_path='processed_image_2.png')
#     # extract_seal(input_image)

# 非白色区域剔除，抠图，透明区域提取
def create_transparent_image(image_path, output_path='transparent_output.png'):
    """
    从文档图片中移除白色背景，创建透明背景的PNG图像

    参数:
    image_path (str): 输入图片路径
    output_path (str): 输出图片路径

    返回:
    bool: 处理成功返回True，否则False
    """
    # 读取图片（包含alpha通道）
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"错误: 无法读取图片 {image_path}")
        return False

    # 分离通道（BGRA格式）
    if img.shape[2] == 3:
        # 如果没有alpha通道，添加一个
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # 定义白色的RGB范围（考虑可能的色差）
    lower_white = np.array([220, 220, 220, 255])
    upper_white = np.array([255, 255, 255, 255])

    # 创建白色背景掩码
    white_mask = cv2.inRange(img, lower_white, upper_white)

    # 反转掩码：白色区域变成透明区域
    non_white_mask = cv2.bitwise_not(white_mask)

    # 应用透明通道
    b, g, r, a = cv2.split(img)
    a = cv2.bitwise_and(a, non_white_mask)

    # 合并回带有透明通道的图像
    transparent_img = cv2.merge((b, g, r, a))

    # 保存为PNG格式（支持透明背景）
    cv2.imwrite(output_path, transparent_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"成功创建透明背景图片保存至: {output_path}")

    return True


# 使用示例
if __name__ == "__main__":
    input_image = r"C:\Users\ZFH\Downloads\processed_image_2.png"  # 替换为您的图片路径
    create_transparent_image(input_image)