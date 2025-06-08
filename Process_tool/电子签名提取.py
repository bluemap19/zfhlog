from PIL import Image


def process_signature(input_path, output_path, background_threshold=240):
    """
    处理签名图片并生成透明背景PNG
    :param input_path: 输入图片路径
    :param output_path: 输出图片路径
    :param background_threshold: 背景颜色阈值（0-255）
    """
    # 打开图片并转换为RGBA模式
    img = Image.open(input_path).convert("RGBA")
    pixels = img.load()

    # 遍历所有像素
    for y in range(img.height):
        for x in range(img.width):
            r, g, b, a = pixels[x, y]

            # 判断是否为背景像素（浅色部分）
            if r > background_threshold and g > background_threshold and b > background_threshold:
                # 将背景设为完全透明
                pixels[x, y] = (255, 255, 255, 0)
            else:
                # 保留签名颜色，设置完全不透明
                pixels[x, y] = (r, g, b, 255)

    # 保存为PNG
    img.save(output_path, "PNG")


# 使用示例
process_signature(
    # input_path=r"C:\Users\ZFH\Desktop\LWD_ZKY\项目结题材料\电子签名\孔雪.jpg",
    input_path=r"C:\Users\ZFH\Desktop\JJJ.png",
    output_path="kkk.png",
    background_threshold=220  # 根据实际情况调整阈值
)