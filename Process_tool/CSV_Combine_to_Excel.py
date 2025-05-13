import os
import pandas as pd
from pathlib import Path
import re


def csv_folder_to_excel(folder_path, output_excel=None, max_sheet_name=31):
    """
    将文件夹内CSV合并为多Sheet的Excel文件

    参数：
        folder_path (str): 包含CSV文件的文件夹路径
        output_excel (str): 输出Excel路径（默认：文件夹名_combined.xlsx）
        max_sheet_name (int): Sheet名称最大长度（Excel限制31字符）

    返回：
        str: 生成的Excel文件路径
    """
    # 验证输入路径
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"文件夹路径不存在：{folder_path}")

    # 获取所有CSV文件
    csv_files = [f for f in Path(folder_path).glob('*.csv') if f.is_file()]
    if not csv_files:
        raise FileNotFoundError("文件夹内未找到CSV文件")

    # 设置输出路径
    if not output_excel:
        folder_name = os.path.basename(folder_path.rstrip('/\\'))
        output_excel = os.path.join(folder_path, f"{folder_name}_COMBINED.xlsx")

    # 清理已存在的输出文件
    if Path(output_excel).exists():
        os.remove(output_excel)

    # 定义合法Sheet名称的正则表达式
    illegal_chars = re.compile(r'[\\/:*?"<>|$$$$]')

    # 创建Excel写入器
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        for csv_file in csv_files:
            try:
                # 生成合法Sheet名称
                sheet_name = csv_file.stem[:max_sheet_name]  # 截断长度
                sheet_name = illegal_chars.sub('_', sheet_name)  # 替换非法字符

                # 处理重复Sheet名称
                original_name = sheet_name
                counter = 1
                while sheet_name in writer.book.sheetnames:
                    sheet_name = f"{original_name[:25]}_{counter}"
                    counter += 1

                # 读取CSV（自动检测编码）
                df = pd.read_csv(csv_file, encoding=detect_encoding(csv_file))

                # 写入Excel
                df.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                    header=True
                )
                print(f"已合并：{csv_file.name} -> Sheet [{sheet_name}]")

            except Exception as e:
                print(f"处理文件失败：{csv_file.name} - {str(e)}")
                continue

    print(f"\n合并完成！生成文件：{output_excel}")
    return output_excel


def detect_encoding(file_path, sample_size=1024):
    """
    自动检测文件编码
    """
    import chardet
    with open(file_path, 'rb') as f:
        rawdata = f.read(sample_size)
    return chardet.detect(rawdata)['encoding']


if __name__ == "__main__":
    # 用户输入
    # input_folder = input("请输入CSV文件夹路径：").strip()
    input_folder = r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\Code_input-O\汇总150口井提交-长73岩相单井剖面+岩相厚度统计_LITHO_TYPE_CSV'

    # 执行合并
    try:
        output_path = csv_folder_to_excel(input_folder)
        print(f"输出文件已保存至：{output_path}")
    except Exception as e:
        print(f"错误发生：{str(e)}")