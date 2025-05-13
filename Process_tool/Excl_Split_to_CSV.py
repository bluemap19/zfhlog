import pandas as pd
import os
import re
from pathlib import Path


def excel_sheets_to_csv(excel_path, output_folder=None, encoding='utf-8-sig', postfix=''):
    """
    将Excel文件每个Sheet保存为单独的CSV文件

    参数：
        excel_path (str): Excel文件路径
        output_folder (str): 输出目录（默认同Excel文件所在目录）
        encoding (str): CSV文件编码格式（默认utf-8-sig解决中文乱码）

    返回：
        list: 生成的CSV文件路径列表
    """
    # 验证输入文件
    if not os.path.isfile(excel_path):
        raise FileNotFoundError(f"Excel文件不存在：{excel_path}")

    if not excel_path.lower().endswith(('.xls', '.xlsx')):
        raise ValueError("文件类型错误，仅支持.xls和.xlsx格式")

    # 设置输出路径
    excel_dir = os.path.dirname(excel_path)
    excel_name = os.path.splitext(os.path.basename(excel_path))[0]
    output_dir = output_folder or os.path.join(excel_dir, f"{excel_name}_CSV")

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 读取Excel所有Sheet名称
    try:
        sheets = pd.ExcelFile(excel_path).sheet_names
    except Exception as e:
        raise RuntimeError(f"读取Excel文件失败：{str(e)}")

    saved_files = []

    # 处理每个Sheet
    for sheet_name in sheets:
        try:
            # 读取Sheet数据
            df = pd.read_excel(excel_path, sheet_name=sheet_name)

            # 清理非法文件名字符
            clean_name = re.sub(r'[\\/*?:"<>|]', '_', sheet_name.strip())

            # 构建CSV路径
            if postfix == '':
                csv_path = os.path.join(output_dir, f"{clean_name}.csv")
            else:
                csv_path = os.path.join(output_dir, f"{clean_name}_{postfix}.csv")

            # 保存CSV（自动处理不同数据类型）
            df.to_csv(csv_path,
                      index=False,
                      encoding=encoding,
                      na_rep='',  # 空值处理
                      float_format='%.4f')  # 浮点数精度
            saved_files.append(csv_path)
            print(f"已生成：{csv_path}")

        except Exception as e:
            print(f"处理Sheet [{sheet_name}] 失败：{str(e)}")
            continue

    print(f"\n转换完成！共生成 {len(saved_files)} 个CSV文件")
    return saved_files


if __name__ == "__main__":
    # 用户输入
    # input_excel = input("请输入Excel文件路径：").strip()
    input_excel = r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\Code_input-O\汇总150口井提交-长73岩相单井剖面+岩相厚度统计_LITHO_TYPE.xlsx'

    # 执行转换
    try:
        result = excel_sheets_to_csv(input_excel, postfix='_LITHO_TYPE')
        print("\n生成的CSV文件列表：")
        print("\n".join(result))
    except Exception as e:
        print(f"\n错误发生：{str(e)}")