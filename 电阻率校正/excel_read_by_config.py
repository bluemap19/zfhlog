import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path


def parse_xml_config(config_path):
    """解析XML配置文件"""
    tree = ET.parse(config_path)
    root = tree.getroot()

    config = {}

    # 文件设置
    file_settings = root.find('file_settings')
    config['input_path'] = file_settings.find('input_path').text
    config['output_path'] = file_settings.find('output_path').text

    # 工作表设置
    sheet_settings = root.find('sheet_settings')
    config['sheet_name'] = sheet_settings.find('sheet_name').text
    config['header_row'] = int(sheet_settings.find('header_row').text)
    config['skip_rows'] = int(sheet_settings.find('skip_rows').text)

    # 列设置
    column_settings = root.find('column_settings')
    config['usecols'] = column_settings.find('usecols').text

    # 数据类型映射
    config['dtypes'] = {}
    dtypes = column_settings.find('dtypes')
    for dtype in dtypes.findall('dtype'):
        col = dtype.get('column')
        dtype_val = dtype.text
        config['dtypes'][col] = dtype_val

    # 数据处理设置
    data_processing = root.find('data_processing')
    config['na_values'] = [v.strip() for v in data_processing.find('na_values').text.split(',')]
    config['parse_dates'] = [col.strip() for col in data_processing.find('parse_dates').text.split(',')]
    config['date_format'] = data_processing.find('date_format').text

    # 引擎设置
    engine_settings = root.find('engine_settings')
    config['engine'] = engine_settings.find('engine').text

    return config


def read_excel_with_xml_config(config_path='excel_config.xml'):
    """
    使用XML配置文件读取Excel文件

    参数:
    config_path: XML配置文件路径

    返回:
    DataFrame: 读取的Excel数据
    """
    # 解析XML配置
    config = parse_xml_config(config_path)

    # 获取文件路径
    file_path = config['input_path']

    # 验证文件存在
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Excel文件不存在: {file_path}")

    # 构建读取参数
    read_params = {
        'sheet_name': config['sheet_name'],
        'header': config['header_row'],
        'skiprows': config['skip_rows'],
        'usecols': config['usecols'],
        'engine': config['engine']
    }

    # 添加可选参数
    if 'na_values' in config and config['na_values']:
        read_params['na_values'] = config['na_values']

    if 'parse_dates' in config and config['parse_dates']:
        read_params['parse_dates'] = config['parse_dates']

    if 'dtypes' in config and config['dtypes']:
        # 转换字符串类型为实际dtype
        dtype_map = {
            'int': 'int32',
            'float': 'float32',
            'str': 'str',
            'category': 'category',
            'bool': 'bool',
            'datetime': 'datetime64'
        }
        dtypes = {col: dtype_map.get(dtype, 'object')
                  for col, dtype in config['dtypes'].items()}
        read_params['dtype'] = dtypes

    # 读取Excel文件
    try:
        df = pd.read_excel(file_path, **read_params)
        print(f"成功读取Excel文件: {file_path}")
        return df
    except Exception as e:
        print(f"读取Excel文件失败: {str(e)}")
        raise


def process_and_save_data(df, config_path='excel_config.xml'):
    """
    处理数据并保存结果

    参数:
    df: 要处理的DataFrame
    config_path: XML配置文件路径
    """
    config = parse_xml_config(config_path)

    # 日期格式转换
    if 'parse_dates' in config and config['parse_dates']:
        date_format = config.get('date_format', None)
        for col in config['parse_dates']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format=date_format)

    # 保存处理后的数据
    output_path = config['output_path']
    df.to_excel(output_path, index=False)
    print(f"数据已保存至: {output_path}")


# 使用示例
if __name__ == "__main__":
    try:
        # 读取Excel数据
        sales_data = read_excel_with_xml_config()

        # 显示前5行
        print("\n数据预览:")
        print(sales_data.head())

        # 处理并保存数据
        process_and_save_data(sales_data)

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")