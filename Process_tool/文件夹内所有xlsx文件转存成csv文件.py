import os
import pandas as pd
import logging
from datetime import datetime
import traceback


def convert_xlsx_to_csv(folder_path, logger=None):
    """
    将文件夹中所有Excel文件的第一个工作表转换为CSV格式

    参数：
    folder_path: 目标文件夹路径
    logger: 日志记录器对象

    返回：
    成功转换的文件数量
    """
    # 初始化日志系统
    if logger is None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger('ExcelToCSVConverter')

    # 验证路径存在
    if not os.path.exists(folder_path):
        logger.error(f"目标文件夹不存在: {folder_path}")
        return 0

    processed_count = 0
    error_count = 0

    try:
        # 递归遍历文件夹
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()

                # 只处理Excel文件
                if file_ext not in ('.xlsx', '.xls'):
                    continue

                try:
                    # 读取Excel文件
                    logger.info(f"处理文件: {file_path}")

                    # 使用openpyxl引擎（支持.xlsx和.xls）
                    df = pd.read_excel(file_path, engine='openpyxl', sheet_name=0)

                    # 构建CSV文件路径
                    csv_path = os.path.splitext(file_path)[0] + '.csv'

                    # 保存为CSV
                    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

                    processed_count += 1
                    logger.info(f"成功转换: {file_path} -> {csv_path}")

                except Exception as e:
                    error_count += 1
                    logger.error(f"转换失败 [{file_path}]: {str(e)}")
                    logger.debug(traceback.format_exc())

        # 结果统计
        logger.info(f"操作完成! 成功转换 {processed_count} 个文件, 失败 {error_count} 个文件")
        return processed_count

    except Exception as e:
        logger.critical(f"全局错误: {str(e)}")
        return processed_count


# 带自定义日志记录器
import logging
logger = logging.getLogger('MyApp')
logger.setLevel(logging.DEBUG)
convert_xlsx_to_csv(
    folder_path=r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV\珠80\Texture_File',
    logger=logger
)