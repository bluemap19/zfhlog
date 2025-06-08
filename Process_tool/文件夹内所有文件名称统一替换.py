import os
import shutil
import logging
from datetime import datetime


def rename_path_components(path, old_str, new_str):
    """将路径中的每个组件进行替换处理"""
    head, tail = os.path.split(path)
    # 处理文件名或目录名
    if old_str in tail:
        tail = tail.replace(old_str, new_str)

    # 递归处理上层目录
    if head and head != os.path.dirname(head):  # 避免无限递归
        head = rename_path_components(head, old_str, new_str)

    return os.path.join(head, tail) if head else tail


def safe_rename(old_path, new_path):
    """安全重命名文件/目录，处理名称冲突"""
    if old_path == new_path:
        return False

    # 创建所有必要的上级目录
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    # 处理名称冲突
    if os.path.exists(new_path):
        base, ext = os.path.splitext(new_path)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        new_path = f"{base}_{counter}{ext}"

    os.rename(old_path, new_path)
    return True


def rename_paths_in_directory(folder_path, string_replace, string_target, backup=True, logger=None):
    """
    专业级文件路径名称批量替换工具

    参数：
    folder_path: 目标文件夹路径
    string_replace: 要替换的原始字符串
    string_target: 替换后的目标字符串
    backup: 是否创建备份(默认为True)
    logger: 日志记录器对象

    返回：
    成功重命名的项目数量
    """
    # 初始化日志系统
    if logger is None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger('PathRenamer')

    # 验证路径存在
    if not os.path.exists(folder_path):
        logger.error(f"目标文件夹不存在: {folder_path}")
        return 0

    # 记录原始路径用于备份和恢复
    original_path = folder_path

    # 创建备份目录
    if backup:
        backup_dir = os.path.join(
            os.path.dirname(folder_path),
            f"backup_rename_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(backup_dir, exist_ok=True)
        backup_target = os.path.join(backup_dir, os.path.basename(folder_path))
        shutil.copytree(folder_path, backup_target)
        logger.info(f"创建完整备份: {backup_target}")

    processed_count = 0

    try:
        # 先处理所有子目录和文件（从最深层次开始）
        all_items = []
        for root, dirs, files in os.walk(folder_path, topdown=False):
            # 先处理文件
            for file in files:
                full_path = os.path.join(root, file)
                all_items.append(full_path)

            # 然后处理目录（不包含当前根目录）
            for dir_name in dirs:
                full_path = os.path.join(root, dir_name)
                all_items.append(full_path)

        # 最后处理根目录
        all_items.append(folder_path)

        # 按照路径长度降序排序（最深的先处理）
        all_items.sort(key=lambda p: len(p), reverse=True)

        # 重命名路径中的所有组件
        for item_path in all_items:
            # 计算新的路径
            new_path = rename_path_components(item_path, string_replace, string_target)

            # 如果路径有变化则执行重命名
            if item_path != new_path:
                if safe_rename(item_path, new_path):
                    processed_count += 1
                    logger.info(f"重命名: '{item_path}' -> '{new_path}'")

                    # 如果重命名的是根目录，更新当前工作路径
                    if item_path == folder_path:
                        folder_path = new_path

        logger.info(f"操作完成! 共重命名 {processed_count} 个项目")
        return processed_count

    except Exception as e:
        logger.critical(f"操作失败: {str(e)}")

        # 如果发生错误且有备份，尝试恢复
        if backup and 'backup_target' in locals():
            try:
                logger.info("尝试从备份恢复...")
                if os.path.exists(original_path):
                    shutil.rmtree(original_path)
                shutil.copytree(backup_target, original_path)
                logger.info(f"成功从备份恢复: {backup_target} -> {original_path}")
            except Exception as restore_error:
                logger.critical(f"恢复备份失败: {str(restore_error)}")

        return processed_count


# # 基本用法 - 替换路径中的关键字
# rename_paths_in_directory(
#     folder_path=r'C:\Project\AA_Data',
#     string_replace='AA',
#     string_target='BB'
# )

# 高级用法 - 不创建备份
rename_paths_in_directory(
    folder_path=r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV\珠80\Texture_File',
    string_replace='Z80_Texture_ALL',
    string_target='珠80_Texture_ALL_logging',
    backup=False
)

# # 带自定义日志记录器
# import logging
# logger = logging.getLogger('MyApp')
# logger.setLevel(logging.DEBUG)
#
# rename_paths_in_directory(
#     folder_path=r'D:\Documents\AA_Reports',
#     string_replace='AA',
#     string_target='BB',
#     logger=logger
# )