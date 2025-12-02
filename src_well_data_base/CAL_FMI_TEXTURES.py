import logging

import numpy as np
import pandas as pd

from src_well_data_base.data_logging_FMI import DataFMI


def user_specific_test():
    """
    用户特定测试 - 使用用户提供的文件路径
    """
    print("\n" + "=" * 60)
    print("用户特定测试")
    print("=" * 60)

    FOLDER_PATH = r'F:\logging_workspace\禄探'
    WINDOWS_LENGTH = 80
    WINDOWS_STEP = 10

    # 用户提供的测试用例
    test_case_DYNA = {
        'path_fmi': FOLDER_PATH + r'\禄探_DYNA.txt',
        'fmi_charter': 'DYNA'
    }
    test_case_STAT = {
        'path_fmi': FOLDER_PATH + r'\禄探_STAT.txt',
        'fmi_charter': 'STAT'
    }

    print(f"测试文件: {test_case_DYNA['path_fmi']} + {test_case_STAT['path_fmi']}")
    print(f"仪器: {test_case_DYNA['fmi_charter']} + {test_case_STAT['fmi_charter']}")
    print("-" * 50)

    try:
        # 创建FMI处理器实例
        test_FMI_DYNA = DataFMI(
            path_fmi=test_case_DYNA['path_fmi'],
            fmi_charter=test_case_DYNA['fmi_charter']
        )
        test_FMI_STAT = DataFMI(
            path_fmi=test_case_STAT['path_fmi'],
            fmi_charter=test_case_STAT['fmi_charter']
        )

        # # 执行用户要求的操作序列
        # print(">>> 执行空白条带删除...")
        # test_FMI_DYNA.ele_stripes_delete()
        # test_FMI_STAT.ele_stripes_delete()

        print(">>> 获取数据...")
        fmi_data_dyna, depth_data = test_FMI_DYNA.get_data()
        fmi_data_stat, depth_data = test_FMI_STAT.get_data()

        print(f"FMI数据形状: {fmi_data_dyna.shape}")
        print(f"深度数据形状: {depth_data.shape}")

        if depth_data.size > 0:
            print(f"深度范围: {depth_data.min():.3f} - {depth_data.max():.3f}")

        # 显示数据摘要
        print("\n>>> 数据摘要:")
        summary = test_FMI_DYNA.get_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")

        FMI_TEXTURE_DYNA = test_FMI_DYNA.get_texture(texture_config = {
                'level': 16,  # 灰度级别
                'distance': [2, 4],  # 像素距离
                # 'angles': [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],  # 角度方向
                'angles': [0, np.pi / 2],  # 角度方向
                'windows_length': WINDOWS_LENGTH,  # 窗口长度
                'windows_step': WINDOWS_STEP  # 滑动步长
            }
        )
        FMI_TEXTURE_STAT = test_FMI_STAT.get_texture(texture_config={
            'level': 16,  # 灰度级别
            'distance': [2, 4],  # 像素距离
            # 'angles': [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],  # 角度方向
            'angles': [0, np.pi / 2],  # 角度方向
            'windows_length': WINDOWS_LENGTH,  # 窗口长度
            'windows_step': WINDOWS_STEP  # 滑动步长
        }
        )
        print(FMI_TEXTURE_DYNA.describe())
        print(FMI_TEXTURE_STAT.describe())
        COLS_DYNA = list(FMI_TEXTURE_DYNA.columns)
        COLS_STAT = list(FMI_TEXTURE_STAT.columns)
        print(COLS_DYNA, '\n',COLS_STAT)
        TEXTURE_ALL = pd.concat([FMI_TEXTURE_STAT, FMI_TEXTURE_DYNA[COLS_DYNA]], axis=1)
        TEXTURE_ALL.to_csv(FOLDER_PATH + f'\\禄探_TEXTURE_{WINDOWS_LENGTH}.csv', index=False)


    except FileNotFoundError:
        print(f"文件不存在: {test_case_DYNA['path_fmi']}")
        print("跳过该测试用例...")
    except Exception as e:
        print(f"测试失败: {e}")
        print("错误详情:", str(e))



if __name__ == '__main__':
    """
    主程序入口
    执行顺序：
    1. 综合测试（使用模拟数据）
    2. 用户特定测试（使用用户提供的文件路径）
    """

    # 设置日志级别
    logging.basicConfig(level=logging.INFO)

    # 执行用户特定测试
    user_specific_test()

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)