import logging

import cv2
import numpy as np

from src_fmi.fmi_data_read import get_random_ele_data
from src_fmi.fmi_fractal_dimension import adaptive_binarization, edge_detection
from src_fmi.fractal_dimension_gxm import multifractal_analysis
from src_plot.TEMP_8 import WellLogVisualizer


def pic_preprocess(image, processing_method='adaptive_binary'):
    # 步骤2: 图像预处理 - 根据地质特征选择合适方法
    if processing_method == 'adaptive_binary':
        # 适合分析孔隙结构、岩性边界
        processed_image = adaptive_binarization(image, 'otsu_adaptive')
    elif processing_method == 'edge_detection':
        # 适合分析裂缝、层理面等线性特征
        processed_image = edge_detection(image, 'canny_adaptive')
    elif processing_method == 'original':
        # 直接使用原始数据，保留所有信息
        processed_image = image
    else:
        processed_image = image

    return processed_image


def cal_fmi_multidimensional_fractal_spectrum(fmi_dict={}, windows_length=200, windows_step=10, processing_method='adaptive_binary'):
    """
        计算电成像数据的分形维数曲线 - 主控函数

        滑动窗口原理：
        通过窗口在深度方向上滑动，计算每个窗口内电成像图像的分形特征，
        形成随深度变化的分形维数曲线，反映地层复杂度的纵向变化。
    """
    # 1. 数据提取，从字典中正确提取深度、电成像图像
    data_depth = fmi_dict['depth'] if 'depth' in fmi_dict else None
    data_img = fmi_dict['image'] if 'image' in fmi_dict else None

    # 数据完整性检查
    if data_depth is None or data_img is None:
        raise ValueError("fmi_dict必须包含'depth', 'image' 两个键")
    # 数据维度一致性检查
    if data_depth.shape[0] != data_img.shape[0]:
        raise ValueError("深度数据与图像数据长度不匹配")

    data_depth = data_depth.ravel()
    windows_num = data_depth.shape[0] // windows_step

    multi_fractal_dimension_result = {}          # 存储深度-分形维数对

    # 初始化结果数组，用于构建完整深度的处理结果
    fmi_result_weight = np.zeros_like(data_img, dtype=np.float32)
    fmi_result = np.zeros_like(data_img, dtype=np.float32)

    # 滑动窗口遍历电成像数据
    for i in range(windows_num):
        index_middle = i * windows_step  # 窗口中心深度
        index_start = max(0, index_middle - windows_length // 2)  # 窗口起始索引，防止越界
        index_end = min(data_depth.shape[0], index_middle + windows_length // 2)  # 窗口结束索引，防止越界

        data_image_window = data_img[index_start:index_end]  # 提取当前窗口内的电成像数据
        depth_t = data_depth[index_middle]  # 当前窗口的中心深度

        processed_img = pic_preprocess(data_image_window, processing_method=processing_method)

        # 计算当前窗口的分形维数 fd_param:当前窗口的分形维数值 processed_img:经过预处理后的窗口图像，边缘检测结果，或者是二值化结果
        analysis_results = multifractal_analysis(image_array=processed_img, q_step=0.2)

        # 存储结果 'NMR_X', 'NMR_Y'
        multi_fractal_dimension_result[depth_t] = {'NMR_X':analysis_results['singularity_exponents'], 'NMR_Y':(analysis_results['multifractal_spectrum']-1.5)}

        # 将处理结果映射回原始尺寸，用于构建完整深度剖面
        processed_img = cv2.resize(processed_img, data_image_window.shape[::-1], cv2.INTER_LINEAR)
        fmi_result[index_start:index_end] += processed_img
        fmi_result_weight[index_start:index_end] += 1

    # 加权平均处理结果（权重归一化）
    fmi_result = fmi_result / fmi_result_weight

    return fmi_result.astype(np.uint8), multi_fractal_dimension_result

def cal_fmis_fractal_dimension_extended(fmi_dict={}, windows_length=200, windows_step=10, method='differential_box', processing_method='adaptive_binary'):
    """
    计算电成像数据的分形维数曲线 - 主控函数

    滑动窗口原理：
    通过窗口在深度方向上滑动，计算每个窗口内电成像图像的分形特征，
    形成随深度变化的分形维数曲线，反映地层复杂度的纵向变化。
    """
    # 1. 数据提取
    # 从字典中正确提取深度、动态图像、静态图像数据
    data_depth = fmi_dict['depth'] if 'depth' in fmi_dict else None
    data_imgs = fmi_dict['fmis'] if 'fmis' in fmi_dict else None
    # output_curves_list = fmi_dict['output_curves_list'] if 'output_curves_list' in fmi_dict else None

    # 数据完整性检查
    if data_depth is None or data_imgs is None:
        raise ValueError("fmi_dict必须包含'depth', 'fmis' 键")

    # 数据维度一致性检查
    for i in range(len(data_imgs)):
        if data_depth.shape[0] != data_imgs[i].shape[0]:
            raise ValueError("深度数据与图像数据长度不匹配")

    fmi_multi_fde_list = []
    fmi_result_list = []
    for i in range(len(data_imgs)):
        # 2. 动态成像的分形维数计算
        fmi_result, multi_fractal_dimension_result = cal_fmi_multidimensional_fractal_spectrum(
            fmi_dict={'depth':data_depth, 'image':data_imgs[i]},
            windows_length=windows_length,  # 窗口长度：平衡纵向分辨率和统计可靠性
            windows_step=windows_step,  # 滑动步长：控制计算密度
            # edge_detection adaptive_binary original
            processing_method=processing_method,
        )
        fmi_result_list.append(fmi_result)
        fmi_multi_fde_list.append(multi_fractal_dimension_result)

    return fmi_result_list, fmi_multi_fde_list

if __name__ == '__main__':
    """
    主程序执行流程
    """
    # 1. 数据读取 - 获取电成像动态、静态数据和深度信息
    data_img_dyna, data_img_stat, data_depth = get_random_ele_data()
    print(f"数据形状: 动态{data_img_dyna.shape}, 静态{data_img_stat.shape}, 深度{data_depth.shape}")

    # 2. 分形维数计算
    fmi_result_list, fmi_multi_fde_list = cal_fmis_fractal_dimension_extended(
        fmi_dict={
            'depth': data_depth,  # 深度数据
            'fmis': [data_img_stat, data_img_dyna],  # 电成像数据
        },
        windows_length=100,  # 窗口长度：平衡纵向分辨率和统计可靠性
        windows_step=20,  # 滑动步长：控制计算密度
        processing_method='original',  # 图像预处理的方式：自适应二值化突出岩性边界
    )

    # 使用类接口进行可视化
    print("创建可视化器...")
    visualizer = WellLogVisualizer()
    try:
        # 启用详细日志级别
        logging.getLogger().setLevel(logging.INFO)

        # 执行可视化
        visualizer.visualize(
            logging_dict=None,
            fmi_dict={  # FMI图像数据
                'depth': data_depth,
                # 'image_data': [data_img_dyna, data_img_stat]+fmi_result_list,
                # 'title': ['FMI动态', 'FMI静态', 'DYNA_binary', 'STAT_binary']
                'image_data': [data_img_dyna, data_img_stat],
                'title': ['FMI动态', 'FMI静态']
            },
            NMR_dict=fmi_multi_fde_list,
            NMR_Config={'X_LOG': [False, False], 'NMR_TITLE': ['α-fα-DYNA', 'α-fα-STAT'], 'X_LIMIT':[[1.0, 4], [1.0, 4]], 'Y_scaling_factor': 0.1},
            # depth_limit_config=[320, 380],  # 深度限制
            figsize=(12, 8)  # 图形尺寸
        )

        # 显示性能统计
        stats = visualizer.get_performance_stats()
        print("性能统计:", stats)

    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback

        traceback.print_exc()  # 打印完整错误堆栈
    finally:
        # 清理资源
        visualizer.close()

