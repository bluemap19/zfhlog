import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from DongYing_WangZiMing.image_to_logging import image_to_trend_curve
from DongYing_WangZiMing.linear_fitting_robust import robust_linear_fit
from DongYing_WangZiMing.pic_transpose import shear_image
from DongYing_WangZiMing.test_depth_correction import denoising_depth_correction
from src_file_op.dir_operation import search_files_by_criteria
from src_fmi.image_operation import show_Pic
from src_plot.plot_logging import visualize_well_logs

# 把岩心扫描图转换成 条带状的地层图
if __name__ == '__main__':
    # PATH_FOLDER = r'C:\Users\ZFH\Desktop\1-15'
    PATH_FOLDER = r'C:\Users\ZFH\Desktop\5b'
    # PATH_FOLDER = r'C:\Users\ZFH\Desktop\6b'
    path_list_target = search_files_by_criteria(PATH_FOLDER, name_keywords=['-'],
                             file_extensions=['.bmp'])

    print(path_list_target)

    for path in path_list_target:
        print('current processing file as:', path)
        well_name = path.split('\\')[-1].split('.')[0]
        IMAGE = cv2.imread(path, cv2.IMREAD_COLOR) # 5646465*1024*3
        IMAGE = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2RGB)
        IMAGE = IMAGE.transpose(1, 0, 2)        # 长宽调换

        width = 256
        height = int(IMAGE.shape[0] * width / IMAGE.shape[1])
        dim = (width, height)
        IMAGE_SCALED = cv2.resize(IMAGE, dim, interpolation = cv2.INTER_AREA)
        print('origin image shape transform to scaled image:{}--->{}'.format(IMAGE.shape,IMAGE_SCALED.shape))

        # 图像灰度化
        IMAGE_GREY = 0.2 * IMAGE_SCALED[:, :, 0] + 0.1 * IMAGE_SCALED[:, :, 1] + 0.7 * IMAGE_SCALED[:, :, 2]

        depth_start = 0
        depth_resolution = 0.0025
        depth_end = IMAGE_GREY.shape[0] * depth_resolution
        DEPTH_TEMP = np.linspace(depth_start, depth_end, IMAGE_GREY.shape[0]).reshape(-1, 1)
        print('IMAGE_GREY shape :{} and set depth shape:{},resolution is {}, depth from {} to {}'.format(IMAGE_GREY.shape, DEPTH_TEMP.shape, depth_resolution, depth_start, depth_end))

        # 这个很重要，如果选的区域较大，就会有比较大的噪声影响，比较小的话，拟合又容易不准
        # IMAGE_GREY = IMAGE_GREY[:, width//7*3:-width//7*3]
        IMAGE_GREY = IMAGE_GREY[:, width//3:width//3*2]
        DF_GREY = pd.DataFrame(np.hstack([DEPTH_TEMP, IMAGE_GREY]))
        target_index = IMAGE_GREY.shape[1]//2 + 1
        data_shift = []
        for j in range(1, DF_GREY.shape[1]):
            index_list = [0, target_index, j]
            df_depth_correct = DF_GREY.iloc[:, index_list]
            correct_shift, _ = denoising_depth_correction(df_depth_correct, depth_col=0, base_col=1, process_col=2,
                                       search_range=0.5, coarse_res=0.02, fine_res=0.001,
                                       denoise_window_point=7, denoise_polyorder=2, plot=False)
            data_shift.append([j, correct_shift])

        # 这个是做PPT图用的东西
        # DF_PLOT = DF_GREY[[0, 10, 42, 72]]
        # DF_PLOT.columns = ['DEPTH', 'D1', 'D2', 'D3']
        # visualize_well_logs(DF_PLOT, depth_col = 'DEPTH', curve_cols = ['D1', 'D2', 'D3'], type_cols = [], figsize = (24, 10))

        data_shift = pd.DataFrame(data_shift, columns=['x', 'y'])
        print("======== 过原点线性模型拟合 (y = ax) ========")
        # model, slope, r2, inliers = robust_linear_fit(
        #     data_shift, x_col='x', y_col='y',
        #     degree=1, fit_intercept=True,
        #     min_samples=0.3, residual_threshold=0.02, plot=True
        # )
        model, slope, intercept, r2, inliers = robust_linear_fit(
            data_shift, x_col='x', y_col='y',
            degree=1, fit_intercept=True,
            min_samples=0.3, residual_threshold=0.0008, plot=False
        )
        slope/=depth_resolution
        intercept/=intercept
        print('file {} use slope as y={}x + {}, it\'s r2 is {}'.format(path.split('\\')[-1].split('.')[0], slope, intercept, r2))

        # 进行图像的拉伸、菱形拉伸
        IMAGE_SHEER = shear_image(IMAGE_SCALED, slope, intercept)

        IMAGE_SHEER_GREY = 0.2 * IMAGE_SHEER[:, :, 0] + 0.1 * IMAGE_SHEER[:, :, 1] + 0.7 * IMAGE_SHEER[:, :, 2]
        # 计算图像最中心的那一系列的均值
        width_target = 16
        middle = width//2
        IMAGE_MIDDLE = IMAGE_SHEER_GREY[:, middle-width_target//2:middle+width_target//2]
        depth_start = 0
        depth_end = IMAGE_MIDDLE.shape[0] * depth_resolution + depth_start
        DEPTH_MIDDLE = np.linspace(depth_start, depth_end, IMAGE_MIDDLE.shape[0]).reshape(-1, 1)
        print('IMAGE_MIDDLE used to calculate logging response, IMAGE_MIDDLE shape is {}, and it\'s depth info is {}'.format(IMAGE_MIDDLE.shape, DEPTH_MIDDLE.shape))

        LOGGING_IMAGE = image_to_trend_curve(IMAGE_MIDDLE, plot=False).reshape(-1, 1)
        DF_LOGGING_IMAGE = pd.DataFrame(np.hstack([DEPTH_MIDDLE, LOGGING_IMAGE]))
        print('LOGGING_IMAGE shape is {}, and DF_LOGGING_IMAGE shape is {}'.format(LOGGING_IMAGE.shape, DF_LOGGING_IMAGE.shape))
        # # 图像数据提取的矩阵保存
        DF_LOGGING_IMAGE.to_csv(PATH_FOLDER+ '\\target\\' + well_name + '.csv', index=False)

        # 复制响应值到每一行的所有列
        IMAGE_STRIPE = np.repeat(LOGGING_IMAGE, width, axis=1)
        IMAGE_STRIPE_REVERSED = shear_image(IMAGE_STRIPE, -slope, intercept)

        show_Pic([IMAGE_SCALED, IMAGE_SHEER, IMAGE_STRIPE, IMAGE_STRIPE_REVERSED], figure=(8, 16), pic_order='14', title=f'WELL {well_name}',
                 pic_str=[f"原始图像 ({height}×{width})", f"斜率 slope={slope:.4f}", "地层图像", '反转地层图像'],
                 path_save=PATH_FOLDER+ '\\target\\' + well_name + '.png', show=False
        )

        # exit(0)
        print('+++++++++++++++++++++++++++++++++++++++++ current well finished +++++++++++++++++++++++++++++++++++++++++')





