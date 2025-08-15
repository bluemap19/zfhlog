import cv2
import numpy as np
import pandas as pd

from src_data_process.cal_data_glcm_texture import cal_image_texture
from src_file_op.dir_operation import search_files_by_criteria
from src_fmi.image_operation import show_Pic

if __name__ == '__main__':

    PATH_FOLDER = r'C:\Users\ZFH\Desktop\1-15'
    path_list_target = search_files_by_criteria(PATH_FOLDER, name_keywords=['-'],
                             file_extensions=['.bmp'])

    print(path_list_target)

    LIST_IMAGE_ORG = []
    IMAGE_ALL = np.array([])
    Image_Log = []
    for path in path_list_target:
        print(path)
        IMAGE = cv2.imread(path, cv2.IMREAD_COLOR)
        IMAGE = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2RGB)

        width = 512
        height = int(IMAGE.shape[1] * width / IMAGE.shape[0])
        dim = (height, width)
        print(IMAGE.shape)
        IMAGE_SCALED = cv2.resize(IMAGE, dim, interpolation = cv2.INTER_AREA)

        print(IMAGE_SCALED.shape)

        if IMAGE_ALL.size == 0:
            IMAGE_ALL = IMAGE_SCALED
        else:
            IMAGE_ALL = np.hstack((IMAGE_ALL, IMAGE_SCALED))

        Image_Log.append([path.split('\\')[-1], IMAGE_SCALED.shape, IMAGE.shape])

    # show_Pic([IMAGE_ALL_Cropped[:, :400, :], IMAGE_ALL_Cropped[:, 400:800, :],
    #           IMAGE_ALL_Cropped[:, 800:1200, :], IMAGE_ALL_Cropped[:, 1200:1600, :],
    #           IMAGE_ALL_Cropped[:, 2000:2400, :], IMAGE_ALL_Cropped[:, 5000:5400, :],
    #           IMAGE_ALL_Cropped[:, 8000:8400, :], IMAGE_ALL_Cropped[:, 10000:10400, :]],
    #          pic_order='24', figure=(16, 8))

    # 图像灰度化
    IMAGE_ALL = 0.2 * IMAGE_ALL[:, :, 0] + 0.1 * IMAGE_ALL[:, :, 1] + 0.7 * IMAGE_ALL[:, :, 2]
    IMAGE_ALL = IMAGE_ALL.transpose((1, 0))

    IMAGE_ALL_Cropped = IMAGE_ALL[:, 128:-128]
    print(IMAGE_ALL.shape)
    print(IMAGE_ALL_Cropped.shape)

    # 转换为DataFrame
    df_log = pd.DataFrame(
        [
            {
                '井名': item[0].replace('.bmp', ''),  # 移除文件扩展名
                'W1': item[1][0],  # 第一个元组的第一个元素
                'H1': item[1][1],  # 第一个元组的第二个元素
                'W2': item[2][0],  # 第二个元组的第一个元素
                'H2': item[2][1]  # 第二个元组的第二个元素
            }
            for item in Image_Log
        ]
    )
    df_log.to_csv('config.csv', index=False)
    print(df_log)

    # show_Pic([IMAGE_ALL_Cropped[:, :400], IMAGE_ALL_Cropped[:, 400:800],
    #           IMAGE_ALL_Cropped[:, 800:1200], IMAGE_ALL_Cropped[:, 1200:1600],
    #           IMAGE_ALL_Cropped[:, 2000:2400], IMAGE_ALL_Cropped[:, 5000:5400],
    #           IMAGE_ALL_Cropped[:, 8000:8400], IMAGE_ALL_Cropped[:, 10000:10400]],
    #          pic_order='24', figure=(16, 8))

    print(IMAGE_ALL_Cropped.shape)
    depth_start = 100
    DEPTH = np.linspace(depth_start, depth_start+0.0025*IMAGE_ALL_Cropped.shape[0], IMAGE_ALL_Cropped.shape[0])
    print(DEPTH.shape, DEPTH[0], DEPTH[-1])


    windows_length = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    for windows in windows_length:
        cal_image_texture(IMAGE_ALL_Cropped, DEPTH, windows=windows, step=2, path_texture_saved='C:\\Users\\ZFH\\Desktop\\1-15\\well_texture_all\\texture_{}_logging.csv'.format(windows))

    # list_image_cropped_head = ['DEPTH']
    # for i in range(IMAGE_ALL_Cropped.shape[1]):
    #     list_image_cropped_head.append('D{:03d}'.format(i))
    # IMAGE_ALL_Cropped = np.hstack((DEPTH.reshape((-1, 1)), IMAGE_ALL_Cropped))
    # IMAGE_ALL_Cropped = pd.DataFrame(IMAGE_ALL_Cropped, columns=list_image_cropped_head)
    # IMAGE_ALL_Cropped.to_csv('Image_Cropped.csv', index=False)
    #
    # list_image_all_head = ['DEPTH']
    # for i in range(IMAGE_ALL.shape[1]):
    #     list_image_all_head.append('D{:03d}'.format(i))
    # IMAGE_ALL = np.hstack((DEPTH.reshape((-1, 1)), IMAGE_ALL))
    # IMAGE_ALL = pd.DataFrame(IMAGE_ALL, columns=list_image_all_head)
    # IMAGE_ALL.to_csv('Image_All.csv', index=False)


