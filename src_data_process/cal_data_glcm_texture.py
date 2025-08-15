import copy
import math

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from src_fmi.image_operation import get_glcm_Features, get_glcm_sub


def cal_image_texture(img, depth, windows=20, step=5, texture_config={'level': 32, 'distance':[1, 2], 'angles':[0, np.pi/4, np.pi/2, np.pi*3/4]}, path_texture_saved='', Curve_List=[]):
    assert img.shape[0] == depth.shape[0], "image 形状长度必须等于 depth长度"
    assert len(img.shape) == 2, "image 图像必须是二维的灰度图像，不能是其他格式的"
    assert path_texture_saved.endswith('csv') or len(path_texture_saved)==0, "保存路径必须是以csv格式结尾的"

    DEPTH_LIST = []
    TEXTURE_IMG = []

    # 根据图像大小，计算迭代次数
    ITER_NUM = math.ceil((img.shape[0] - windows) / step) + 1
    print('current step:{}, windows:{}, iter_num:{}, texture config:{}'.format(step, windows, ITER_NUM, texture_config))

    # 迭代获取图像纹理信息
    with tqdm(total=ITER_NUM) as pbar:
        pbar.set_description('Processing of Extraction image texture information:')
        for i in range(ITER_NUM):
            INDEX_START = i * step
            window_img = copy.deepcopy(img[INDEX_START:INDEX_START + windows, :])
            scale_factor = 256 / window_img.shape[-1]
            window_img = cv2.resize(
                window_img,
                None,  # 目标尺寸设为None
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_AREA  # 缩小首选插值
            )
            # print(window_img.shape, window_img.dtype, window_img.min(), window_img.max())

            # 动态图像的纹理信息提取
            texture_org, glcm_map, _, _ = get_glcm_Features(window_img,
                                                                      level=texture_config['level'],
                                                                      distance=texture_config['distance'],
                                                                      angles=texture_config['angles'])
            # 静态图像的纹理差信息提取
            texture_sub = get_glcm_sub(window_img, level=texture_config['level'],
                                            distance=texture_config['distance'])

            # 纹理差信息合并
            texture_all = np.concatenate((texture_org.ravel(), texture_sub.ravel()), axis=0)

            DEPTH_LIST.append(depth[INDEX_START + windows // 2])
            TEXTURE_IMG.append(texture_all)
            pbar.update(1)

    DEPTH = np.array(DEPTH_LIST).reshape(-1, 1)
    TEXTURE_IMG = np.array(TEXTURE_IMG)
    TEXTURE_LOGGING = np.concatenate((DEPTH, TEXTURE_IMG), axis=1)
    print('depth shape:{}, texture_img shape:{}, texture_logging:{}'.format(DEPTH.shape, TEXTURE_IMG.shape,
                                                                            TEXTURE_LOGGING.shape))
    if Curve_List:
        pass
    else:
        Curve_List = ('CON\tDIS\tHOM\tENG\tCOR\tASM\tENT\t'
                  'XY_CON\tXY_DIS\tXY_HOM\tXY_ENG\tXY_COR\tXY_ASM\tXY_ENT').split('\t')
    np.set_printoptions(suppress=True)
    target_df = pd.DataFrame(TEXTURE_LOGGING, columns=['DEPTH'] + Curve_List)
    # target_df.to_excel(Path_Folder + "\\{}_{}_{}_{}.xlsx".format(CHARTER, texture_target_string, windows, step), sheet_name='Sheet_Texture', index=False)

    if len(path_texture_saved) == 0:
        target_df.to_csv('texture_result.csv', index=False)
    else:
        target_df.to_csv(path_texture_saved, index=False)

    # return TEXTURE_LOGGING