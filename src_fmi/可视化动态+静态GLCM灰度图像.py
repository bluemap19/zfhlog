import cv2
import numpy as np

from src_file_op.dir_operation import get_all_file_paths
from src_fmi.fmi_data_read import get_ele_data_from_path
from src_fmi.image_operation import get_glcm_sub, show_Pic, get_glcm_xy, get_glcm_Features

if __name__ == '__main__':
    path_folder = r'C:\Users\Administrator\Desktop\纹理案例'
    # path_folder = r'C:\Users\ZFH\Desktop\纹理案例'
    list_path_texture = get_all_file_paths(path_folder)


    IMG_LIST = []
    GLCM_SUB_LIST= []
    for i in list_path_texture:
        EMPTY_PIC = np.zeros([128, 128])
        EMPTY_PIC.fill(255)

        if i.__contains__('stat'):
            img_stat, depth_data = get_ele_data_from_path(i)
            path_dyna = i.replace('stat', 'dyna')
            img_dyna, depth_data = get_ele_data_from_path(path_dyna)
            CHAR_TEMP = i.split('/')[-1].split('\\')[-1].split('.txt')[0]
            print('{} image shape is:{}'.format(CHAR_TEMP, img_stat.shape))
            img_stat = cv2.resize(img_stat, [128, 128], interpolation=cv2.INTER_NEAREST)
            img_dyna = cv2.resize(img_dyna, [128, 128], interpolation=cv2.INTER_NEAREST)

            distance = [2, 4]
            texture_sub_stat = get_glcm_sub(img_stat, distance=distance)
            texture_average_stat, glcm_map_average_stat, _, _ = get_glcm_Features(img_stat, distance=distance)
            texture_sub_dyna = get_glcm_sub(img_dyna, distance=distance)
            texture_average_dyna, glcm_map_average_dyna, _, _ = get_glcm_Features(img_dyna, distance=distance)

            # print(texture_sub.shape, texture_x.shape, texture_y.shape, glcm_map_x.shape, glcm_map_y.shape, texture_average.shape, glcm_map_average.shape)

            feature_all = [texture_average_stat.ravel(), texture_sub_stat.ravel(), texture_average_dyna.ravel(), texture_sub_dyna.ravel()]
            print(np.array(feature_all))
            IMG_LIST = [255-img_stat, glcm_map_average_stat, EMPTY_PIC, EMPTY_PIC, 255-img_dyna, glcm_map_average_dyna, EMPTY_PIC, EMPTY_PIC]
            show_Pic(IMG_LIST, pic_order='18', figure=(42, 5), pic_str=['Img', 'GLCM_Mean', 'Radar', 'GLCM_X', 'Radar', 'GLCM_Y', 'Radar', 'Radar'], title='')
            exit(0)
