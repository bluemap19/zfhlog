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
        img_data, depth_data = get_ele_data_from_path(i)
        CHAR_TEMP = i.split('/')[-1].split('\\')[-1].split('.txt')[0]
        print('{} image shape is:{}'.format(CHAR_TEMP, img_data.shape))
        EMPTY_PIC = np.zeros([128, 128])
        EMPTY_PIC.fill(255)

        if i.__contains__('stat'):
            img_data = cv2.resize(img_data, [128, 128], interpolation=cv2.INTER_NEAREST)
            IMG_LIST.append(255-img_data)

            texture_average_d1, glcm_map_average_d1, _, _ = get_glcm_Features(img_data, distance=[2])
            texture_average_d2, glcm_map_average_d2, _, _ = get_glcm_Features(img_data, distance=[4])
            texture_average_mean, glcm_map_average_mean, _, _ = get_glcm_Features(img_data, distance=[2, 4])
            IMG_LIST = [img_data, glcm_map_average_mean, EMPTY_PIC, glcm_map_average_d1, EMPTY_PIC, glcm_map_average_d2, EMPTY_PIC]
            show_Pic(IMG_LIST, pic_order='17', figure=(37, 5), pic_str=['STAT', 'GLCM_Mean', 'Radio_Mean', 'GLCM_D1', 'Radio1', 'GLCM_D2', 'Radio2'], title='')
            exit(0)
