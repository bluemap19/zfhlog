import os
import numpy as np
import pandas as pd
from src_data_process.cal_data_glcm_texture import cal_images_texture

def ele_stripes_delete(Pic, shape_target=(100, 8), delete_pix = 0):
    """
    空白条带删除函数，采用多退少补原则，多的话直接截取部分有用的，少的话，直接复制最后一段有用的
    :param Pic: 原始图像
    :param shape_target:目标图像形状大小，这个y方向一定保持一致，x方向一定小于原始输入图像
    :param delete_pix: 要删除的像素点大小
    :return: 返回删除指定像素点后的新图像
    """
    pic_new = np.zeros(shape_target, np.float64)

    if shape_target[0] != Pic.shape[0]:
        print('shape error, org shape is :{}, target shape is :{}'.format(Pic.shape, shape_target))

    for i in range(pic_new.shape[0]):
        # 查找符合条件的像素点坐标
        index_temp = np.where((Pic[i, :].ravel() >= delete_pix))

        # 符合条件的像素点个数
        len_fit = np.array(index_temp).shape[-1]

        # 刚好相等，不用修改
        if len_fit == shape_target[-1]:
            row_pix = Pic[i, index_temp]
        # 0个合格的元素，则进行零填充，将这一行全部都用0元素进行填充
        elif len_fit == 0:
            row_pix = 0
        # 多了，j进行相应的形变
        else:
            row_pix = Pic[i, index_temp]
            row_pix = np.resize(row_pix, (shape_target[-1], ))

        pic_new[i, :] = row_pix

    return pic_new


class data_FMI:
    def __init__(self, path_dyna='', path_stat='', well_name=''):
        self._table_2 = pd.DataFrame()
        self._well_name = well_name         # 井名
        self._data_dyna = pd.DataFrame()     # 动态数据体存放的是从文件中读取的原始dataframe
        self._data_stat = pd.DataFrame()    # 静态数据体
        self._data_depth_stat = pd.DataFrame()  # 静态深度数据体
        self._data_depth_dyna = pd.DataFrame()  # 动态深度数据体
        self._resolution = 0.0025               # 分辨率
        self._texture_dyna = pd.DataFrame()     # 动态纹理
        self._texture_stat = pd.DataFrame()     # 静态纹理
        self._path_dyna = path_dyna             # 动态文件 路径
        self._path_stat = path_stat             # 静态文件 路径

        if not os.path.isfile(path_dyna):
            print('\033[31m' + 'File not found or does not exist:{}'.format(path_dyna) + '\033[0m')
        if not os.path.isfile(path_stat):
            print('\033[31m' + 'File not found or does not exist:{}'.format(path_stat) + '\033[0m')

    def read_data(self, file_path=''):
        if os.path.isfile(file_path):
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, index_col=0)
                data = df.values
            elif file_path.endswith('.txt'):
                df = np.loadtxt(file_path, delimiter='\t', skiprows=8)
                data = np.array(df)
            else:
                print('\033[31m' + 'Error file extension as:{}'.format(file_path.split('\\')[-1]) + '\033[0m')

            return data
        else:
            print('\033[31m' + 'File path does not exist as:{}'.format(file_path) + '\033[0m')
            return

    def get_data(self, mode='DYNA'):
        if mode == 'DYNA':
            if self._data_dyna.empty:
                data_dyna = self.read_data(self._path_dyna)
                self._data_dyna = data_dyna[:, 1:]
                self._data_depth_dyna = data_dyna[:, 0]
            return {'DYNA':self._data_dyna, 'DEPTH_DYNA':self._data_depth_dyna}
        elif mode == 'STAT':
            if self._data_stat.empty:
                data_stat = self.read_data(self._path_stat)
                self._data_stat = data_stat[:, 1:]
                self._data_depth_stat = data_stat[:, 0]
            return {'STAT':self._data_stat, 'DEPTH_STAT':self._data_depth_stat}
        elif mode == 'ALL':
            if self._data_dyna.size == 0:
                data_dyna = self.read_data(self._path_dyna)
                self._data_dyna = data_dyna[:, 1:]
                self._data_depth_dyna = data_dyna[:, 0]
            if self._data_stat.size == 0:
                data_stat = self.read_data(self._path_stat)
                self._data_stat = data_stat[:, 1:]
                self._data_depth_stat = data_stat[:, 0]
            if self._data_dyna.shape != self._data_stat.shape:
                print('\033[31m' + 'Data shape mismatch as dyna-stat:{} - {}'.format(self._data_dyna.shape, self._data_stat.shape) + '\033[0m')
            return {'DYNA':self._data_dyna, 'STAT':self._data_stat, 'DEPTH_DYNA':self._data_depth_dyna, 'DEPTH_STAT':self._data_depth_stat}
        else:
            print('ERROR MODE')
            exit(0)

    def ele_stripes_delete(self, mode='ALL', delete_pix=0, width_ratio=0.8):
        if mode == 'ALL':
            self._data_dyna = ele_stripes_delete(self._data_dyna,
                                               shape_target=(self._data_dyna.shape[0], int(self._data_dyna.shape[1]*width_ratio)),
                                               delete_pix=delete_pix)
            self._data_stat = ele_stripes_delete(self._data_stat,
                                               shape_target=(self._data_stat.shape[0], int(self._data_stat.shape[1] * width_ratio)),
                                               delete_pix=delete_pix)
        elif mode == 'STAT':
            self._data_stat = ele_stripes_delete(self._data_stat,
                                               shape_target=(self._data_stat.shape[0], int(self._data_stat.shape[1] * width_ratio)),
                                               delete_pix=delete_pix)
        elif mode == 'DYNA':
            self._data_dyna = ele_stripes_delete(self._data_dyna,
                                               shape_target=(self._data_dyna.shape[0], int(self._data_dyna.shape[1]*width_ratio)),
                                               delete_pix=delete_pix)
        else:
            print('ERROR MODE as :{}'.format(mode))
            exit(0)

    def get_texture(self, mode='ALL', texture_config={'level':16, 'distance':[2,4], 'angles':[0, np.pi/4, np.pi/2, np.pi*3/4], 'windows_length':80, 'windows_step':5}, save_texture=''):
        self.get_data(mode)
        img_dyna = self._data_dyna
        img_stat = self._data_stat

        # 只使用静态的深度，动态的深度不进行改变
        depth_img = self._data_depth_stat
        print('current path_saved is {}'.format(save_texture))

        if mode == 'DYNA':
            print('\033[31m' + 'DYNA MODE FUNCTION IS EMPTY' + '\033[0m')
        elif mode == 'STAT':
            print('\033[31m' + 'STAT MODE FUNCTION IS EMPTY' + '\033[0m')
        elif mode == 'ALL':
            TEXTURE_LOGGING = cal_images_texture(imgs=[img_dyna, img_stat], depth=depth_img, windows=texture_config['windows_length'], step=texture_config['windows_step'],
                               texture_config = texture_config,
                               path_texture_saved = save_texture,
                               texture_headers=['CON_MEAN_DYNA', 'DIS_MEAN_DYNA', 'HOM_MEAN_DYNA', 'ENG_MEAN_DYNA', 'COR_MEAN_DYNA', 'ASM_MEAN_DYNA', 'ENT_MEAN_DYNA',
                                                'CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA', 'ENG_SUB_DYNA', 'COR_SUB_DYNA', 'ASM_SUB_DYNA', 'ENT_SUB_DYNA',
                                                'CON_X_DYNA', 'DIS_X_DYNA', 'HOM_X_DYNA', 'ENG_X_DYNA', 'COR_X_DYNA', 'ASM_X_DYNA', 'ENT_X_DYNA',
                                                'CON_Y_DYNA', 'DIS_Y_DYNA', 'HOM_Y_DYNA', 'ENG_Y_DYNA', 'COR_Y_DYNA', 'ASM_Y_DYNA', 'ENT_Y_DYNA',
                                                'CON_MEAN_STAT', 'DIS_MEAN_STAT', 'HOM_MEAN_STAT', 'ENG_MEAN_STAT', 'COR_MEAN_STAT', 'ASM_MEAN_STAT', 'ENT_MEAN_STAT',
                                                'CON_SUB_STAT', 'DIS_SUB_STAT', 'HOM_SUB_STAT', 'ENG_SUB_STAT', 'COR_SUB_STAT', 'ASM_SUB_STAT', 'ENT_SUB_STAT',
                                                'CON_X_STAT', 'DIS_X_STAT', 'HOM_X_STAT', 'ENG_X_STAT', 'COR_X_STAT', 'ASM_X_STAT', 'ENT_X_STAT',
                                                'CON_Y_STAT', 'DIS_Y_STAT', 'HOM_Y_STAT', 'ENG_Y_STAT', 'COR_Y_STAT', 'ASM_Y_STAT', 'ENT_Y_STAT',
                                                ])
            return TEXTURE_LOGGING
        else:
            print('ERROR MODE as :{}'.format(mode))
            exit(0)

if __name__ == '__main__':
    # test_FMI = data_FMI(
    #     # path_dyna=r'F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\城96_FMI_Main 1700-2120_dyna.txt',
    #     # path_stat=r'F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\城96_FMI_Main 1700-2120_stat.txt',
    #     path_dyna = r'F:\桌面\算法测试-长庆数据收集\logging_CSV\SIMU4\dyna_simu.txt',
    #     path_stat = r'F:\桌面\算法测试-长庆数据收集\logging_CSV\SIMU4\stat_simu.txt',
    #     well_name='SIMU4')
    #
    # DYNA = test_FMI.get_data(mode='DYNA')
    # print(DYNA['DYNA'].shape, DYNA['DEPTH'].shape,)
    # STAT = test_FMI.get_data(mode='STAT')
    # print(STAT['STAT'].shape, STAT['DEPTH'].shape,)
    #
    # test_FMI.get_texture(mode='ALL', save_texture=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\SIMU4\simu_texture_file.csv')

    test_FMI = data_FMI(
        # path_dyna=r'F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\城96_FMI_Main 1700-2120_dyna.txt',
        # path_stat=r'F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\城96_FMI_Main 1700-2120_stat.txt',
        path_dyna=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\FY1-15\FY1-15_dyna.txt',
        path_stat=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\FY1-15\FY1-15_stat.txt',
        well_name='FY1-15')

    DYNA = test_FMI.get_data(mode='DYNA')
    print(DYNA['DYNA'].shape, DYNA['DEPTH_DYNA'].shape, )
    STAT = test_FMI.get_data(mode='STAT')
    print(STAT['STAT'].shape, STAT['DEPTH_STAT'].shape, )

    test_FMI.ele_stripes_delete(mode='ALL')
    test_FMI.get_texture(mode='ALL',
                         save_texture=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\FY1-15\FY1-15_texture.csv')
