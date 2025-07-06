import numpy as np
import xlwt
import cv2

def get_ele_data_from_path(filepath_FMI =r'D:\Data\target\107S\YS107_FMI_BorEID_FA.txt', depth = [-1.0, -1.0]):
    img_data = np.array([])
    depth_data = np.array([])
    if (filepath_FMI.endswith('.png') | filepath_FMI.endswith('.jpg')) :
        img_data = cv2.imread(filepath_FMI, cv2.IMREAD_GRAYSCALE)
        if img_data.shape[0] == 0:
            print('image in data is empty:{}'.format(filepath_FMI))
            exit(0)
        depth_data = np.zeros((img_data.shape[0], 1))
        if (len(filepath_FMI.split('_')) < 4)|(filepath_FMI.__contains__('texture_set')):
            print(filepath_FMI)
            print('file name error, donnot contain depth information:{}'.format(filepath_FMI))
            depth_data = np.linspace(100,104,img_data.shape[0]).reshape((img_data.shape[0], 1))
        else:
            # print(strname.split('\\')[-1].split('_')[1], strname.split('_')[1], strname.split('\\')[-1])
            startdep = float(filepath_FMI.split('\\')[-1].split('/')[-1].split('_')[-2])
            enddep = float(filepath_FMI.split('\\')[-1].split('/')[-1].split('_')[-1].split('.png')[0].split('.jpg')[0])
            # print(startdep, enddep)
            Step = (enddep-startdep)/img_data.shape[0]
            for i in range(img_data.shape[0]):
                depth_data[i] = i*Step + startdep
    elif (filepath_FMI.endswith('.txt')):
        AllData = []
        with open(filepath_FMI, 'r', encoding='GBK') as file:
            lines = file.readlines()
            for (i, line) in enumerate(lines):
                if i > 7:
                    values = list(filter(None, line.strip().split()))       # 过滤空值
                    values = [float(x) for x in values]
                    AllData.append(values)
        AllData = np.array(AllData)


        # AllData = np.loadtxt(filepath_FMI, delimiter=' ', skiprows=9, encoding='GBK')
        if AllData.shape[0] ==0 :
            print('text in data is empty:{}'.format(filepath_FMI))
            exit(0)
        img_data = AllData[:, 1:]
        depth_data = AllData[:, 0].reshape((-1, 1))
        # startdep = depth_data[0, 0]
        # enddep = depth_data[0, 0]
        # print(AllData.shape)
        # print('from Txt get ELE Data')
    elif (filepath_FMI.endswith('.csv')):
        print('CsvData read txt')

    # 根据深度 进行 图像以及深度数据的截取
    if (depth[0] < 0) & (depth[1] < 0):
        pass
    elif ((depth[0] > 0) & (depth[0] < depth_data[0, 0])) | ((depth[0] > 0) & (depth[0] > depth_data[-1, 0])):
        print('depth out of range, target depth is:{}, current file is:[{}, {}]'.format(depth, depth_data[0, 0], depth_data[-1, 0]))
        exit(0)
    elif ((depth[1] > 0) & (depth[1] < depth_data[0, 0])) | ((depth[1] > 0) & (depth[1] > depth_data[-1, 0])):
        print('depth out of range, target depth is:{}, current file is:[{}, {}]'.format(depth, depth_data[0, 0], depth_data[-1, 0]))
        exit(0)
    else:
        Step = (depth_data[-1] - depth_data[0])/depth_data.shape[0]
        # if (depth[0]<depth_data[0,0])|(depth[0]>depth_data[-1,0])|(depth[1]<depth_data[0,0])|(depth[1]>depth_data[-1,0]):
        #     print('depth error, depth from:{} to {},target depth is :{}'.format(depth_data[0,0],depth_data[-1,0], depth))
        #     exit(0)

        start_index = 0
        end_index = 0

        if depth[0] <= 0:
            start_index = 0
        if depth[1] <= 0:
            end_index = img_data.shape[0]-1
        for i in range(depth_data.shape[0]):
            if abs((depth_data[i]-depth[0])) <= Step/2:
                start_index = i
            if abs((depth_data[i]-depth[1])) <= Step/2:
                end_index = i

        # print(step, depth_data[0], depth_data[-1], depth, start_index, end_index)
        return img_data[start_index:end_index, :], depth_data[start_index:end_index, :]

    return img_data, depth_data



def get_random_ele_data():
    # path_test1 = r'D:\Data\target_stage1_small\guan17-11_342_3751.0025_3751.6675_dyna.png'
    # path_test2 = r'D:\Data\target_stage1_small\guan17-11_342_3751.0025_3751.6675_stat.png'

    path_test1 = r'D:\Data\target_stage1_small\LG7-11_79_5087.4775_5088.1100_dyna.png'
    path_test2 = r'D:\Data\target_stage1_small\LG7-11_79_5087.4775_5088.1100_stat.png'

    # path_test1 = r'D:\Data\target_stage1_small\lg701-h1_476_5403.0020_5403.6670_dyna.png'
    # path_test2 = r'D:\Data\target_stage1_small\lg701-h1_476_5403.0020_5403.6670_stat.png'

    # path_test1 = r'D:\Data\pic_seg_choices\data_hole\hole_paper_show\LG701-H1_250_5324.5020_5325.1645_dyna.png'
    # path_test2 = r'D:\Data\pic_seg_choices\data_hole\hole_paper_show\LG701-H1_250_5324.5020_5325.1645_stat.png'

    # path_test1 = r'D:\Data\target_stage1_small\guan17-11_372_3766.0025_3766.6350_dyna.png'
    # path_test2 = r'D:\Data\target_stage1_small\guan17-11_372_3766.0025_3766.6350_stat.png'

    # path_test1 = r'D:\Data\target_stage1_small\lg701_222_5224.5000_5225.1450_dyna.png'
    # path_test2 = r'D:\Data\target_stage1_small\lg701_222_5224.5000_5225.1450_stat.png'

    data_img_dyna, data_depth = get_ele_data_from_path(path_test1)
    data_img_stat, data_depth = get_ele_data_from_path(path_test2)

    return data_img_dyna, data_img_stat, data_depth

