import copy
import math
import pandas as pd
from scipy.signal import find_peaks
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
# 对聚类结果进行画图
from pylab import mpl
# 设置中文显示字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False


def find_mode(arr: np.array) -> np.float64:
    """返回数组中的所有众数"""
    # 展平数组并计算唯一值与频次
    flat_arr = arr.ravel()
    unique_vals, counts = np.unique(flat_arr, return_counts=True)

    # 找出最高频次对应的所有值
    max_count = np.max(counts)
    modes = unique_vals[counts == max_count]
    return modes[0]


def get_resolution_by_depth(depth_array: np.array) -> np.float64:
    depth_array = depth_array.ravel()
    depth_diff = np.diff(depth_array)
    depth_resolution = find_mode(depth_diff)
    return depth_resolution


# 根据深度获取曲线的index，只适合连续测井深度数据
def get_index_by_depth(logging, depth):
    """
    根据深度获取曲线的index，只适合连续测井深度数据
    :param n_logging: depth information, don't include other logging_data data
    :param depth: folat:target depth need to get index
    :return: int:index to taget depth
    """
    depth_t = logging.ravel()
    index_temp = int(logging.shape[0] * (depth-depth_t[0])/(depth_t[-1]-depth_t[0]) - 5)
    while((depth_t[index_temp]<depth) & (index_temp<depth_t.shape[0])):
        index_temp += 1

    return index_temp


# 这个只能对连续深度的数据起作用，不是连续数据的话，不能用这个
# 根据深度，获取指定数据的 曲线信息 并不返回深度信息
def get_data_by_depth(dep_temp, curve):
    """
    这个只能对连续深度的数据起作用，不是连续数据的话，不能用这个
    根据深度，获取指定数据的 曲线信息 并不返回深度信息
    :param dep_temp:
    :param curve:
    :return:
    """
    dep_start = curve[0][0]
    dep_end = curve[-1][0]

    if dep_temp < (dep_start-0.01):
        print('error dep:{}，depth information：{}'.format(dep_temp, curve[:10]))
        exit(0)

    index_start = max(int((dep_temp-dep_start)/(dep_end-dep_start)*curve.shape[0]) - 10, 0)

    index_loc = 0
    for i in range(curve.shape[0] - index_start):
        if curve[index_start+i][0] > dep_start:
            index_loc = index_start+i
            break

    if index_loc==0:
        return curve[index_loc, 1:]
    elif index_loc==curve.shape[0]-1:
        return curve[index_loc, 1:]
    else:
        return (curve[index_loc, 1:]+curve[index_loc-1, 1:])/2


# 根据深度list:[dep_start, dep_end]获取指定范围内的测井数据
def get_data_by_depths(logging_data, depth):
    """
    根据深度list:[dep_start, dep_end]获取指定范围内的测井数据
    :param data_normal_curve: ndarray:depth*number
    :param depth: list:[dep_start, dep_end]
    :return: ndarray:depth*number
    """
    if (depth[0] < logging_data[0][0]) | (depth[1] > logging_data[-1][0]) | (depth[1] < depth[0]):
        print('depths error :{}, {}->{}'.format(depth, logging_data[0][0], logging_data[-1][0]))
        exit(0)

    index_start = get_index_by_depth(logging_data[:, 0], depth[0])
    index_end = get_index_by_depth(logging_data[:, 0], depth[1])

    return logging_data[index_start:index_end, :]



# 数据合并，把几种带有深度的 测井信息进行 通过深度的 数据合并导出，按统一深度把数据合并，方便数据统一后进行无监督聚类
# curve_zip为 曲线集合list，曲线必须带与深度，且深度信息放在第一列
def data_combine(curve_zip, depth=[-1, 0], step='MIN'):
    """
    数据合并，把几种带有深度的 测井信息进行 通过深度的 数据合并导出，按统一深度把数据合并，方便数据统一后进行无监督聚类
    curve_zip为 曲线集合list，曲线必须带与深度，且深度信息放在第一列，并且深度信息必须连续且等差
    :param curve_zip: list:include different logging_data curve
    :param depth: [depth_start. depth_end]:target depth to get
    :param step: step config:'MIN','MAX' or float
    :return: ndarray: target logging_data curve combined in current depth
    """

    # 计算最小的step， 当作整体的step, 计算最小的end_dep， 当作整体的end_dep, 计算最大的start_dep， 当作整体的start_dep
    step_final = 0
    # 总的曲线条数
    curve_num = 0

    # 遍历曲线集合，统计曲线条数，统计曲线的顶深底深，用来新建合并后的曲线信息
    depth_start_list = []
    depth_end_list = []
    curve_length_list = []
    for i in range(len(curve_zip)):
        # 计算 该数据集的 曲线 step
        step_n = (curve_zip[i][-1, 0] - curve_zip[i][0, 0]) / curve_zip[i].shape[0]
        # 计算 该数据集的 曲线条数
        curve_num += (curve_zip[i].shape[1] - 1)
        # 设置生成曲线的分辨率大小
        if i == 0:
            step_final = step_n
        else:
            if step == 'MIN':
                step_final = min(step_n, step_final)
            elif step == 'MAX':
                step_final = max(step_n, step_final)
            else:
                step_final = step

        depth_start_list.append(curve_zip[i][0, 0])
        depth_end_list.append(curve_zip[i][-1, 0])
        curve_length_list.append(curve_zip[i].shape[0])

    # 统计各自的顶底信息，并选取最大的顶深start_depth，最小的底深end_depth
    start_depth = np.max(np.array(depth_start_list))
    end_depth = np.min(np.array(depth_end_list))

    # 判断生成曲线的深度信息是否合理
    if (start_depth > end_depth):
        print('data_combine:error target depth:{}-->{}'.format(start_depth, end_depth))
        exit(0)

    if (np.var(np.array(depth_start_list))<1) & ((np.var(np.array(depth_end_list))<1)) & ((np.var(np.array(curve_length_list))<1)):
        print('data_combine:combine  directly....')
        curve_final = np.array([])
        for i in range(len(curve_zip)):
            if i==0:
                curve_final = curve_zip[i]
            else:
                curve_final = np.hstack((curve_final, curve_zip[i]))

        return curve_final

    # 如果深度初始化了，进行深度初始化的赋值
    if (depth[0] >= 0) & (depth[1] >= 0):
        if (min(depth) < start_depth) | (max(depth) > end_depth):
            print('data_combine:dep pre value error:合理区间{}， 实际区间{}'.format([start_depth, end_depth], depth))
            exit(0)
        start_depth = max(min(depth), start_depth)
        end_depth = min(max(depth), end_depth)

    step_num = int((end_depth - start_depth)/step_final)
    all_curve_np = np.zeros((step_num, 1))

    index_curve = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    values = [[], [], [], [], [], [], [], [], [], []]
    for i in range(step_num):
        temp_dep = start_depth + step_final * i
        all_curve_np[i][0] = temp_dep
        for j in range(len(curve_zip)):
            while index_curve[j] < curve_zip[j].shape[0]-1:
                if (temp_dep >= curve_zip[j][index_curve[j], 0]) & (temp_dep <= curve_zip[j][index_curve[j]+1, 0]) | (abs(temp_dep-curve_zip[j][index_curve[j], 0]) <= step_final/2):
                    values[j].append(curve_zip[j][index_curve[j], 1:])
                    # print(index_curve[j])
                    index_curve[j] -= 1
                    break
                else:
                    index_curve[j] += 1

    for i in range(len(curve_zip)):
        all_curve_np = np.hstack((all_curve_np, np.array(values[i])))

    return all_curve_np


# 数据合并，把几种带有深度的 测井信息进行 通过深度的 数据合并导出，按统一深度把数据合并，方便数据统一后进行无监督聚类
# curve_zip为 曲线集合list，曲线必须带与深度，且深度信息放在第一列
def data_combine_new(curve_zip, depth=[-1, 0], step='MIN', replace=np.nan):
    # 计算最小的step， 当作整体的step, 计算最大的end_dep， 当作整体的end_dep, 计算最小的start_dep， 当作整体的start_dep
    step_final = 0
    # 总的曲线条数
    curve_num = 0

    # 遍历曲线集合，统计曲线条数，统计曲线的顶深底深，用来新建合并后的曲线信息
    depth_start_list = []
    depth_end_list = []
    curve_length_list = []
    curve_resolution_list = []
    for i in range(len(curve_zip)):
        # 计算 该数据集的 曲线 step
        step_n = find_mode(np.diff(curve_zip[i][:, 0]))
        curve_resolution_list.append(step_n)
        # 计算 该数据集的 曲线条数
        curve_num += (curve_zip[i].shape[1] - 1)
        depth_start_list.append(curve_zip[i][0, 0])
        depth_end_list.append(curve_zip[i][-1, 0])
        curve_length_list.append(curve_zip[i].shape[0])

    # 设置生成曲线的分辨率大小
    if step == 'MIN':
        step_final = np.min(curve_resolution_list)
    elif step == 'MAX':
        step_final = np.max(curve_resolution_list)
    else:
        step_final = step

    print('curve resolution is:{}---->{}'.format(curve_resolution_list, step_final))

    # 统计各自的顶底信息，并选取最小的顶深start_depth，最大的底深end_depth
    start_depth = np.min(np.array(depth_start_list))
    end_depth = np.max(np.array(depth_end_list))

    # 判断生成曲线的深度信息是否合理
    if (start_depth > end_depth):
        print('depth_start_list : {},\t depth_end_list:{}'.format(depth_start_list, depth_end_list))
        print('data combine error target depth:{}-->{}'.format(start_depth, end_depth))
        exit(0)

    if (np.var(np.array(depth_start_list))<1) and ((np.var(np.array(depth_end_list))<1)) and ((np.var(np.array(curve_length_list))<1)):
        print('data combine directly and return ....')
        curve_final = np.array([])
        for i in range(len(curve_zip)):
            if i==0:
                curve_final = curve_zip[i]
            else:
                curve_final = np.hstack((curve_final, curve_zip[i]))
        return curve_final

    # 如果深度初始化了，进行深度初始化的赋值
    if (depth[0] >= 0) and (depth[1] >= 0):
        if ((min(depth) < start_depth) or (max(depth) > end_depth)):
            print('data_combine:dep pre value error:合理区间{}， 实际区间{}'.format([start_depth, end_depth], depth))
            exit(0)
        start_depth = max(min(depth), start_depth)
        end_depth = min(max(depth), end_depth)

    step_num = int((end_depth - start_depth)/step_final)

    # 记录每一个 测井数据 开始进行遍历迭代的起始位置
    index_curve = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    data_all = []
    for i in range(step_num):
        dep_temp = start_depth + step_final * i
        curve_responce_temp = [dep_temp]
        for j in range(len(curve_zip)):
            curve_data = curve_zip[j]
            depth_start_temp = curve_data[0, 0]
            depth_end_temp = curve_data[-1, 0]
            if ((dep_temp < depth_start_temp) or (dep_temp > depth_end_temp)):
                # print('t1', dep_temp, curve_data[0, 0], curve_data[-1, 0])
                data_length = curve_data.shape[1]-1
                data_temp = list(np.full((data_length, ), replace))
                curve_responce_temp = curve_responce_temp + data_temp
                continue
            else:
                # print('t2', dep_temp, curve_data[0, 0], curve_data[-1, 0])
                flag = True
                for k in range(curve_data.shape[0] - index_curve[j]):
                    depth_iter_temp = curve_data[k+index_curve[j], 0]
                    if (abs(dep_temp-depth_iter_temp) <= (curve_resolution_list[j]/2+0.001)):
                        # print('t21', dep_temp, depth_iter_temp)
                        curve_responce_temp = curve_responce_temp + list(curve_data[k+index_curve[j], 1:])
                        index_curve[j] = max(index_curve[j] + k - 1, 0)
                        flag = False
                        break
                    if depth_iter_temp > dep_temp:
                        # print('t22', dep_temp, depth_iter_temp, k, index_curve[j])
                        index_curve[j] = max(index_curve[j] + k - 2, 0)
                        break
                if flag:
                    data_length = curve_data.shape[1]-1
                    data_temp = list(np.full((data_length, ), replace))
                    curve_responce_temp = curve_responce_temp + data_temp

        # print('s1', len(curve_responce_temp), curve_responce_temp)
        data_all.append(np.array(curve_responce_temp))
    all_curve_np = np.array(data_all)

    return all_curve_np




def data_combine_table3col(data_main, table_vice, drop=True):
    """
    数据合并
    :param data_main:主数据，为n行m列的测井数据，n*m 必须连续且等差
    :param table_vice: 表格数据，一般为解释结论n*3 必须遵循 [depth_start, depth_end, Type]
    :param drop:是否抛弃 解释结论表格没有的数据，True的话，就舍弃掉，数字的话，就用那个数字代替，False的话，就用-1代替
    :return: numpy.ndarray:合并后的测井数据
    """
    # 表格信息的审查，确保表格信息的合理可用
    for i in range(table_vice.shape[0]):
        dep_s, dep_e = table_vice[i, :2]
        if dep_s > dep_e:
            print('Error dep configuration:{}'.format(table_vice[i]))
            exit(0)
        # if i>0:
        #     if dep_s < table_vice[i-1, 1]:
        #         print('Error dep configuration:{}-->{}'.format(table_vice[i-1], table_vice[i]))
        #         exit(0)

    data_final = []

    for i in range(data_main.shape[0]):
        depth = data_main[i, 0]
        flag = True
        for j in range(table_vice.shape[0]):
            dep_s, dep_e = table_vice[j, :2]
            type = table_vice[j, -1]
            if (depth >= dep_s) & (depth <= dep_e):
                data_final.append(np.append(data_main[i].ravel(), type))
                flag = False

        if flag == True:
            if drop == True:
                pass
            else:
                data_final.append(np.append(data_main[i].ravel(), np.nan))

    return np.array(data_final)


def data_combine_table2col(data_main, table_vice, drop=True):
    """
    数据合并
    :param data_main:主数据，为n行m列的测井数据，n*m 必须连续且等差
    :param table_vice: 表格数据，一般为解释结论n*2 必须遵循 [depth, Type]   Type必须为数字，否则会报错。
    :param drop:是否抛弃 解释结论表格没有的数据，True的话，就舍弃掉，数字的话，就用那个数字代替，False的话，就用-1代替
    :return: numpy.ndarray:合并后的测井数据
    """
    # 表格信息的审查，确保表格信息的合理可用
    for i in range(table_vice.shape[0]):
        if i>0:
            if table_vice[i-1, 0] > table_vice[i, 0]:
                print('Error dep configuration:{}-->{}'.format(table_vice[i-1], table_vice[i]))
                exit(0)

    depth_table_resolution = get_resolution_by_depth(table_vice[:, 0])
    depth_resolution = get_resolution_by_depth(data_main[:, 0])
    # table_vice[:, 0].astype(np.float64)

    data_final = []
    index_table_continue = 0
    combined_successfully = 0
    for i in range(data_main.shape[0]):
        depth = data_main[i, 0]
        flag = True
        for j in range(table_vice.shape[0]-index_table_continue):
            dep_table = float(table_vice[j+index_table_continue, 0])
            type = table_vice[j+index_table_continue, -1]
            if abs(depth-dep_table) < (depth_table_resolution/2+0.001):
                data_final.append(np.append(data_main[i].ravel(), type))
                index_table_continue = j+index_table_continue
                combined_successfully += 1
                flag = False
                # index_table_continue = np.max(0, index_table_continue-1)
                break

        if flag == True:
            if drop == True:
                pass
            else:
                data_final.append(np.append(data_main[i].ravel(), np.nan))

    data_final = np.array(data_final)
    # data_final[:, 0].astype(np.float64)
    print('data logging:[{:.2f},{:.2f}] ＋ data table:[{},{}] --> data combined :{},[{}, {}]， '
          'with resolution {:.4f}, success count:{}'.format(data_main[0, 0], data_main[-1, 0], table_vice[0, 0], table_vice[-1, 0],
                                                        data_final.shape, data_final[0, 0], data_final[-1, 0], depth_resolution, combined_successfully))
    # print('data logging_data:[{},{}] ++++ data table:[{},{}] ----> data_combined :{},[{}, {}]\n'
    #       'with resolution {}, success count:{}'.format(data_main[0, 0], data_main[-1, 0], table_vice[0, 0],
    #                                                         table_vice[-1, 0],
    #                                                         data_final.shape, data_final[0, 0], data_final[-1, 0],
    #                                                         depth_resolution, combined_successfully))

    # print('success combine table info:{}'.format(combined_successfully))
    return data_final



# 获取指定曲线 指定比例 指定范围内的 最大值、最小值
def get_extreme_value_by_ratio(curve=np.array([]), ratio_c=0.2, range_c=[-99, 9999]):
    """
    获取指定曲线 指定比例 指定范围内的 最大值、最小值
    :param curve:
    :param ratio_c:
    :param range_c:
    :return:
    """
    max_list = []
    min_list = []

    d_t = curve.ravel()

    num_ratio = int(d_t.shape[0] * ratio_c)
    # print(d_t.shape)
    for i in range(d_t.shape[0]):
        if (d_t[i]>range_c[0]) & (d_t[i]<range_c[1]):
            if len(max_list) < num_ratio:
                # 初始化 最大列表、最小列表
                max_list.append(d_t[i])
                min_list.append(d_t[i])
            else:
                max_from_min_list = max(min_list)
                min_from_max_list = min(max_list)
                if d_t[i] > min_from_max_list:
                    index_min = max_list.index(min_from_max_list)
                    max_list[index_min] = d_t[i]
                if d_t[i] < max_from_min_list:
                    index_max = min_list.index(max_from_min_list)
                    min_list[index_max] = d_t[i]
        else:
            # 在阈值之外，直接跳过，进行下一个的迭代
            continue


    max_list = np.array(max_list)
    min_list = np.array(min_list)
    # print('max_list shape is :{}'.format(max_list.shape))
    if (max_list.size >= 0) & (min_list.size >= 0):
        max_mean = np.mean(max_list)
        min_mean = np.mean(min_list)
    else:
        print('error max or min list:{} or {}'.format(max_list, min_list))
    return max_mean, min_mean




# 整体范围上的数据标准化
def data_normalized(logging_data, max_ratio=0.1, logging_range=[-99, 9999], DEPTH_USE=False):
    """
    整体数据范围特征上的数据标准化
    :param logging_data:
    :param max_ratio:
    :param logging_range:
    :param DEPTH_USE:
    :return:
    """
    if DEPTH_USE:
        logging_data_N = copy.deepcopy(logging_data[:, 1:])
    else:
        logging_data_N = copy.deepcopy(logging_data)

    extreme_list = []
    # print('logging_data_N shape is {}'.format(logging_data_N.shape))
    for j in range(logging_data_N.shape[1]):
        max_F, min_F = get_extreme_value_by_ratio(logging_data_N[:, j], ratio_c=max_ratio, range_c=logging_range)
        if (max_F==None) | (min_F==None):
            logging_data_N[:, j] = 0
            extreme_list.append([max_F, min_F])
        else:
            logging_data_N[:, j] = (logging_data_N[:, j]-min_F)/(max_F-min_F+0.001)
            extreme_list.append([max_F, min_F])
        logging_data_N[:, j][logging_data_N[:, j]<0] = 0
        logging_data_N[:, j][logging_data_N[:, j]>1] = 1

    if DEPTH_USE:
        logging_data_N[(logging_data[:, 1:] < logging_range[0]) | (logging_data[:, 1:] > logging_range[1])] = np.nan
        logging_data_N = np.hstack((logging_data[:, 0].reshape(-1, 1), logging_data_N))
    else:
        logging_data_N[(logging_data >= logging_range[0]) & (logging_data < logging_range[1])] = np.nan

    return logging_data_N, extreme_list


def data_normalized_manually(logging_data, limit=[[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]], DEPTH_USE=False):
    """
    整体数据范围特征上的数据标准化
    :param logging_data:
    :param max_ratio:
    :param logging_range:
    :param DEPTH_USE:
    :return:
    """
    if DEPTH_USE:
        logging_data_N = copy.deepcopy(logging_data[:, 1:])
    else:
        logging_data_N = copy.deepcopy(logging_data)

    extreme_list = limit
    for j in range(logging_data_N.shape[1]):
        max_F, min_F = extreme_list[j]
        logging_data_N[:, j] = (logging_data_N[:, j]-min_F)/(max_F-min_F)

    if DEPTH_USE:
        logging_data_N = np.hstack((logging_data[:, 0].reshape(-1, 1), logging_data_N))

    return logging_data_N


# 基于局部特征的数据标准化
def data_normalized_locally(logging_data, windows_length=400, max_ratio=0.1, logging_range=[-999, 9999], DEPTH_USE=False):
    """
    基于局部特征的数据标准化
    :param logging_data:
    :param windows_length:
    :param max_ratio:
    :param logging_range:
    :param DEPTH_USE:
    :return:
    """
    if DEPTH_USE:
        logging_data_N = copy.deepcopy(logging_data[:, 1:])
        logging_data_N2 = copy.deepcopy(logging_data[:, 1:])
    else:
        logging_data_N = copy.deepcopy(logging_data)
        logging_data_N2 = copy.deepcopy(logging_data)

    for i in range(logging_data.shape[0]):
        index_start = i
        index_end = min(index_start+windows_length//2, logging_data.shape[0]-1)
        index_start = max(0, index_end-windows_length)
        logging_data_temp = logging_data_N[index_start:index_end, :]
        for j in range(logging_data_N.shape[1]):
            max_F, min_F = get_extreme_value_by_ratio(logging_data_temp[:, j], ratio_c=max_ratio, range_c=logging_range)
            logging_data_N2[i,j] = (logging_data_N[i, j]-min_F)/(abs((max_F-min_F))+0.01)

    if DEPTH_USE:
        logging_data_N2 = np.hstack((logging_data[:,0].reshape(-1, 1), logging_data_N2))
        # logging_data[:, 1:] = logging_data_N2
        # logging_data_N2 = logging_data

    return logging_data_N2



# 测井曲线归一化
def data_Normalized(curve_org, DEPTH_USE=True, local_normalized=False, logging_range=[-99, 9999], max_ratio=0.1):
    curve_normalize = copy.deepcopy(curve_org)

    # 局部特征以及整体特征的曲线归一化
    # 局部的曲线归一化，及其消耗时间，非必要一般不进行处理
    if local_normalized:
        curve_normalize_locally = data_normalized_locally(curve_normalize, DEPTH_USE=True)
        return curve_normalize_locally

    curve_normalize_fully, extreme_list = data_normalized(curve_normalize, DEPTH_USE=DEPTH_USE, logging_range=logging_range, max_ratio=max_ratio)
    extreme_list = np.array(extreme_list)
    # np.set_printoptions(precision=4)
    # print('curve normalized shape is :{}, extreme list:\n{}'.format(curve_normalize_fully.shape, extreme_list))

    return curve_normalize_fully




def two_disScatter(KmeansPred, X_Input, Y_Input, pltTitle='', img_path=r'', skip_point = -1,
                   class_name=[], x_label='x_label', y_label='y_label'):
    """
    :param KmeansPred: 分类预测的结果数据，这个必须从0开始，到n+1，不能从1开始
    :param X_Input: 输入的X轴数据
    :param Y_Input: 输入的Y轴数据
    :param pltTitle: 绘制图形的标题
    :param img_path: 文件保存的目标路径
    :param skip_point: 点数太多，是否跳点绘图，跳点的步长设置
    :param class_name: 分类的类别名称
    :param x_label: x坐标，设置
    :param y_label: y坐标，设置
    :return:
    """
    X_Input_N = []
    Y_Input_N = []
    KmeansPred_N = []
    if skip_point > 0:
        for i in range(X_Input.shape[0]):
            if i % skip_point == 0:
                X_Input_N.append(X_Input[i])
                Y_Input_N.append(Y_Input[i])
                KmeansPred_N.append(KmeansPred[i])

        X_Input = np.array(X_Input_N)
        Y_Input = np.array(Y_Input_N)
        KmeansPred = np.array(KmeansPred_N).astype(np.int64)

    print('Drawing Scatter..........')
    # 最大类别设置
    ClassNum = int(np.max(KmeansPred))+1
    # 颜色设置
    # ColorList = ['black', 'red', 'chartreuse', 'springgreen', 'orange', 'dodgerblue', 'fuchsia', 'cornflowerblue', 'maroon', 'slategray']
    ColorList = ['black', 'darkred', 'darkslateblue', 'm', 'seagreen', 'cadetblue', 'tan', 'olivedrab', 'peru', 'slategray']
    # 不同类别，绘图的符号设置
    # MarkerList = [',', '.', '1', '2', '3', '4', '8', 's', 'o', 'v', '+', 'x', '^', '<', '>', 'p']
    MarkerList = ['v', '*', 'X', '+', 'D', 'h', '1', 's', 'o', '.', 'd', '>', '^', '<', 'p']
    for i in range(ClassNum):
        if i>= len(class_name):
            class_name.append('label{}'.format(i+1))

    # 基于类绘制图形
    pltPic = []
    for i in range(KmeansPred.shape[0]):
        a = []
        for j in range(ClassNum):
            if KmeansPred[i] == j:
                a = plt.scatter(X_Input[i], Y_Input[i], c=ColorList[j], marker=MarkerList[j], s=20, label=class_name[j])
        pltPic.append(a)

    plt.xlabel(x_label.replace('chu', '/'))
    plt.ylabel(y_label.replace('chu', '/'))
    plt.legend(pltPic, class_name[:ClassNum])
    plt.title(pltTitle.replace('chu', '/'))
    if img_path != '':
        plt.savefig(img_path+'/{}.png'.format(pltTitle))
    plt.show()


def data_disScatter(data, labels, pltTitle='', img_path=r'', skip_point=-1,
                   class_name=[], label=['x_label', 'y_label']):
    """
    :param data: 输入数据源 np.array, [N,Dim]
    :param labels: 对应标签数据 np.array, [N,]
    :param pltTitle: 标题
    :param img_path: 散点分布图保存图像路径
    :param skip_point: 跳点个数
    :param class_name: 类别名称
    :param x_label: x轴文本
    :param y_label: y轴文本
    :return:
    """
    pca = PCA(n_components=2)  # 加载PCA算法，设置降维后主成分数目为2
    data = pca.fit_transform(data)  # 对样本进行降维
    # print(data.shape, labels.shape)

    data_N = []
    labels_N = []
    if skip_point > 0:
        for i in range(data.shape[0]):
            if i % skip_point == 0:
                data_N.append(data[i, :])
                labels_N.append(labels[i])

        data_N = np.array(data_N)
        labels_N = np.array(labels_N).astype(np.int64)
    else:
        data_N = data
        labels_N = labels

    print('Drawing Scatter..........')
    # 尝试找到数组中的最大值，并设置最大类别个数
    try:
        ClassNum = int(np.max(labels_N)) + 1
    except ValueError as e:
        print(f"捕获到错误: {e}")
        exit(0)
    # 颜色设置
    ColorList = ['black', 'darkred', 'darkslateblue', 'm', 'seagreen', 'cadetblue', 'tan', 'olivedrab', 'peru', 'slategray']
    # 不同类别，绘图的符号设置
    MarkerList = ['v', '*', 'X', '+', 'D', 'h', '1', 's', 'o', '.', 'd', '>', '^', '<', 'p']
    # 标签设置
    for i in range(ClassNum):
        if i >= len(class_name):
            class_name.append('label{}'.format(i + 1))

    # 基于类绘制图形
    pltPic = []
    for j in range(ClassNum):
        labels_Temp = labels_N[labels_N==j]
        X_TEMP = data_N[labels_N==j]
        pltPic.append(plt.scatter(X_TEMP[:, 0], X_TEMP[:, 1], c=ColorList[j], marker=MarkerList[j], s=10))
    # for i in range(data_N.shape[0]):
    #     a = []
    #     for j in range(ClassNum):
    #         if labels_N[i] == j:
    #             a = plt.scatter(data_N[i, 0], data_N[i, 1], c=ColorList[j], marker=MarkerList[j], s=10)
    #     pltPic.append(a)

    plt.xlabel(label[0].replace('chu', '/'))
    plt.ylabel(label[1].replace('chu', '/'))
    plt.legend(pltPic, class_name[:ClassNum])
    # plt.legend()
    plt.title(pltTitle.replace('chu', '/'))
    if img_path != '':
        plt.savefig(img_path+'/{}.png'.format(pltTitle))
    # plt.clf()
    # plt.close()
    # # plt.savefig()
    plt.show()






# layer_inf_str=np.array([[1, 1], [2, 3], [3, 1], [4, 2], [5, 1], [6, 1], [7, 1], [8, 2], [9, 1], [10, 2], [11, 1], [12, 3]])
# print(layer_inf_str, '\n', layer_table_combine(layer_inf_str))



# # 将三维的 顶深-底深-类别 n*3信息 转换成二维的 深度-类别 的n*2信息,只能处理连续的分层，中间有断层的处理不了
# def layer_table_to_list(np_layer, step=0.0025, depth=[-1, -1]):
#     if np_layer.shape[1] > 3:
#         print('label if too larget, please give label as n*3 shape...')
#         exit()
#     dep_start = np_layer[0][0]
#     dep_end = np_layer[-1][1]
#     if (depth[0] > 0) & (depth[0] >= dep_start-step) & (depth[0] <= dep_end + step):
#         dep_start = depth[0]
#     if (depth[1] > 0) & (depth[1] >= dep_start-step) & (depth[0] <= dep_end + step):
#         dep_end = depth[1]
#
#     num_dep = int((dep_end - dep_start) / step)
#
#     # print(num_dep)
#     list_layer_depth = []
#     list_layer_class = []
#     layer_index = 0
#     # print(list_layer_class.shape)
#     for i in range(num_dep):
#         dep_temp = dep_start + i * step
#
#         while (layer_index < np_layer.shape[0]):
#             if ((dep_temp >= np_layer[layer_index][0]-step) & (
#                     dep_temp <= np_layer[layer_index][1]+step)):
#                 list_layer_depth.append(dep_temp)
#                 list_layer_class.append(np_layer[layer_index][2])
#                 # layer_index -= 1
#                 break
#             else:
#                 layer_index += 1
#     list_layer_depth = np.array(list_layer_depth).astype(float)
#     list_layer_class = np.array(list_layer_class)
#
#     return np.hstack((list_layer_depth.reshape((-1, 1)), list_layer_class.reshape((-1, 1))))



# 无监督聚类效果分析
def data_analysis_result_unsupervised_effect(data_normal_curve, list_layer_class):
    index_continue = 0
    data_box = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

    num_cluster = int(max(list_layer_class[:, -1]))
    if num_cluster >= len(data_box):
        print('cluster is too many.......')
        exit(0)

    # 对不同类别的数据进行分别装盒
    for i in range(list_layer_class.shape[0]):
        layer_class = int(list_layer_class[i, -1])
        # print(list_layer_class[i, 0], layer_class)
        data, index_continue = get_data_by_depth(data_normal_curve, depth=list_layer_class[i, 0], continue_index=index_continue)
        if data != []:
            data_box[layer_class].append(data)
        # print(data, index_continue)

    # 分别对不同盒中的数据进行分析
    data_statistical = []
    for i in range(num_cluster+1):
        # 对输入的j列数据进行分析，j==0时，数据为深度列
        for j in range(data_normal_curve.shape[1]):
            if j == 0:
                continue
            if data_box[i] != []:
                data_temp = np.array(data_box[i])
                a = np.mean(data_temp[:, j])
                c = np.var(data_temp[:, j], ddof=1)
                d = np.sqrt(np.var(data_temp[:, j]))
                # print(a, c, d)
                data_statistical.append([a, c, d])
            # 如果盒子为空，代表没有此类数据
            else:
                data_statistical.append([9999, 99999, 99999])
    data_statistical_3D = np.array(data_statistical).reshape((num_cluster+1, data_normal_curve.shape[1]-1, 3))
    data_statistical = np.array(data_statistical)
    # print(data_statistical)

    return data_statistical, data_statistical_3D


# 渗透率处理曲线，主要包括曲线的数据范围预处理，以及，曲线的函数滤波
def PERM_Process_fun(x, data, show=False, show_index_range=[200, 1200]):
    x = x.ravel()
    data_N = data.ravel()

    # 曲线数据范围矫正 以及 曲线取对数
    for i in range(data_N.shape[0]):
        if data_N[i] <= 0.01:
            data_N[i] = 0.01
        if data_N[i] > 1000:
            data_N[i] = 1000
        data_N[i] = math.log10(data_N[i])

    if show:
        # 可视化图线
        plt.plot(x[show_index_range[0]:show_index_range[1]], data_N[show_index_range[0]:show_index_range[1]], 'r')

    # 使用Savitzky-Golay 滤波器后得到平滑图线
    from scipy.signal import savgol_filter

    data_N = savgol_filter(data_N, 7, 2, mode='nearest')

    if show:
        # 可视化图线
        plt.plot(x[show_index_range[0]:show_index_range[1]], y[show_index_range[0]:show_index_range[1]], 'b', label='savgol')
        # 显示曲线
        plt.show()

    # 曲线数据范围矫正 以及 曲线取对数
    for i in range(data_N.shape[0]):
        data_N[i] = 10**data_N[i]
        if data_N[i] < 0.01:
            data_N[i] = 0.01
        if data_N[i] > 1000:
            data_N[i] = 1000

    return data_N
# PERM_Process_fun()


# # 针对深度不连续测井数据设计的 根据深度信息，获取附近测井数据的函数
# def get_data_by_depth(data_normal_curve, depth, continue_index=0):
#     step = (data_normal_curve[-1, 0] - data_normal_curve[0, 0]) / data_normal_curve.shape[0]
#     data = []
#
#     if (depth<data_normal_curve[0][0]) | (depth>data_normal_curve[-1][0]):
#         return data, continue_index
#
#     for i in range(data_normal_curve.shape[0] - continue_index):
#         depth_temp = data_normal_curve[i + continue_index][0]
#         if  (abs(depth-depth_temp) <= step/2) | (i==data_normal_curve.shape[0]-continue_index-1) | (depth<=depth_temp):
#             continue_index -= 2
#             break
#     data = data_normal_curve[i, :]
#     continue_index += i
#
#     return data, max(continue_index, 0)


# 根据深度信息删除测井曲线数据
def delete_logging_by_depth(data_curve, depth_delete=[(1,2), (3,4)]):
    """
    :param data_curve:要输入的测井曲线信息，第一列必须为深度列,且必须连续
    :param depth_delete: 要删除的深度段信息
    :return:
    """
    # 首先检查要剔除的深度信息列表depth_delete是否有误
    depth_t = 0
    if len(depth_delete)>0:
        for i in range(len(depth_delete)):
            depth_start, depth_end = depth_delete[i]
            if depth_start > depth_end:
                print('depth error, index:{}, depth :[{}, {}]'.format(i, depth_start, depth_end))
                exit(0)

            if i!=0:
                if depth_start < depth_t:
                    print('depth error, index:{}->{}, depth :[{}, {}, {}]'.format(i-1, i, depth_t, depth_start, depth_t))
                    exit(0)
            depth_t = depth_end
    else:
        return data_curve

    # 进行信息的挑选
    logging_target = []
    j = 0
    depth_temp = data_curve[j, 0]
    for i in range(len(depth_delete)):
        if j<data_curve.shape[0]:
            depth_start, depth_end = depth_delete[i]
        else:
            break
        while ((depth_temp < depth_end) and (j < data_curve.shape[0])):
            if depth_temp < depth_start:
                logging_target.append(data_curve[j])
            j += 1
            if j<data_curve.shape[0]:
                depth_temp = data_curve[j, 0]
    while (j < data_curve.shape[0]):
        logging_target.append(data_curve[j])
        j += 1

    logging_target = np.array(logging_target)
    return logging_target




def activity_based_segmentation(df, window_size=5, threshold=0.2, cols=None):
    """
    基于活度函数的地层分割接口
    参数：
        df: 测井数据DataFrame，首列为深度列
        window_size: 滑动窗口大小（奇数）
        threshold: 峰值检测阈值(0-1)
        cols: 指定处理的测井曲线列名列表
    返回：
        分割点深度值列表
    """
    # 数据预处理
    depth_col = df.columns[0]
    if cols is None:
        cols = df.columns[1:]  # 自动排除深度列
    else:
        cols = [c for c in cols if c != depth_col]

    # 计算各曲线活度
    activity = pd.DataFrame()
    for col in cols:
        # 滑动窗口局部方差计算
        rolling_var = df[col].rolling(window=window_size, center=True).var().fillna(0)
        activity[col] = rolling_var / rolling_var.max()  # 归一化

    # 综合活度计算
    composite_activity = activity.mean(axis=1)  # 多参数平均

    # 寻找活度峰值
    peaks, _ = find_peaks(composite_activity, height=threshold)

    # 获取分割深度
    depth_values = df[depth_col].values
    split_depths = depth_values[peaks]

    return sorted(split_depths.tolist())


def cluster_by_activity(df, window_size=5, threshold=0.2, cols=['GR', 'RT']):
    # 调用接口
    segments = activity_based_segmentation(
        df,
        window_size=window_size,
        threshold=threshold,
        cols=cols
    )
    segments = [df.iloc[0,0]] + segments + [df.iloc[-1,0]]
    # print("地层分割深度点：", segments)

    type_3_col = []
    for i in range(len(segments)):
        if i == 0:
            pass
        else:
            type_3_col.append([segments[i-1], segments[i], i-1])
    type_3_col = np.array(type_3_col)


    type_2_col = data_combine_table3col(df.values, type_3_col)


    df_result = pd.DataFrame(type_2_col, columns=list(df.columns)+['Activity'])
    return df_result
############ cluster_by_activity函数测试
# # 生成模拟测井数据（10种岩性，其中3种为稀有类别）
# X, y = make_classification(n_samples=2000, n_features=8, n_informative=5,
#                            n_classes=10, weights=[0.05,0.03,0.02]+[0.9/7]*7,
#                            random_state=42)
#
# # 创建数据框
# feature_cols = ['GR','RT','SP','CALI','DEN','CNL','PE','AC']
# df = pd.DataFrame(X, columns=feature_cols)
# df['岩性'] = y
#
# print("原始分布:", Counter(y))
#
# # 调用处理接口（组合方法示例）
# balanced_data = smart_balance_dataset(df, method='smote')
#
# print("平衡后分布:", Counter(balanced_data['岩性']))



################ data_combine_new函数测试
# np.set_printoptions(precision=2, suppress=True)
# a = create_random_logging(logging_data_shape=(100, 5), logging_resolution=1, dep_start=100)
# print(a.shape)
# print(a[-10:, :])
# b = delete_logging_by_depth(a, depth_delete=[(110,140), (150, 200)])
# print(b.shape)
# print(b)
# log = create_random_logging((30, 3), 1, 25)
# table_test = np.array([
#     [20, 30, 2],
#     [33, 35, 1],
#     [40, 50, 3],
#     # [82, 80, 3],
# ])
# print(data_combine_limitation(log, table_test, drop=-1))
# np.set_printoptions(precision=4, suppress=True)
# a = create_random_logging(logging_data_shape=(16, 5), logging_resolution=0.1, dep_start=99.3)
# b = create_random_logging(logging_data_shape=(7, 5), logging_resolution=0.15, dep_start=99)
# a = np.delete(a, 3, axis=0)
# a = np.delete(a, 3, axis=0)
# b = np.delete(b, -2, axis=0)
#
# data_all = data_combine_new([a, b], step='MIN')
# print(data_all)

