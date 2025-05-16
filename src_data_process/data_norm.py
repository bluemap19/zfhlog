

def data_Normalized(curve_org, DEPTH_USE=True, local_normalized=False, logging_range=[-99, 9999], max_ratio=0.1):
    curve_normalize = copy.deepcopy(curve_org)

    curve_normalize_fully, extreme_list = data_normalized(curve_normalize, DEPTH_USE=DEPTH_USE, logging_range=logging_range, max_ratio=max_ratio)
    extreme_list = np.array(extreme_list)
    np.set_printoptions(precision=4)
    print('curve normalized shape is :{}, extreme list:\n{}'.format(curve_normalize_fully.shape, extreme_list))

    # 局部特征以及整体特征的曲线归一化
    # 局部的曲线归一化，及其消耗时间，非必要一般不进行处理
    if local_normalized:
        curve_normalize_locally = data_normalized_locally(curve_normalize, DEPTH_USE=True)
        return curve_normalize_locally

    return curve_normalize_fully


def data_norm(self, i):
    print('current well name:{}'.format(self.Config_logging[i]['sheet_name']))
    logging_data_Temp = data_Normalized(self.data_list[i].values, DEPTH_USE=True, local_normalized=False)
    logging_data_Temp[logging_data_Temp < -99] = np.nan  # 对不合规的数据 进行原地修改
    logging_data_Temp = pd.DataFrame(logging_data_Temp, columns=self.Config_logging[i]['curve_name'])
    print(f'data normalized describe:\n{logging_data_Temp.describe()}')
    self.data_norm_list.append(logging_data_Temp)