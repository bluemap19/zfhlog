from src_well_data.data_FMI import data_FMI


if __name__ == '__main__':
    test_FMI = data_FMI(
        path_dyna=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\FY1-15\FY1-15_dyna.txt',
        path_stat=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\FY1-15\FY1-15_stat.txt',
        well_name='FY1-15')

    Data_all = test_FMI.get_data(mode='ALL')
    print(Data_all['DYNA'].shape, Data_all['STAT'].shape, Data_all['DEPTH_DYNA'].shape, )

    test_FMI.ele_stripes_delete(mode='ALL')
    test_FMI.get_texture(mode='ALL', save_texture=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\FY1-15\FY1-15_texture.csv')