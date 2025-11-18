import logging

import numpy as np

from src_plot.TEMP_4 import WellLogVisualizer
from src_well_data.DATA_WELL import DATA_WELL

if __name__ == '__main__':
    # WELL_TEST = WELL_Class(path_folder=r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV\城96', WELL_NAME='城96')
    #
    # print(WELL_TEST.file_path_dict)
    #
    # # print(WELL_TEST.get_type_3())
    # print(WELL_TEST.get_type_2())
    #
    # # print(WELL_TEST.get_type_3_replaced())
    # print(WELL_TEST.get_type_2_replaced())
    #
    # path_logging = WELL_TEST.search_data_path_by_charters(target_path_feature=['logging_', '_110_'], target_file_type='logging')
    # path_table = WELL_TEST.search_data_path_by_charters(target_path_feature=['litho_'], target_file_type='table')
    # print(path_logging)
    # print(path_table)
    #
    # WELL_TEST.reset_table_replace_dict(path_table,
    #                                    replace_dict={'中GR长英黏土质': 0, '中低GR长英质': 1, '富有机质长英质页岩': 2,
    #                                                  '富有机质黏土质页岩': 4, '高GR富凝灰长英质': 3})
    # print('current table {} replace dict is :{}'.format(path_table, WELL_TEST.get_table_replace_dict(table_key=path_table)))
    #
    # # print(WELL_TEST.get_type_3_replaced())
    # print(WELL_TEST.get_type_2_replaced())
    #
    # data_combined = WELL_TEST.combine_logging_table(well_key=path_logging, table_key=path_table, Norm=False,
    #                                                 replace_dict={'中GR长英黏土质': 0, '中低GR长英质': 1,
    #                                                               '富有机质长英质页岩': 2, '富有机质黏土质页岩': 2, '高GR富凝灰长英质': 1})
    # print(data_combined.describe())
    #
    # data_logging = WELL_TEST.get_logging_data(well_key=path_logging, curve_names=[])
    # print(data_logging)

    # WELL_TEST = DATA_WELL(path_folder=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\城96', WELL_NAME='城96')
    # print(WELL_TEST.file_path_dict)
    # print(WELL_TEST.get_FMI_path_list())
    # print(WELL_TEST.get_type_3(table_key='F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\城96__纹理岩相划分_LITHO_TYPE.csv'))
    # print(WELL_TEST.get_type_2(table_key='F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\城96__纹理岩相划分_LITHO_TYPE.csv'))
    # print(WELL_TEST.combine_logging_table(well_key='F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\Texture_File\\城96_texture_logging_Texture_ALL_80_5.csv', curve_names_logging=[],
    #                                 table_key='F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\城96\\城96__纹理岩相划分_LITHO_TYPE.csv', curve_names_table=[],
    #                                 replace_dict={}, new_col='', Norm=False))

    # WELL_TEST = DATA_WELL(path_folder=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\樊页3HF')
    # # WELL_TEST = DATA_WELL(path_folder=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\樊页2HF')
    # print(WELL_TEST.get_FMI_path_list())
    # Data_all = WELL_TEST.get_fmi_data()
    # print(Data_all['DYNA'].shape, Data_all['STAT'].shape, Data_all['DEPTH_DYNA'].shape)
    # taaaaaa = WELL_TEST.get_fmi_texture(path_saved='F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\樊页3HF\\樊页3HF_TEXTURE.csv', texture_config={'level':16, 'distance':[2,4], 'angles':[0, np.pi/4, np.pi/2, np.pi*3/4], 'windows_length':80, 'windows_step':5})
    # # taaaaaa = WELL_TEST.get_fmi_texture(path_saved='F:\\桌面\\算法测试-长庆数据收集\\logging_CSV\\樊页2HF\\樊页2HF_TEXTURE.csv')
    # print(taaaaaa.shape)


    WELL_TEST = DATA_WELL(path_folder=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\桃镇1H')
    # WELL_TEST = DATA_WELL(path_folder=r'F:\桌面\算法测试-长庆数据收集\logging_CSV\樊页2HF')
    path_list_dyna, path_list_stat = WELL_TEST.get_FMI_path_list()
    print(path_list_dyna, path_list_stat)
    path_stat = path_list_dyna[2]
    path_dyna = path_list_stat[2]
    print(path_stat, path_dyna)
    WELL_TEST.data_FMI_init(path_stat=path_stat, path_dyna=path_dyna)
    fmi_data_dict = WELL_TEST.get_fmi_data(fmi_stat_key=path_stat, fmi_dyna_key=path_dyna)
    # 'DYNA':self._data_dyna, 'STAT':self._data_stat, 'DEPTH_DYNA':self._data_depth_dyna, 'DEPTH_STAT':self._data_depth_stat
    print(fmi_data_dict['DYNA'].shape)
    print(fmi_data_dict['DEPTH_DYNA'].shape)
    print(fmi_data_dict['STAT'].shape)
    print(fmi_data_dict['DEPTH_STAT'].shape)

    Logging_data = WELL_TEST.get_logging_data()
    print(Logging_data.columns, Logging_data.describe())

    # 使用类接口进行可视化
    print("创建可视化器...")
    visualizer = WellLogVisualizer()
    try:
        # 启用详细日志
        logging.getLogger().setLevel(logging.INFO)

        visualizer.visualize(
            data=Logging_data,
            depth_col='#DEPTH',
            curve_cols=['BIT', 'CAL', 'DEVOD', 'DEVST', 'DIP1', 'DIP2', 'DIP3', 'DIP4', 'DIP5', 'DIP6', 'GR', 'GR_S', 'K', 'RLLS', 'RM', 'RM_S', 'SW', 'TEN'],
            type_cols=[],
            # type_cols=['LITHOLOGY', 'FACIES'],
            # legend_dict={
            #     0: '砂岩',
            #     1: '页岩',
            #     2: '石灰岩',
            #     3: '白云岩'
            # },
            fmi_dict={
                'depth': fmi_data_dict['DEPTH_DYNA'],
                'image_data': [fmi_data_dict['DYNA'], fmi_data_dict['STAT']],
                'title': ['FMI动态', 'FMI静态']
            },
            # fmi_dict=None,
            # depth_limit_config=[320, 380],  # 只显示320-380米段
            figsize=(12, 8)
        )

        # 显示性能统计
        stats = visualizer.get_performance_stats()
        print("性能统计:", stats)

    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 清理资源
        visualizer.close()


