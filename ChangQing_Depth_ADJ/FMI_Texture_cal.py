import numpy as np

from src_well_project.LOGGING_PROJECT import LOGGING_PROJECT



if __name__ == '__main__':
    LG = LOGGING_PROJECT(project_path=r'F:\桌面\算法测试-长庆数据收集\logging_CSV')
    table = LG.get_table_3_all_data(['SIMU4'])
    print(table)

    # texture_set = LG.get_fmi_texture(well_names=['SIMU4'], Mode='ALL', texture_config={'level':16, 'distance':[2,4], 'angles':[0, np.pi/4, np.pi/2, np.pi*3/4], 'windows_length':80, 'windows_step':5})