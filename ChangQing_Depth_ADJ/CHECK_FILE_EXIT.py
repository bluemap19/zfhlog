import pandas as pd
from src_file_op.dir_operation import get_all_file_paths

if __name__ == '__main__':
    # 统计CSV中存在的文件
    path_config = r'C:\Users\ZFH\Desktop\长73数据调整\直井332\Target_workspace.csv'
    data_df_depth_config = pd.read_csv(path_config, encoding='gbk')
    well_name_list = data_df_depth_config['WELL_NAME'].tolist()
    print(data_df_depth_config.describe())

    # 所有的分层文件
    file_source = r'C:\Users\ZFH\Desktop\长73数据调整\直井332\长71-长81'
    path_list = get_all_file_paths(file_source)
    charter_folder = []
    for path in path_list:
        path_charter = path.split('\\')[-1].split('_')[0]
        charter_folder.append(path_charter)

    # 查看哪些 CSV井名，能够在文件系统中找到 分层源文件
    find_list = []
    for charter in charter_folder:
        for j in range(len(well_name_list)):
            well_name_temp = well_name_list[j]
            if well_name_temp == charter:
                find_list.append(well_name_temp)
                break

    print(len(find_list), find_list)

    # 转换为集合并计算差集
    diff = list(set(charter_folder) - set(find_list))  # 或 set(A).difference(set(B))
    print(diff)  # 输出: [1, 3] （顺序可能随机）