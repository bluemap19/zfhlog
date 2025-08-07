import numpy as np
import pandas as pd

from src_file_op.dir_operation import search_files_by_criteria

# 长庆重新分层，这个主要作用是从原始的分类分层文件中截取合适的层段并进行保存
if __name__ == '__main__':
    # path_config = r'C:\Users\ZFH\Desktop\长73数据调整\斜井110\Target_workspace.csv'
    # target_path = r'C:\Users\ZFH\Desktop\长73数据调整\斜井110\长73-筛选'
    # file_source = r'C:\Users\ZFH\Desktop\长73数据调整\斜井110\长71-长81'

    path_config = r'C:\Users\ZFH\Desktop\长73数据调整\直井332\Target_workspace.csv'
    target_path = r'C:\Users\ZFH\Desktop\长73数据调整\直井332\长73-筛选'
    file_source = r'C:\Users\ZFH\Desktop\长73数据调整\直井332\长71-长81'
    data_df_depth_config = pd.read_csv(path_config, encoding='gbk')
    print(data_df_depth_config.describe())
    # data_df_depth_config = pd.DataFrame([['蔡221', '2050', '2091', '2056.3', '2091'], ["白75", "2320.6", "2355.6", "2314.6", "2355.6"], ["白78", "2176.8", "2205.2", "2173.5", "2205.2"]], columns=['WELL_NAME', 'DEPTH_START_OLD', 'DEPTH_END_OLD', 'DEPTH_START_NEW', 'DEPTH_END_NEW'])

    data_df_depth_config['DEPTH_START_NEW'] = data_df_depth_config['DEPTH_START_NEW'].astype(float)
    data_df_depth_config['DEPTH_END_NEW'] = data_df_depth_config['DEPTH_END_NEW'].astype(float)

    print(data_df_depth_config)
    NOT_FIND_WELL_NAME = []
    CANT_HANDLE_WELL_NAME = []

    # 根据井，遍历文件，进行数据截取
    for i in range(0,len(data_df_depth_config)):
        WELL_NAME = data_df_depth_config['WELL_NAME'][i]
        DEPTH_START = data_df_depth_config['DEPTH_START_NEW'][i]
        DEPTH_END = data_df_depth_config['DEPTH_END_NEW'][i]

        path_target = search_files_by_criteria(search_root=file_source, name_keywords=['last_litho_tops', WELL_NAME+'_'], file_extensions=['csv'])
        print(WELL_NAME, DEPTH_START, DEPTH_END, path_target)

        if len(path_target) == 1:
            print('find target well as charter {},as result :{}'.format(WELL_NAME, path_target))
        elif len(path_target) == 0:
            print('Not find target file:{}, now continue'.format(WELL_NAME))
            NOT_FIND_WELL_NAME.append(WELL_NAME)
            continue
        else:
            print('find multi target file use charter:{},as result :{}'.format(WELL_NAME, path_target))
            exit(0)

        # 测井分层数据文件读取
        df_layer_data = pd.read_csv(path_target[0], encoding='gbk')
        df_layer_data.drop(0, axis=0, inplace=True)  # 删除数据第一行的文字信息
        df_layer_data.reset_index(drop=True, inplace=True)

        # 第一列Depth深度转换为 float，第三列EFAC_1研祥类型转换为 int
        df_layer_data['DEPTH'] = df_layer_data['DEPTH'].astype(float)
        df_layer_data['EFAC_1'] = df_layer_data['EFAC_1'].astype(int)
        # print(df_layer_data.head(10))

        # 目的层数据获取
        index_start = -1
        index_end = -1

        # 查找最接近 DEPTH_START 的索引
        # 处理起始索引
        if DEPTH_START <= df_layer_data['DEPTH'].iloc[0]:
            index_start = 0
        elif DEPTH_START >= df_layer_data['DEPTH'].iloc[-1]:
            index_start = len(df_layer_data) - 1
        else:
            for j in range(0, len(df_layer_data) - 1):
                current = df_layer_data['DEPTH'].iloc[j]
                next_val = df_layer_data['DEPTH'].iloc[j + 1]
                # 浮点数安全比较
                if np.isclose(current, DEPTH_START, atol=0.01):
                    index_start = j
                    break
                elif current < DEPTH_START < next_val:
                    index_start = j
                    break

        # 同样逻辑处理 index_end
        if DEPTH_END >= df_layer_data['DEPTH'].iloc[-1]:
            index_end = len(df_layer_data) - 1
        elif DEPTH_END <= df_layer_data['DEPTH'].iloc[0]:
            index_end = 0
        else:
            for j in range(0, len(df_layer_data) - 1):
                current = df_layer_data['DEPTH'].iloc[j]
                next_val = df_layer_data['DEPTH'].iloc[j + 1]
                if np.isclose(current, DEPTH_END, atol=0.01):
                    index_end = j
                    break
                elif current < DEPTH_END < next_val:
                    index_end = j
                    break

        # 验证索引有效性
        if index_start == -1 or index_end == -1 or index_start==index_end:
            print('error depth from {} to {}, file depth is from {} to {}'.format(DEPTH_START, DEPTH_END, df_layer_data['DEPTH'].iloc[0], df_layer_data['DEPTH'].iloc[-1]))
            CANT_HANDLE_WELL_NAME.append(WELL_NAME)
            continue

        print('get data from index {} to {}'.format(index_start, index_end))
        # 根据筛选的index，选取目标数据
        target_data_df = df_layer_data.iloc[index_start:index_end + 1, :]
        print(target_data_df.head(5))
        print(target_data_df.tail(5))
        target_data_df = target_data_df.reset_index(drop=True)


        # 删除 EFAC_1 = -999 的行，重置索引
        if target_data_df['EFAC_1'].iloc[-1] == -999:       # 最后一行的话，就不要删除了，删除了的话会出问题
            target_data_df.iloc[-1, target_data_df.columns.get_loc('EFAC_1')] = 10
        target_data_df = target_data_df[target_data_df['EFAC_1'] != -999].reset_index(drop=True)
        # print(target_data_df.tail(5))

        # 删除最后几行，不是凝灰岩的层
        if target_data_df.iloc[-3, target_data_df.columns.get_loc('NAME_1')].__contains__('凝灰岩'):
            target_data_df = target_data_df.iloc[:-2, :].reset_index(drop=True)
        elif target_data_df.iloc[-4, target_data_df.columns.get_loc('NAME_1')].__contains__('凝灰岩'):
            target_data_df = target_data_df.iloc[:-3, :].reset_index(drop=True)
        else:
            target_data_df = target_data_df.reset_index(drop=True)
        # print(target_data_df.tail(5))

        # 删除中间层中，那些重复出现的层
        index_drop_target = []
        for j in range(0,len(target_data_df)-2):
            litho_current = target_data_df['NAME_1'][j]
            litho_next = target_data_df['NAME_1'][j+1]
            if litho_current == litho_next:
                index_drop_target.append(j)
        target_data_df = target_data_df.drop(index_drop_target).reset_index(drop=True)
        print(index_drop_target, '\n',target_data_df.tail(5))

        # 将最后一行中，COLOR_1、EFAC_1、NAME_1、PATTERN_1四列的内容删除
        target_data_df.iloc[-1, target_data_df.columns.get_loc('COLOR_1')] = None
        target_data_df.iloc[-1, target_data_df.columns.get_loc('EFAC_1')] = None
        target_data_df.iloc[-1, target_data_df.columns.get_loc('NAME_1')] = None
        target_data_df.iloc[-1, target_data_df.columns.get_loc('PATTERN_1')] = None

        # 文件保存成CSV文件
        path_save = target_path+'\\'+WELL_NAME+'_last_litho_tops_c73.csv'
        target_data_df.to_csv(path_save, index=False)


    print(NOT_FIND_WELL_NAME)
    print(CANT_HANDLE_WELL_NAME)
