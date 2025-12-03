import logging
import os

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union

from src_data_process.data_correction_analysis import feature_influence_analysis
from src_data_process.data_dilute import dilute_dataframe
from src_data_process.data_linear_regression import calculate_predictions
from src_data_process.data_linear_regression_2 import MultiVariateLinearRegressor
from src_file_op.dir_operation import search_files_by_criteria
from src_logging.logging_combine import data_combine_table2col
from src_plot.TEMP_9 import WellLogVisualizer
from src_plot.plot_matrxi_scatter import plot_matrxi_scatter
from src_well_data_base.data_logging_FMI import DataFMI
from src_well_data_base.data_logging_normal import DataLogging
from src_well_data_base.data_logging_table import DataTable


class DATA_WELL:
    """
    井数据统一管理器：
    - 日常曲线测井数据 DataLogging
    - 电成像 FMI DataFMI
    - 表格类型数据 DataTable
    - 未来拓展 NMR 数据
    """

    # =============== 基础初始化 ==================
    def __init__(self, path_folder: str = '', WELL_NAME: str = ''):

        # ---- 数据容器 ----
        self.logging_dict: Dict[str, DataLogging] = {}
        self.table_dict: Dict[str, DataTable] = {}
        self.FMI_dict: Dict[str, DataFMI] = {}
        self.NMR_dict: Dict[str, Any] = {}

        # ---- 路径容器 ----
        self.path_list_logging: List[str] = []
        self.path_list_table: List[str] = []
        self.path_list_fmi: List[str] = []
        self.path_list_nmr: List[str] = []

        # 根路径
        self.well_path = path_folder

        # ---- 井名判定 ----
        if WELL_NAME:
            self.WELL_NAME = WELL_NAME
        else:
            self.WELL_NAME = os.path.basename(path_folder)

        # ---- 文件识别关键字 ----
        self.LOGGING_KW = ['logging']
        self.TABLE_KW = ['table', 'LITHO_TYPE']
        self.FMI_KW = ['DYNA', 'STAT']
        self.NMR_KW = ['nmr']

        # 初始化路径扫描
        self.scan_files()

    # =========================================================================
    #                          文件扫描模块
    # =========================================================================
    def scan_files(self):
        """扫描井目录，识别各类文件路径"""
        if not os.path.exists(self.well_path):
            print(f"[WARN] 路径不存在: {self.well_path}")
            return

        self.path_list_logging = search_files_by_criteria(
            self.well_path,
            name_keywords=self.LOGGING_KW,
            file_extensions=['.xlsx', '.csv'],
            all_keywords=False
        )

        self.path_list_table = search_files_by_criteria(
            self.well_path,
            name_keywords=self.TABLE_KW,
            file_extensions=['.xlsx', '.csv'],
            all_keywords=False
        )

        self.path_list_fmi = search_files_by_criteria(
            self.well_path,
            name_keywords=self.FMI_KW,
            file_extensions=['.txt'],
            all_keywords=False
        )

        self.path_list_nmr = search_files_by_criteria(
            self.well_path,
            name_keywords=self.NMR_KW,
            file_extensions=['.csv'],
            all_keywords=False
        )

    # =========================================================================
    #                          内部辅助函数
    # =========================================================================
    def _get_default_obj(self, data_dict: Dict, key: str = ''):
        """
        dict 为空 → 返回空
        key为空  → 返回第一个
        key匹配  → 返回对应对象
        """
        if not data_dict:
            print("\033[33m[WARN] 数据未初始化\033[0m")
            return None

        if not key:
            return next(iter(data_dict.values()))  # 返回第一个对象

        # 支持模糊匹配
        for k in data_dict.keys():
            if key in k:
                return data_dict[k]

        # 完全匹配失败
        return None

    # =========================================================================
    #                          数据初始化模块
    # =========================================================================
    def init_logging(self, path: str = ''):
        """初始化普通测井数据"""
        if not path:
            if not self.path_list_logging:
                return
            path = self.path_list_logging[0]

        if path not in self.logging_dict:
            self.logging_dict[path] = DataLogging(path=path, well_name=self.WELL_NAME)

    def init_table(self, path: str = ''):
        """初始化表格数据"""
        if not path:
            if not self.path_list_table:
                return
            path = self.path_list_table[0]

        if path not in self.table_dict:
            self.table_dict[path] = DataTable(path=path, well_name=self.WELL_NAME)

    def init_FMI(self, path: str = ''):
        """初始化电成像数据（stat/dyna均可）"""
        if not path:
            if not self.path_list_fmi:
                return
            path = self.path_list_fmi[0]

        if path not in self.FMI_dict:
            self.FMI_dict[path] = DataFMI(path_fmi=path)

    # =========================================================================
    #                          统一访问接口
    # =========================================================================
    def get_logging(self, key: str = '', curve_names: List[str] = None, norm=False):
        """
        获取测井数据 DataFrame

        :param key: 文件名或关键字
        :param curve_names: 需要的曲线列表
        :param norm: 是否归一化
        """
        self.init_logging(key)
        obj = self._get_default_obj(self.logging_dict, key)
        if obj is None:
            return pd.DataFrame()
        return obj.get_data_normed(curve_names) if norm else obj.get_data(curve_names)

    def get_table(self, key: str = '', mode='3', replaced=False, replace_dict=None, new_col='Type_Replaced'):
        """
        mode='3': depth_start, depth_end, type
        mode='2': depth, type
        """
        self.init_table(key)
        obj = self._get_default_obj(self.table_dict, key)
        if obj is None:
            return pd.DataFrame()

        if replaced and replace_dict:
            obj._apply_type_replacement(replace_dict=new_col)

        return obj.get_table_3() if mode == '3' else obj.get_table_2()

    def get_FMI(self, key: str = '', depth: Optional[List[float]] = None):
        """获得 FMI 电成像数据"""
        self.init_FMI(key)
        obj = self._get_default_obj(self.FMI_dict, key)
        if obj is None:
            return None
        return obj.get_data(depth)

    # =========================================================================
    #                          数据概览接口
    # =========================================================================
    def well_summary(self) -> Dict[str, Any]:
        return {
            "well": self.WELL_NAME,
            "path": self.well_path,
            "paths_logging": self.path_list_logging,
            "paths_fmi": self.path_list_fmi,
            "paths_table": self.path_list_table,
            "paths_nmr": self.path_list_nmr,
            "logging_files_num": len(self.path_list_logging),
            "fmi_files_num": len(self.path_list_fmi),
            "table_files_num": len(self.path_list_table),
            "nmr_files_num": len(self.path_list_nmr),
        }

    def __repr__(self):
        return f"<DATA_WELL {self.WELL_NAME} | logging={len(self.logging_dict)}, fmi={len(self.FMI_dict)}, table={len(self.table_dict)}>"

    def combine_logging_table(
            self,
            logging_key='',
            curve_names_logging=None,
            table_key='',
            replace_dict=None,
            new_col='Type',
            norm=False,
    ):
        """
        将连续曲线logging与类型表（3列或2列）合并
        生成 (depth + curves + lithology_label)
        """

        # 1 获取曲线数据
        df_log = self.get_logging(logging_key, curve_names_logging, norm)
        depth_col = df_log.columns[0]

        # 2 获取 table
        self.init_table(table_key)
        table_obj = self._get_default_obj(self.table_dict, table_key)

        if replace_dict:
            table_obj._apply_type_replacement(replace_dict=replace_dict, new_col=new_col)

        df_tab = table_obj.get_table_2_replaced()

        # 排序
        df_log = df_log.sort_values(depth_col)
        df_tab = df_tab.sort_values(df_tab.columns[0])


        logging_columns = list(df_log.columns)
        table_columns = list(df_tab.columns)
        array_logging = df_log.values.astype(np.float32)
        array_table = df_tab.values.astype(np.float32)
        array_merge = data_combine_table2col(array_logging, array_table, drop=True)

        data_columns = logging_columns + [table_columns[-1]]
        df_merge = pd.DataFrame(array_merge, columns=data_columns)
        df_merge.dropna(inplace=True)
        df_merge[table_columns[-1]] = df_merge[table_columns[-1]].astype(int)

        if new_col != '' or new_col is None:
            ##### 重命名
            df_merge.rename(columns={table_columns[-1]: new_col}, inplace=True)

        return df_merge


    def update_data_with_type(self, key, df: pd.DataFrame):
        obj = self._get_default_obj(self.logging_dict, key)
        if obj is None: return
        obj.data_with_type = df  # 推荐公开属性，不要操作 protected 成员

    def search_data_path(self, keywords: List[str], path_list):
        for p in path_list:
            if all(k.lower() in p.lower() for k in keywords):
                return p
        return None

    def search_logging_data(self, keywords=[], curve_names=None, norm=False):
        path = self.search_data_path(keywords, self.path_list_logging)
        return self.get_logging(path, curve_names, norm)

    def get_table_replace_dict(self, table_key=''):
        self.init_table(table_key)
        table_obj = self._get_default_obj(self.table_dict, table_key)
        return table_obj.get_replace_dict()

if __name__ == '__main__':
    # well = DATA_WELL("F:\logging_workspace\桃镇1H")
    well = DATA_WELL(r'F:\logging_workspace\禄探')

    summary_temp = well.well_summary()
    for k, val in summary_temp.items():
        print(k, val)

    print(well.get_table_replace_dict(table_key='F:\\logging_workspace\\禄探\\df_层理类型_table.csv'))

    logging_data_temp = well.get_logging()

    # input_cols = ['AC', 'CAL', 'CNL', 'DEN', 'DTS', 'GR', 'RT', 'RXO']
    # input_cols = ['CON_MEAN_STAT', 'DIS_MEAN_STAT', 'HOM_MEAN_STAT', 'ENG_MEAN_STAT', 'COR_MEAN_STAT', 'ASM_MEAN_STAT', 'ENT_MEAN_STAT', 'CON_SUB_STAT', 'DIS_SUB_STAT', 'HOM_SUB_STAT', 'ENG_SUB_STAT', 'COR_SUB_STAT', 'ASM_SUB_STAT', 'ENT_SUB_STAT', 'CON_X_STAT', 'DIS_X_STAT', 'HOM_X_STAT', 'ENG_X_STAT', 'COR_X_STAT', 'ASM_X_STAT', 'ENT_X_STAT', 'CON_Y_STAT', 'DIS_Y_STAT', 'HOM_Y_STAT', 'ENG_Y_STAT', 'COR_Y_STAT', 'ASM_Y_STAT', 'ENT_Y_STAT']
    # input_cols = ['COR_MEAN_STAT', 'COR_Y_STAT', 'ASM_SUB_STAT', 'HOM_SUB_STAT', 'COR_X_STAT', 'ENG_SUB_STAT', 'ENT_SUB_STAT', 'HOM_X_STAT']

    # input_cols = ['CON_MEAN_DYNA', 'DIS_MEAN_DYNA', 'HOM_MEAN_DYNA', 'ENG_MEAN_DYNA', 'COR_MEAN_DYNA', 'ASM_MEAN_DYNA', 'ENT_MEAN_DYNA', 'CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA', 'ENG_SUB_DYNA', 'COR_SUB_DYNA', 'ASM_SUB_DYNA', 'ENT_SUB_DYNA', 'CON_X_DYNA', 'DIS_X_DYNA', 'HOM_X_DYNA', 'ENG_X_DYNA', 'COR_X_DYNA', 'ASM_X_DYNA', 'ENT_X_DYNA', 'CON_Y_DYNA', 'DIS_Y_DYNA', 'HOM_Y_DYNA', 'ENG_Y_DYNA', 'COR_Y_DYNA', 'ASM_Y_DYNA', 'ENT_Y_DYNA']
    # input_cols = ['ENT_SUB_DYNA', 'HOM_X_DYNA', 'ENG_SUB_DYNA', 'HOM_SUB_DYNA', 'ASM_SUB_DYNA', 'CON_SUB_DYNA', 'CON_X_DYNA', 'DIS_SUB_DYNA']
    input_cols = ['HOM_X_STAT', 'HOM_X_DYNA', 'ENT_SUB_DYNA', 'ASM_SUB_STAT', 'CON_X_DYNA', 'CON_SUB_DYNA', 'COR_MEAN_STAT', 'ASM_SUB_DYNA', 'HOM_SUB_STAT', 'COR_Y_STAT', 'HOM_SUB_DYNA', 'ENG_SUB_DYNA', 'ENT_SUB_STAT', 'COR_X_STAT', 'DIS_SUB_DYNA', 'ENG_SUB_STAT']

    target_col = 'Type'
    logging_data_temp_type = well.combine_logging_table(
            logging_key='F:\\logging_workspace\\禄探\\禄探_TEXTURE_logging_120.csv',
            curve_names_logging=[],
            table_key='F:\\logging_workspace\\禄探\\df_层理类型_table.csv',
            replace_dict={},
            new_col=target_col,
            norm=True,
    )
    print(list(logging_data_temp_type.columns))
    print(logging_data_temp_type.describe())
    # print(logging_data_temp_type.tail(10))

    # 创建模型实例
    model = MultiVariateLinearRegressor(fit_intercept=True)

    # 训练模型（使用x1, x2, x3预测y1, y2）
    model.fit(logging_data_temp_type, x_cols=['HOM_X_STAT', 'HOM_X_DYNA', 'ENT_SUB_DYNA', 'ASM_SUB_STAT'], y_cols=['Type'])

    # 进行预测
    logging_data_temp_type['Type_pred'] = model.predict(logging_data_temp_type)
    # 计算评估指标
    test_metrics = model.score(logging_data_temp_type)

    # pearson_result, pearson_sorted, rf_result, rf_sorted = feature_influence_analysis(
    #     df_input=logging_data_temp_type,
    #     input_cols=input_cols,
    #     target_col=target_col,
    #     regressor_use=False,
    #     replace_dict={},
    # )
    # print("\n按皮尔逊系数排序的属性:", pearson_sorted)
    # print("\n按随机森林特征重要性排序的属性:", rf_sorted)

    # # 按组抽稀，每个组保留50%的数据
    # logging_data_dilute = dilute_dataframe(logging_data_temp_type, ratio=5, method='random', group_by=target_col)
    # print(f"按组抽稀50%后形状: {logging_data_dilute.shape}")
    # plot_matrxi_scatter(logging_data_dilute, ['HOM_X_STAT', 'HOM_X_DYNA', 'ENT_SUB_DYNA', 'ASM_SUB_STAT', 'CON_X_DYNA', 'CON_SUB_DYNA', 'COR_MEAN_STAT'], target_col, target_col_dict={})

    depth_config = [logging_data_temp_type['DEPTH'].min(), logging_data_temp_type['DEPTH'].max()]
    fmi_dynamic, depth_dyna = well.get_FMI(key='F:\\logging_workspace\\禄探\\禄探_DYNA.txt', depth=depth_config)
    fmi_static, depth_stat = well.get_FMI(key='F:\\logging_workspace\\禄探\\禄探_STAT.txt', depth=depth_config)

    # 使用类接口进行可视化
    print("创建可视化器...")
    visualizer = WellLogVisualizer()
    try:
        # 启用详细日志级别
        logging.getLogger().setLevel(logging.INFO)

        # 执行可视化
        visualizer.visualize(
            logging_dict={'data': logging_data_temp_type,
                          'depth_col': 'DEPTH',
                          'curve_cols': ['CON_MEAN_STAT', 'DIS_MEAN_STAT', 'HOM_MEAN_STAT', 'ENG_MEAN_STAT', 'Type_pred'],  # 选择显示的曲线
                          'type_cols': ['Type'],  # 分类数据
                          'legend_dict': {0: '0', 1: '1', 2: '2', 3: '3'}  # 图例定义
                          },
            fmi_dict={  # FMI图像数据
                'depth': depth_dyna,
                'image_data': [fmi_dynamic, fmi_static],
                'title': ['FMI动态', 'FMI静态']
            },
            # # NMR_dict=[NMR_DICT1, NMR_DICT2],
            # NMR_dict={  # NMR数据
            #     'depth': depth_fmi,
            #     'nmr_data': [fmi_dynamic, fmi_static],
            #     'title': ['NMR动态', 'NMR静态']
            # },
            # NMR_CONFIG={'X_LOG': [False, True], 'NMR_TITLE': ['N1_谱', 'N2_谱'], 'X_LIMIT': [[1, 1000], [1, 1000]],
            #             'Y_scaling_factor': 12, 'JUMP_POINT': 15},
            # depth_limit_config=[320, 380],  # 深度限制
            figsize=(16, 12)  # 图形尺寸
        )

        # 显示性能统计
        stats = visualizer.get_performance_stats()
        print("性能统计:", stats)

    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback

        traceback.print_exc()  # 打印完整错误堆栈
    finally:
        # 清理资源
        visualizer.close()