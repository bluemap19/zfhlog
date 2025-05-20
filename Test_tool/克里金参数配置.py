import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QLineEdit, QPushButton, QGroupBox,
                             QFormLayout, QDoubleSpinBox, QSpinBox)
from PyQt5.QtCore import Qt


class EnhancedKrigingConfigWindow(QMainWindow):
    def __init__(self, wells, well_curves):
        super().__init__()
        self.wells = wells  # 新增井列表
        self.well_curves = well_curves
        self.initUI()

    def initUI(self):
        self.setWindowTitle('增强版克里金配置')
        self.setGeometry(300, 300, 500, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # ================= 新增部分开始 =================
        # 井选择与深度设置
        well_group = QGroupBox("井选择与深度设置")
        well_layout = QFormLayout()

        # 井A选择
        self.well_a_combo = QComboBox()
        self.well_a_combo.addItems(self.wells)
        self.well_a_depth = QLineEdit("0-1000")
        well_layout.addRow("井A选择:", self.well_a_combo)
        well_layout.addRow("井A深度(m):", self.well_a_depth)

        # 井B选择
        self.well_b_combo = QComboBox()
        self.well_b_combo.addItems(self.wells)
        self.well_b_depth = QLineEdit("0-1000")
        well_layout.addRow("井B选择:", self.well_b_combo)
        well_layout.addRow("井B深度(m):", self.well_b_depth)

        well_group.setLayout(well_layout)
        layout.addWidget(well_group)
        # ================= 新增部分结束 =================

        # 原曲线选择组
        curve_group = QGroupBox("测井曲线选择")
        curve_layout = QVBoxLayout()
        self.curve_selector = QComboBox()
        self.curve_selector.addItems(self.well_curves)
        curve_layout.addWidget(QLabel("选择插值曲线:"))
        curve_layout.addWidget(self.curve_selector)
        curve_group.setLayout(curve_layout)
        layout.addWidget(curve_group)


        # 1. 测井曲线选择
        curve_group = QGroupBox("测井曲线选择")
        curve_layout = QVBoxLayout()
        self.curve_selector = QComboBox()
        self.curve_selector.addItems(self.well_curves)
        curve_layout.addWidget(QLabel("选择插值曲线:"))
        curve_layout.addWidget(self.curve_selector)
        curve_group.setLayout(curve_layout)
        layout.addWidget(curve_group)

        # 2. 变异函数参数
        vario_group = QGroupBox("变异函数参数")
        vario_layout = QFormLayout()

        # 变异函数类型
        self.vario_type = QComboBox()
        self.vario_type.addItems(['球状模型', '指数模型', '高斯模型'])

        # 变程
        self.range = QDoubleSpinBox()
        self.range.setRange(0.1, 1000.0)
        self.range.setValue(100.0)

        # 块金效应
        self.nugget = QDoubleSpinBox()
        self.nugget.setRange(0.0, 100.0)
        self.nugget.setValue(0.1)

        # 基台值
        self.sill = QDoubleSpinBox()
        self.sill.setRange(0.1, 1000.0)
        self.sill.setValue(1.0)

        vario_layout.addRow("模型类型:", self.vario_type)
        vario_layout.addRow("变程(m):", self.range)
        vario_layout.addRow("块金效应:", self.nugget)
        vario_layout.addRow("基台值:", self.sill)
        vario_group.setLayout(vario_layout)
        layout.addWidget(vario_group)

        # 3. 搜索参数
        search_group = QGroupBox("搜索参数")
        search_layout = QFormLayout()

        # 搜索半径
        self.radius = QDoubleSpinBox()
        self.radius.setRange(10.0, 1000.0)
        self.radius.setValue(200.0)

        # 最小邻居数
        self.min_pts = QSpinBox()
        self.min_pts.setRange(1, 20)
        self.min_pts.setValue(4)

        # 最大邻居数
        self.max_pts = QSpinBox()
        self.max_pts.setRange(1, 50)
        self.max_pts.setValue(12)

        search_layout.addRow("搜索半径(m):", self.radius)
        search_layout.addRow("最小邻居数:", self.min_pts)
        search_layout.addRow("最大邻居数:", self.max_pts)
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)

        # 操作按钮
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton('确定')
        self.btn_cancel = QPushButton('取消')
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_ok)
        layout.addLayout(btn_layout)

        # 信号连接
        self.btn_ok.clicked.connect(self.on_confirm)

        layout.addLayout(btn_layout)

    def get_config(self):
        """获取完整配置"""
        config = {
            # 新增配置项
            'wells': {
                'A': {
                    'name': self.well_a_combo.currentText(),
                    'depth_range': self.parse_depth(self.well_a_depth.text())
                },
                'B': {
                    'name': self.well_b_combo.currentText(),
                    'depth_range': self.parse_depth(self.well_b_depth.text())
                }
            },
            # 原配置项
            'curve': self.curve_selector.currentText(),
            # [其他原有配置项...]
        }
        return config

    def parse_depth(self, text):
        """解析深度输入"""
        try:
            start, end = map(float, text.replace(' ', '').split('-'))
            return (start, end)
        except:
            return (0.0, 1000.0)  # 默认值

    def on_confirm(self):
        """验证并输出配置"""
        config = self.get_config()
        # 深度格式验证
        if not self.validate_depth(config['wells']['A']['depth_range']):
            self.well_a_depth.setStyleSheet("border: 1px solid red;")
            return
        if not self.validate_depth(config['wells']['B']['depth_range']):
            self.well_b_depth.setStyleSheet("border: 1px solid red;")
            return

        print("完整配置:")
        print(json.dumps(config, indent=2, ensure_ascii=False))

    def validate_depth(self, depth_range):
        """验证深度范围有效性"""
        return depth_range[0] < depth_range[1]


if __name__ == '__main__':
    # 测试数据
    test_wells = ['白75', '白159', '白291']
    test_curves = ['GR', 'DEN', 'CNL', 'RT', 'SP']

    app = QApplication(sys.argv)
    window = EnhancedKrigingConfigWindow(wells=test_wells, well_curves=test_curves)
    window.show()
    sys.exit(app.exec_())