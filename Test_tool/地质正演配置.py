import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QGroupBox, QComboBox, QLineEdit, QPushButton,
                             QHBoxLayout, QLabel)
from PyQt5.QtCore import Qt


class ConfigWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 主窗口设置
        self.setWindowTitle('地质模型参数配置')
        self.setGeometry(300, 300, 400, 350)

        # 创建主容器
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 添加配置组件
        self.add_model_selection(layout)
        self.add_well_trajectory(layout)
        self.add_forward_algorithm(layout)
        self.add_depth_range(layout)
        self.add_confirm_button(layout)

    def add_model_selection(self, layout):
        """模型选择组件"""
        group = QGroupBox("模型配置")
        vbox = QVBoxLayout()

        # 模型选择
        lbl_model = QLabel("模型选择:")
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(['白75-白159', '白75-元543', '元543-白159'])

        vbox.addWidget(lbl_model)
        vbox.addWidget(self.cmb_model)
        group.setLayout(vbox)
        layout.addWidget(group)

    def add_well_trajectory(self, layout):
        """井轨迹选择组件"""
        group = QGroupBox("井轨迹配置")
        vbox = QVBoxLayout()

        # 井轨迹选择
        lbl_trajectory = QLabel("井轨迹:")
        self.cmb_trajectory = QComboBox()
        self.cmb_trajectory.addItems(['白75-15', '白75-23', '白159-21'])

        vbox.addWidget(lbl_trajectory)
        vbox.addWidget(self.cmb_trajectory)
        group.setLayout(vbox)
        layout.addWidget(group)

    def add_forward_algorithm(self, layout):
        """正演算法选择组件"""
        group = QGroupBox("正演算法")
        vbox = QVBoxLayout()

        # 算法选择
        lbl_algorithm = QLabel("算法类型:")
        self.cmb_algorithm = QComboBox()
        self.cmb_algorithm.addItems(['GR', '电磁波', '声波', '中子'])

        vbox.addWidget(lbl_algorithm)
        vbox.addWidget(self.cmb_algorithm)
        group.setLayout(vbox)
        layout.addWidget(group)

    def add_depth_range(self, layout):
        """深度范围组件"""
        group = QGroupBox("模型范围（米）")
        vbox = QVBoxLayout()

        # X范围
        hbox_x = QHBoxLayout()
        lbl_start_x = QLabel("起始深度X:")
        self.txt_start_x = QLineEdit()
        lbl_end_x = QLabel("结束深度X:")
        self.txt_end_x = QLineEdit()
        hbox_x.addWidget(lbl_start_x)
        hbox_x.addWidget(self.txt_start_x)
        hbox_x.addWidget(lbl_end_x)
        hbox_x.addWidget(self.txt_end_x)

        # Y范围
        hbox_y = QHBoxLayout()
        lbl_start_y = QLabel("起始深度Y:")
        self.txt_start_y = QLineEdit()
        lbl_end_y = QLabel("结束深度Y:")
        self.txt_end_y = QLineEdit()
        hbox_y.addWidget(lbl_start_y)
        hbox_y.addWidget(self.txt_start_y)
        hbox_y.addWidget(lbl_end_y)
        hbox_y.addWidget(self.txt_end_y)

        vbox.addLayout(hbox_x)
        vbox.addLayout(hbox_y)
        group.setLayout(vbox)
        layout.addWidget(group)

    def add_confirm_button(self, layout):
        """确认按钮"""
        self.btn_confirm = QPushButton("确认配置")
        self.btn_confirm.clicked.connect(self.on_confirm)
        layout.addWidget(self.btn_confirm, alignment=Qt.AlignCenter)

    def on_confirm(self):
        """确认按钮点击事件"""
        params = {
            'model': self.cmb_model.currentText(),
            'trajectory': self.cmb_trajectory.currentText(),
            'algorithm': self.cmb_algorithm.currentText(),
            'depth_range': {
                'start_x': self.txt_start_x.text(),
                'end_x': self.txt_end_x.text(),
                'start_y': self.txt_start_y.text(),
                'end_y': self.txt_end_y.text()
            }
        }
        print("当前配置参数:")
        print(f"模型选择: {params['model']}")
        print(f"井轨迹: {params['trajectory']}")
        print(f"正演算法: {params['algorithm']}")
        print(f"深度范围X: {params['depth_range']['start_x']} - {params['depth_range']['end_x']}")
        print(f"深度范围Y: {params['depth_range']['start_y']} - {params['depth_range']['end_y']}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ConfigWindow()
    window.show()
    sys.exit(app.exec_())