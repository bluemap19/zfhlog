import sys
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QCheckBox, QComboBox, QLabel, QLineEdit, QPushButton,
                             QScrollArea, QTextEdit)
from PyQt5.QtCore import Qt


class DataPreprocessingDialog(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("测井数据预处理配置")
        self.setGeometry(300, 200, 800, 600)

        # 主界面布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 滚动区域
        scroll = QScrollArea()
        content_widget = QWidget()
        self.scroll_layout = QVBoxLayout(content_widget)

        # 步骤1：数据选择
        self.create_data_selection_group()

        # 步骤2：数据归一化
        self.create_normalization_group()

        # 步骤3：异常值处理
        self.create_outlier_group()

        # 步骤4：数据均衡化
        self.create_balance_group()

        # 步骤5：数据降维
        self.create_dimension_reduction_group()

        # 结果展示
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)

        # 按钮组
        btn_layout = QHBoxLayout()
        self.btn_confirm = QPushButton("执行处理")
        self.btn_cancel = QPushButton("取消")
        btn_layout.addWidget(self.btn_confirm)
        btn_layout.addWidget(self.btn_cancel)

        # 布局组装
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        main_layout.addWidget(self.output_area)
        main_layout.addLayout(btn_layout)

        # 信号连接
        self.btn_confirm.clicked.connect(self.process_config)
        self.btn_cancel.clicked.connect(self.close)

    def create_data_selection_group(self):
        """步骤1：数据选择"""
        group = QGroupBox("1. 数据选择（多选）")
        layout = QVBoxLayout()

        # 模拟测井曲线
        self.curve_checkboxes = []
        curves = ["AC", "CNL", "DEN", "GR", "CAL", "SP", "RD", "Rxo", "DT24", "POR"]
        for curve in curves:
            cb = QCheckBox(curve)
            self.curve_checkboxes.append(cb)
            layout.addWidget(cb)

        group.setLayout(layout)
        self.scroll_layout.addWidget(group)

    def create_normalization_group(self):
        """步骤2：数据归一化"""
        group = QGroupBox("2. 数据归一化")
        layout = QVBoxLayout()

        # 方法选择
        self.norm_method = QComboBox()
        self.norm_method.addItems(["Min-Max", "Z-Score", "Decimal Scaling"])

        # 参数配置
        param_layout = QHBoxLayout()
        self.norm_params = QLineEdit("feature_range=(0,1)")

        layout.addWidget(QLabel("归一化方法:"))
        layout.addWidget(self.norm_method)
        layout.addWidget(QLabel("参数配置:"))
        layout.addWidget(self.norm_params)

        group.setLayout(layout)
        self.scroll_layout.addWidget(group)

    def create_outlier_group(self):
        """步骤3：异常值处理"""
        group = QGroupBox("3. 异常值检测与处理")
        layout = QVBoxLayout()

        # 方法选择
        self.outlier_method = QComboBox()
        self.outlier_method.addItems(["Isolation Forest", "Z-Score", "IQR"])

        # 动态参数
        self.outlier_params = QLineEdit()
        self.outlier_method.currentTextChanged.connect(self.update_outlier_params)

        layout.addWidget(QLabel("检测方法:"))
        layout.addWidget(self.outlier_method)
        layout.addWidget(QLabel("参数配置:"))
        layout.addWidget(self.outlier_params)

        group.setLayout(layout)
        self.scroll_layout.addWidget(group)

    def create_balance_group(self):
        """步骤4：数据均衡化"""
        group = QGroupBox("4. 数据均衡化")
        layout = QVBoxLayout()

        self.balance_method = QComboBox()
        self.balance_method.addItems(["SMOTE", "RandomUnderSampler", "ADASYN"])

        layout.addWidget(QLabel("采样方法:"))
        layout.addWidget(self.balance_method)

        group.setLayout(layout)
        self.scroll_layout.addWidget(group)

    def create_dimension_reduction_group(self):
        """步骤5：数据降维"""
        group = QGroupBox("5. 数据降维")
        layout = QVBoxLayout()

        # 方法选择
        self.dr_method = QComboBox()
        self.dr_method.addItems(["PCA", "t-SNE", "LDA"])

        # 参数配置
        self.dr_params = QLineEdit()
        self.dr_method.currentTextChanged.connect(self.update_dr_params)

        layout.addWidget(QLabel("降维方法:"))
        layout.addWidget(self.dr_method)
        layout.addWidget(QLabel("参数配置:"))
        layout.addWidget(self.dr_params)

        group.setLayout(layout)
        self.scroll_layout.addWidget(group)

    def update_outlier_params(self, method):
        """动态更新异常检测参数"""
        params = {
            "Isolation Forest": "n_estimators=100, contamination=0.1",
            "Z-Score": "threshold=3",
            "IQR": "range_multiplier=1.5"
        }
        self.outlier_params.setText(params.get(method, ""))

    def update_dr_params(self, method):
        """动态更新降维参数"""
        params = {
            "PCA": "n_components=2",
            "t-SNE": "perplexity=30, learning_rate=200",
            "LDA": "n_components=1"
        }
        self.dr_params.setText(params.get(method, ""))

    def process_config(self):
        """处理配置参数"""
        config = {
            "selected_curves": [cb.text() for cb in self.curve_checkboxes if cb.isChecked()],
            "normalization": {
                "method": self.norm_method.currentText(),
                "params": self.norm_params.text()
            },
            "outlier_processing": {
                "method": self.outlier_method.currentText(),
                "params": self.outlier_params.text()
            },
            "data_balance": {
                "method": self.balance_method.currentText()
            },
            "dimension_reduction": {
                "method": self.dr_method.currentText(),
                "params": self.dr_params.text()
            }
        }

        # 格式化输出
        output = json.dumps(config, indent=2, ensure_ascii=False)
        self.output_area.setPlainText("预处理配置参数：\n" + output)

        # 此处可添加实际的数据处理逻辑
        print("执行预处理操作...")
        print(config)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataPreprocessingDialog()
    window.show()
    sys.exit(app.exec_())