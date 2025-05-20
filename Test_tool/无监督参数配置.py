import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QCheckBox, QGroupBox, QScrollArea, QFormLayout,
                             QSpinBox, QComboBox, QPushButton, QDoubleSpinBox)
from PyQt5.QtCore import Qt


class UnsupervisedConfigWindow(QMainWindow):
    def __init__(self, wells, curves):
        super().__init__()
        self.wells = wells
        self.curves = curves
        self.initUI()

    def initUI(self):
        self.setWindowTitle('无监督聚类分析平台')
        self.setGeometry(300, 200, 800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 使用标签页组织不同配置
        self.tabs = QTabWidget()

        # 数据选择页
        self.data_tab = self.create_data_selection()
        self.tabs.addTab(self.data_tab, "数据选择")

        # 算法配置页
        self.tabs.addTab(self.create_kmeans_config(), "KMeans")
        self.tabs.addTab(self.create_hierarchical_config(), "层次聚类")
        self.tabs.addTab(self.create_gmm_config(), "GMM")
        self.tabs.addTab(self.create_spectral_config(), "谱聚类")

        # 操作按钮
        self.btn_confirm = QPushButton('开始对比分析')
        self.btn_confirm.clicked.connect(self.show_config)

        layout.addWidget(self.tabs)
        layout.addWidget(self.btn_confirm)

    def create_data_selection(self):
        """数据选择页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 井选择
        well_group = QGroupBox("选择井（多选）")
        self.well_checks = []
        well_layout = QVBoxLayout()
        for well in self.wells:
            cb = QCheckBox(well)
            cb.setChecked(True)
            self.well_checks.append(cb)
            well_layout.addWidget(cb)
        well_group.setLayout(well_layout)

        # 曲线选择
        curve_group = QGroupBox("选择测井曲线（多选）")
        self.curve_checks = []
        curve_layout = QVBoxLayout()
        for curve in self.curves:
            cb = QCheckBox(curve)
            cb.setChecked(True)
            self.curve_checks.append(cb)
            curve_layout.addWidget(cb)
        curve_group.setLayout(curve_layout)

        # 添加滚动区域
        scroll = QScrollArea()
        content = QWidget()
        scroll_layout = QVBoxLayout(content)
        scroll_layout.addWidget(well_group)
        scroll_layout.addWidget(curve_group)
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)

        layout.addWidget(scroll)
        return tab

    def create_kmeans_config(self):
        """KMeans参数"""
        widget = QWidget()
        layout = QFormLayout(widget)

        self.kmeans_n_clusters = QSpinBox()
        self.kmeans_n_clusters.setRange(2, 100)
        self.kmeans_n_clusters.setValue(8)

        self.kmeans_init = QComboBox()
        self.kmeans_init.addItems(['k-means++', 'random'])

        self.kmeans_max_iter = QSpinBox()
        self.kmeans_max_iter.setRange(100, 1000)
        self.kmeans_max_iter.setValue(300)

        layout.addRow("簇数量:", self.kmeans_n_clusters)
        layout.addRow("初始化方法:", self.kmeans_init)
        layout.addRow("最大迭代次数:", self.kmeans_max_iter)

        return widget

    def create_hierarchical_config(self):
        """层次聚类参数"""
        widget = QWidget()
        layout = QFormLayout(widget)

        self.hierarchical_n_clusters = QSpinBox()
        self.hierarchical_n_clusters.setRange(2, 100)
        self.hierarchical_n_clusters.setValue(5)

        self.hierarchical_linkage = QComboBox()
        self.hierarchical_linkage.addItems(['ward', 'complete', 'average', 'single'])

        self.hierarchical_metric = QComboBox()
        self.hierarchical_metric.addItems(['euclidean', 'cosine', 'manhattan'])

        layout.addRow("簇数量:", self.hierarchical_n_clusters)
        layout.addRow("链接方式:", self.hierarchical_linkage)
        layout.addRow("距离度量:", self.hierarchical_metric)

        return widget

    def create_gmm_config(self):
        """高斯混合模型参数"""
        widget = QWidget()
        layout = QFormLayout(widget)

        self.gmm_n_components = QSpinBox()
        self.gmm_n_components.setRange(2, 100)
        self.gmm_n_components.setValue(5)

        self.gmm_covariance_type = QComboBox()
        self.gmm_covariance_type.addItems(['full', 'tied', 'diag', 'spherical'])

        self.gmm_max_iter = QSpinBox()
        self.gmm_max_iter.setRange(100, 1000)
        self.gmm_max_iter.setValue(200)

        layout.addRow("高斯组件数:", self.gmm_n_components)
        layout.addRow("协方差类型:", self.gmm_covariance_type)
        layout.addRow("最大迭代次数:", self.gmm_max_iter)

        return widget

    def create_spectral_config(self):
        """谱聚类参数"""
        widget = QWidget()
        layout = QFormLayout(widget)

        self.spectral_n_clusters = QSpinBox()
        self.spectral_n_clusters.setRange(2, 100)
        self.spectral_n_clusters.setValue(5)

        self.spectral_affinity = QComboBox()
        self.spectral_affinity.addItems(['rbf', 'nearest_neighbors', 'cosine'])

        self.spectral_gamma = QDoubleSpinBox()
        self.spectral_gamma.setRange(0.1, 10.0)
        self.spectral_gamma.setValue(1.0)

        layout.addRow("簇数量:", self.spectral_n_clusters)
        layout.addRow("相似度度量:", self.spectral_affinity)
        layout.addRow("核系数gamma:", self.spectral_gamma)

        return widget

    def get_selected_data(self):
        """获取选中的数据"""
        return {
            "wells": [cb.text() for cb in self.well_checks if cb.isChecked()],
            "curves": [cb.text() for cb in self.curve_checks if cb.isChecked()]
        }

    def get_all_config(self):
        """获取所有配置参数"""
        return {
            "data": self.get_selected_data(),
            "KMeans": {
                "n_clusters": self.kmeans_n_clusters.value(),
                "init": self.kmeans_init.currentText(),
                "max_iter": self.kmeans_max_iter.value()
            },
            "Hierarchical": {
                "n_clusters": self.hierarchical_n_clusters.value(),
                "linkage": self.hierarchical_linkage.currentText(),
                "metric": self.hierarchical_metric.currentText()
            },
            "GMM": {
                "n_components": self.gmm_n_components.value(),
                "covariance_type": self.gmm_covariance_type.currentText(),
                "max_iter": self.gmm_max_iter.value()
            },
            "Spectral": {
                "n_clusters": self.spectral_n_clusters.value(),
                "affinity": self.spectral_affinity.currentText(),
                "gamma": self.spectral_gamma.value()
            }
        }

    def show_config(self):
        """显示配置结果"""
        config = self.get_all_config()
        print("当前所有配置参数:")
        import json
        print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    # 测试数据
    test_wells = ["白75", "白159", "白291", "白302", "白413"]
    test_curves = ["GR", "DEN", "CNL", "RT", "SP", "CAL", "AC"]

    app = QApplication(sys.argv)
    window = UnsupervisedConfigWindow(wells=test_wells, curves=test_curves)
    window.show()
    sys.exit(app.exec_())