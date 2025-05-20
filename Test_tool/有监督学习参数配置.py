import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QLabel, QComboBox, QLineEdit, QPushButton,
                             QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox)
from PyQt5.QtCore import Qt


class ModelConfigWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('多模型参数配置平台')
        self.setGeometry(300, 200, 800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 使用标签页组织不同模型
        self.tabs = QTabWidget()

        # 神经网络配置
        self.nn_tab = self.create_nn_config()
        self.tabs.addTab(self.nn_tab, "神经网络")

        # KNN配置
        self.knn_tab = self.create_knn_config()
        self.tabs.addTab(self.knn_tab, "KNN")

        # SVM配置
        self.svm_tab = self.create_svm_config()
        self.tabs.addTab(self.svm_tab, "SVM")

        # 朴素贝叶斯配置
        self.nb_tab = self.create_nb_config()
        self.tabs.addTab(self.nb_tab, "朴素贝叶斯")

        # 随机森林配置
        self.rf_tab = self.create_rf_config()
        self.tabs.addTab(self.rf_tab, "随机森林")

        # GBM配置
        self.gbm_tab = self.create_gbm_config()
        self.tabs.addTab(self.gbm_tab, "GBM")

        # 操作按钮
        self.btn_confirm = QPushButton('开始对比实验')
        self.btn_confirm.clicked.connect(self.show_config)

        layout.addWidget(self.tabs)
        layout.addWidget(self.btn_confirm)

    def create_nn_config(self):
        """神经网络参数"""
        tab = QWidget()
        layout = QFormLayout(tab)

        self.nn_layers = QSpinBox()
        self.nn_layers.setRange(1, 10)
        self.nn_layers.setValue(3)

        self.nn_units = QLineEdit("64,32,16")
        self.nn_activation = QComboBox()
        self.nn_activation.addItems(['relu', 'sigmoid', 'tanh'])

        self.nn_optimizer = QComboBox()
        self.nn_optimizer.addItems(['adam', 'sgd', 'rmsprop'])

        self.nn_lr = QDoubleSpinBox()
        self.nn_lr.setRange(0.0001, 1.0)
        self.nn_lr.setValue(0.001)

        layout.addRow("网络层数:", self.nn_layers)
        layout.addRow("每层神经元:", self.nn_units)
        layout.addRow("激活函数:", self.nn_activation)
        layout.addRow("优化器:", self.nn_optimizer)
        layout.addRow("学习率:", self.nn_lr)

        return tab

    def create_knn_config(self):
        """KNN参数"""
        tab = QWidget()
        layout = QFormLayout(tab)

        self.knn_neighbors = QSpinBox()
        self.knn_neighbors.setRange(1, 50)
        self.knn_neighbors.setValue(5)

        self.knn_metric = QComboBox()
        self.knn_metric.addItems(['euclidean', 'manhattan', 'minkowski'])

        self.knn_weights = QComboBox()
        self.knn_weights.addItems(['uniform', 'distance'])

        layout.addRow("邻居数量:", self.knn_neighbors)
        layout.addRow("距离度量:", self.knn_metric)
        layout.addRow("权重方式:", self.knn_weights)

        return tab

    def create_svm_config(self):
        """SVM参数"""
        tab = QWidget()
        layout = QFormLayout(tab)

        self.svm_c = QDoubleSpinBox()
        self.svm_c.setRange(0.1, 100.0)
        self.svm_c.setValue(1.0)

        self.svm_kernel = QComboBox()
        self.svm_kernel.addItems(['rbf', 'linear', 'poly'])

        self.svm_gamma = QComboBox()
        self.svm_gamma.addItems(['scale', 'auto'])

        layout.addRow("正则化系数(C):", self.svm_c)
        layout.addRow("核函数:", self.svm_kernel)
        layout.addRow("Gamma参数:", self.svm_gamma)

        return tab

    def create_nb_config(self):
        """朴素贝叶斯参数"""
        tab = QWidget()
        layout = QFormLayout(tab)

        self.nb_alpha = QDoubleSpinBox()
        self.nb_alpha.setRange(0.0, 1.0)
        self.nb_alpha.setValue(1.0)

        self.nb_fit_prior = QComboBox()
        self.nb_fit_prior.addItems(['True', 'False'])

        layout.addRow("平滑参数(alpha):", self.nb_alpha)
        layout.addRow("使用先验概率:", self.nb_fit_prior)

        return tab

    def create_rf_config(self):
        """随机森林参数"""
        tab = QWidget()
        layout = QFormLayout(tab)

        self.rf_n_estimators = QSpinBox()
        self.rf_n_estimators.setRange(10, 500)
        self.rf_n_estimators.setValue(100)

        self.rf_max_depth = QSpinBox()
        self.rf_max_depth.setRange(1, 50)
        self.rf_max_depth.setValue(10)

        self.rf_criterion = QComboBox()
        self.rf_criterion.addItems(['gini', 'entropy'])

        layout.addRow("树的数量:", self.rf_n_estimators)
        layout.addRow("最大深度:", self.rf_max_depth)
        layout.addRow("分裂标准:", self.rf_criterion)

        return tab

    def create_gbm_config(self):
        """GBM参数"""
        tab = QWidget()
        layout = QFormLayout(tab)

        self.gbm_learning_rate = QDoubleSpinBox()
        self.gbm_learning_rate.setRange(0.01, 1.0)
        self.gbm_learning_rate.setValue(0.1)

        self.gbm_n_estimators = QSpinBox()
        self.gbm_n_estimators.setRange(10, 500)
        self.gbm_n_estimators.setValue(100)

        self.gbm_subsample = QDoubleSpinBox()
        self.gbm_subsample.setRange(0.1, 1.0)
        self.gbm_subsample.setValue(1.0)

        layout.addRow("学习率:", self.gbm_learning_rate)
        layout.addRow("树的数量:", self.gbm_n_estimators)
        layout.addRow("子样本比例:", self.gbm_subsample)

        return tab

    def get_all_config(self):
        """获取所有配置参数"""
        return {
            "NeuralNetwork": {
                "layers": self.nn_layers.value(),
                "units": self.nn_units.text(),
                "activation": self.nn_activation.currentText(),
                "optimizer": self.nn_optimizer.currentText(),
                "learning_rate": self.nn_lr.value()
            },
            "KNN": {
                "n_neighbors": self.knn_neighbors.value(),
                "metric": self.knn_metric.currentText(),
                "weights": self.knn_weights.currentText()
            },
            "SVM": {
                "C": self.svm_c.value(),
                "kernel": self.svm_kernel.currentText(),
                "gamma": self.svm_gamma.currentText()
            },
            "NaiveBayes": {
                "alpha": self.nb_alpha.value(),
                "fit_prior": self.nb_fit_prior.currentText() == 'True'
            },
            "RandomForest": {
                "n_estimators": self.rf_n_estimators.value(),
                "max_depth": self.rf_max_depth.value(),
                "criterion": self.rf_criterion.currentText()
            },
            "GBM": {
                "learning_rate": self.gbm_learning_rate.value(),
                "n_estimators": self.gbm_n_estimators.value(),
                "subsample": self.gbm_subsample.value()
            }
        }

    def show_config(self):
        """显示配置结果"""
        config = self.get_all_config()
        print("当前所有模型配置:")
        import json
        print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ModelConfigWindow()
    window.show()
    sys.exit(app.exec_())