import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QToolBar, QAction, QTreeWidget, QTreeWidgetItem, QSplitter,
                             QTextEdit, QLabel, QStackedWidget, QDialog, QGroupBox, QCheckBox, QButtonGroup,
                             QRadioButton, QPushButton, QGridLayout, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon


class WellAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initToolbar()
        self.initTreeData()
        self.initContentArea()

    def initUI(self):
        """初始化主界面布局"""
        self.setWindowTitle('随钻测录导数据协同算法与软件模块')
        self.setGeometry(300, 200, 2000, 1000)

        # 主容器
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 主布局
        main_layout = QHBoxLayout(main_widget)

        # 分割器（左右布局）
        self.splitter = QSplitter(Qt.Horizontal)

        # 左侧树状导航（占1/4）
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setStyleSheet("""
            QTreeWidget {
                background: #F5F5F5;
                border-right: 1px solid #DDD;
                font: 12px 'Microsoft YaHei';
            }
            QTreeWidget::item {
                height: 28px;
                padding: 4px;
            }
            QTreeWidget::item:selected {
                background: #E1F0FF;
                color: #0066CC;
            }
        """)
        self.splitter.addWidget(self.tree)

        # 右侧内容区域（占3/4）
        self.content_stack = QStackedWidget()
        self.splitter.addWidget(self.content_stack)

        # 设置分割比例
        self.splitter.setSizes([300, 900])

        main_layout.addWidget(self.splitter)

    def initToolbar(self):
        """初始化工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        toolbar.setStyleSheet("""
            QToolBar {
                background: #F8F9FA;
                border-bottom: 1px solid #DEE2E6;
                spacing: 8px;
                padding: 4px;
            }
            QToolButton {
                padding: 6px 12px;
            }
        """)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        # 工具栏动作
        actions = [
            ('新建井', 'document-new', self.newWell),
            ('参数设置', 'settings', self.showSettings),
            ('特征分析', 'chart', self.analyzeFeatures),
            # ('地质建模', 'modeling', self.geoModeling),
            # ('模型正演', 'acting', self.modelActing),
            ('自动分层', 'unsupervised', self.autoStratify),
            ('分类识别', 'supervised', self.classifySupervised),
            # ('分类识别', 'layers', self.classifyData),
            # ('分类识别', 'layers', self.classifyData),
        ]

        for text, icon, callback in actions:
            action = QAction(QIcon.fromTheme(icon), text, self)
            action.triggered.connect(callback)
            toolbar.addAction(action)
            # toolbar.addSeparator()

    def initTreeData(self):
        """生成示例树状数据结构"""
        # 第一口井
        well1 = QTreeWidgetItem(["Well1"])
        well1.addChildren([
            self.createTreeItem("井参数概况", "well_params"),
            self.createTreeItem("测井曲线", "well_curves", [
                "AC", "CNL", "DEN", "GR", "CAL", "SP", "RD", "Rxo", "DT24", "POR", 'Type'
            ]),
            self.createTreeItem("其他数据", "well_features", [
                "岩性分类", "地质分层", "流体分类", "录井资料"
            ])
        ])

        # 第二口井
        well2 = QTreeWidgetItem(["Well2"])
        well2.addChildren([
            self.createTreeItem("井参数概况", "well_params"),
            self.createTreeItem("测井曲线", "well_curves", [
                "AC", "CNL", "DEN", "GR", "CAL", "CALX", "CALY", "SP", "Rlld", "Rs", "DTC", "PORC"
            ]),
            self.createTreeItem("其他数据", "classification", [
                "岩性分类", "地质分层", "流体分类", "录井资料"
            ])
        ])

        # 第二口井
        well3 = QTreeWidgetItem(["Well3"])
        well3.addChildren([
            self.createTreeItem("井参数概况", "well_params"),
            self.createTreeItem("测井曲线", "well_curves", [
                "AC", "CNL", "DEN", "GR", "CAL", "CALX", "CALY", "SP", "Rlld", "Rs", "DTC", "PORC"
            ]),
            self.createTreeItem("其他数据", "classification", [
                "岩性分类", "地质分层", "流体分类", "录井资料"
            ])
        ])

        # 第二口井
        well4 = QTreeWidgetItem(["Well4"])
        well4.addChildren([
            self.createTreeItem("井参数概况", "well_params"),
            self.createTreeItem("测井曲线", "well_curves", [
                "AC", "CNL", "DEN", "GR", "CAL", "CALX", "CALY", "SP", "Rlld", "Rs", "DTC", "PORC"
            ]),
            self.createTreeItem("其他数据", "classification", [
                "岩性分类", "地质分层", "流体分类", "录井资料"
            ])
        ])

        self.tree.addTopLevelItems([well1, well2, well3, well4])
        self.tree.itemClicked.connect(self.updateContent)

    def createTreeItem(self, title, data_type, children=None):
        """创建带数据类型的树节点"""
        item = QTreeWidgetItem([title])
        item.setData(0, Qt.UserRole, data_type)
        if children:
            item.addChildren([QTreeWidgetItem([child]) for child in children])
        return item

    def initContentArea(self):
        """初始化内容显示区域"""
        # 默认内容
        default_content = QLabel("请选择左侧导航项查看详细信息")
        default_content.setAlignment(Qt.AlignCenter)
        default_content.setStyleSheet("""
            QLabel {
                color: #666;
                font: 14px 'Microsoft YaHei';
            }
        """)

        # 参数设置页
        self.settings_editor = QTextEdit()
        self.settings_editor.setPlaceholderText("参数设置区域...")

        # 添加页面
        self.content_stack.addWidget(default_content)
        self.content_stack.addWidget(self.settings_editor)

    def updateContent(self, item):
        """更新右侧内容区域"""
        data_type = item.data(0, Qt.UserRole)

        if data_type == "well_params":
            self.showSettingsContent(item)
        elif data_type == "well_curves":
            self.showCurveContent(item)
        else:
            self.showDefaultContent(item)

    def showSettingsContent(self, item):
        """显示参数设置界面"""
        content = QTextEdit()
        content.setHtml(f"""
            <h3>{item.text(0)}</h3>
            <p>当前井：{item.parent().text(0)}</p>
            <ul>
                <li>采样间隔：0.125m</li>
                <li>深度范围：1223.52-2537.26m</li>
                <li>坐标系：WGS84</li>
            </ul>
        """)
        self.content_stack.addWidget(content)
        self.content_stack.setCurrentWidget(content)

    def showCurveContent(self, item):
        """显示测井曲线内容"""
        content = QLabel(f"测井曲线分析：{item.text(0)}")
        content.setAlignment(Qt.AlignCenter)
        self.content_stack.addWidget(content)
        self.content_stack.setCurrentWidget(content)

    def showDefaultContent(self, item):
        """默认内容显示"""
        content = QLabel(f"当前选择：{item.text(0)}")
        content.setAlignment(Qt.AlignCenter)
        self.content_stack.addWidget(content)
        self.content_stack.setCurrentWidget(content)

    # 工具栏功能实现（示例）
    def newWell(self):
        print("新建井操作...")

    def showSettings(self):
        self.content_stack.setCurrentWidget(self.settings_editor)

    #     # ***AI看这里***
    #     # 现在请你在这里生成代码，要求弹出一个配置框，配置框内要求显示：井选择（CheckBox，可多选）、输入数据选择（CheckBox，可多选）、输出数据选择（单选）、数据分析算法选择（单选：矩阵散点-核密度图、小波变换、随机森林）
    #     # 然后提供确定、取消按钮，确定按钮点击时，销毁窗口并打印选择的配置，取消按钮点击时，也销毁窗口，然后什么都不做，返回主界面
    def analyzeFeatures(self):
        print("启动特征分析...")

        class ConfigDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("特征分析配置")
                self.setFixedSize(600, 500)  # 调整对话框尺寸

                # 主布局结构
                main_layout = QVBoxLayout(self)
                main_layout.setContentsMargins(10, 10, 10, 10)

                # 滚动区域容器
                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                content_widget = QWidget()
                scroll.setWidget(content_widget)

                # 内容布局（包含所有配置组件）
                content_layout = QVBoxLayout(content_widget)
                content_layout.setSpacing(10)

                # ===== 井选择区域 =====
                well_group = QGroupBox("井选择（多选）")
                well_layout = QVBoxLayout()
                wellnames_items = ["Well1", "Well2", "Well3", "Well4"]
                self.well_checks = {item: QCheckBox(item) for item in wellnames_items}
                for cb in self.well_checks.values():
                    well_layout.addWidget(cb)
                well_group.setLayout(well_layout)
                content_layout.addWidget(well_group)

                # ===== 数据选择区域 =====
                data_columns = QHBoxLayout()
                data_columns.setSpacing(15)

                # 输入数据（多选）- 网格布局
                input_group = QGroupBox("输入数据（多选）")
                input_layout = QGridLayout()
                input_items = ["AC", "CNL", "DEN", "GR", "CAL", "SP", "RD", "Rxo", "DT24", "POR", 'Type']
                self.input_checks = {item: QCheckBox(item) for item in input_items}

                # 分两列布局
                for idx, cb in enumerate(self.input_checks.values()):
                    row = idx // 2
                    col = idx % 2
                    input_layout.addWidget(cb, row, col)
                input_group.setLayout(input_layout)
                data_columns.addWidget(input_group)

                # 输出数据（单选）- 网格布局
                output_group = QGroupBox("任务目标数据选择（单选）")
                output_layout = QGridLayout()
                self.output_radio = QButtonGroup(self)
                output_items = ["AC", "CNL", "DEN", "GR", "CAL", "SP", "RD", "Rxo", "DT24", "POR", 'Type']

                for idx, item in enumerate(output_items):
                    rb = QRadioButton(item)
                    self.output_radio.addButton(rb)
                    row = idx // 2
                    col = idx % 2
                    output_layout.addWidget(rb, row, col)
                output_group.setLayout(output_layout)
                data_columns.addWidget(output_group)

                content_layout.addLayout(data_columns)

                # ===== 算法选择区域 =====
                algo_group = QGroupBox("分析算法（单选）")
                algo_layout = QVBoxLayout()
                self.algo_radio = QButtonGroup(self)
                algorithms = ['矩阵散点-核密度图', '小波变换', '随机森林']

                for algo in algorithms:
                    rb = QRadioButton(algo)
                    self.algo_radio.addButton(rb)
                    algo_layout.addWidget(rb)
                algo_group.setLayout(algo_layout)
                content_layout.addWidget(algo_group)

                # ===== 按钮区域 =====
                btn_layout = QHBoxLayout()
                btn_layout.addStretch()
                ok_btn = QPushButton("确定")
                ok_btn.clicked.connect(self.on_confirm)
                cancel_btn = QPushButton("取消")
                cancel_btn.clicked.connect(self.reject)
                btn_layout.addWidget(ok_btn)
                btn_layout.addWidget(cancel_btn)

                # 组合布局
                main_layout.addWidget(scroll)
                main_layout.addLayout(btn_layout)

                # ===== 样式设置 =====
                self.setStyleSheet("""
                    QGroupBox {
                        font: bold 12px 'Microsoft YaHei';
                        border: 1px solid #E0E0E0;
                        margin-top: 10px;
                        padding-top: 15px;
                    }
                    QCheckBox, QRadioButton {
                        font: 11px 'Microsoft YaHei';
                        spacing: 6px;
                        min-width: 80px;
                    }
                    QPushButton {
                        min-width: 90px;
                        padding: 6px 12px;
                        font: 11px 'Microsoft YaHei';
                    }
                """)

                # 设置默认选项
                self.well_checks['Well1'].setChecked(True)
                self.input_checks['AC'].setChecked(True)
                list(self.output_radio.buttons())[0].setChecked(True)
                list(self.algo_radio.buttons())[0].setChecked(True)

            def on_confirm(self):
                """收集配置数据"""
                config = {
                    'wells': [name for name, cb in self.well_checks.items() if cb.isChecked()],
                    'inputs': [name for name, cb in self.input_checks.items() if cb.isChecked()],
                    'output': self.output_radio.checkedButton().text() if self.output_radio.checkedButton() else None,
                    'algorithm': self.algo_radio.checkedButton().text() if self.algo_radio.checkedButton() else None
                }

                print("\n当前配置:")
                print(f"选中的井: {config['wells']}")
                print(f"输入数据: {config['inputs']}")
                print(f"输出格式: {config['output']}")
                print(f"分析算法: {config['algorithm']}\n")
                self.accept()

        # 创建并显示对话框
        dialog = ConfigDialog(self)
        dialog.exec_()

    def classifyData(self):
        print("执行分类识别...")

    def autoStratify(self):
        print("自动分层处理...")

    def classifySupervised(self):
        print("启动特征分析...")

    def modelActing(self):
        print("执行分类识别...")

    def geoModeling(self):
        print("自动分层处理...")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = WellAnalysisApp()
    window.show()
    sys.exit(app.exec_())