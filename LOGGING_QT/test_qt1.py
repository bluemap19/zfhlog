import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QToolBar, QAction, QTreeWidget, QTreeWidgetItem, QSplitter,
                             QTextEdit, QLabel, QStackedWidget)
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
            ('地质建模', 'modeling', self.geoModeling),
            ('模型正演', 'acting', self.modelActing),
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
        well1 = QTreeWidgetItem(["白75"])
        well1.addChildren([
            self.createTreeItem("参数设置", "well_params"),
            self.createTreeItem("测井曲线", "well_curves", [
                "AC", "CNL", "DEN", "GR", "CAL", "SP", "RD", "Rxo", "DT24", "POR"
            ]),
            self.createTreeItem("其他数据", "well_features", [
                "岩性分类", "地质分层", "流体分类", "录井资料"
            ])
        ])

        # 第二口井
        well2 = QTreeWidgetItem(["白159"])
        well2.addChildren([
            self.createTreeItem("参数设置", "well_params"),
            self.createTreeItem("测井曲线", "well_curves", [
                "AC", "CNL", "DEN", "GR", "CAL", "CALX", "CALY", "SP", "Rlld", "Rs", "DTC", "PORC"
            ]),
            self.createTreeItem("其他数据", "classification", [
                "岩性分类", "地质分层", "流体分类", "录井资料"
            ])
        ])

        self.tree.addTopLevelItems([well1, well2])
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
                <li>采样间隔：0.5m</li>
                <li>深度范围：1200-2500m</li>
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

    def analyzeFeatures(self):
        print("启动特征分析...")

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