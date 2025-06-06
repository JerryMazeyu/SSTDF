import sys
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                             QTabWidget, QPushButton, QMenu, QAction, QMessageBox)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QIcon
import logging

# 导入各个功能标签页（预留）
try:
    from app.views.tabs.tab1 import Tab1Widget  # 异常检测
except ImportError:
    Tab1Widget = None
    
try:
    from app.views.tabs.tab2 import Tab2Widget  # 哨兵系统 - 模型运行状态监控
except ImportError:
    Tab2Widget = None
    
try:
    from app.views.tabs.tab3 import Tab3Widget  # 趋势预测
except ImportError:
    Tab3Widget = None
    
try:
    from app.views.tabs.tab4 import Tab4Widget  # 模型剪枝
except ImportError:
    Tab4Widget = None


class MainWindow(QMainWindow):
    """主窗口类，用于集成各个功能模块"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("智能模型监控与优化系统")
        
        # 先设置一个合理的初始大小
        self.setGeometry(100, 100, 1200, 800)
        
        # 设置窗口状态为最大化
        self.setWindowState(Qt.WindowMaximized)
        
        # 创建中心部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 创建标签页部件
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #4CAF50;
            }
            QTabBar::tab:hover {
                background-color: #e0e0e0;
            }
        """)
        
        # 添加各个功能标签页
        self.setup_tabs()
        
        main_layout.addWidget(self.tab_widget)
        
        # 创建浮动设置按钮
        self.settings_button = QPushButton("☰", self.tab_widget)  # 将tab_widget作为父组件
        self.settings_button.setFixedSize(QSize(28, 28))  # 减小按钮尺寸
        self.settings_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(240, 240, 240, 200);
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(224, 224, 224, 240);
            }
            QPushButton:pressed {
                background-color: rgba(208, 208, 208, 240);
            }
        """)
        self.settings_button.clicked.connect(self.show_settings_menu)
        
        # 设置按钮位置到右上角（将在resizeEvent中调整）
        self.settings_button.move(self.tab_widget.width() - 35, 3)
        self.settings_button.raise_()  # 确保按钮在最上层
        
        # 定时器用于延迟设置按钮位置（确保tab_widget已正确渲染）
        QTimer.singleShot(100, self.adjust_settings_button_position)
        
    def setup_tabs(self):
        """设置各个功能标签页"""
        # Tab 1: 异常检测
        if Tab1Widget:
            self.tab1 = Tab1Widget()
            self.tab_widget.addTab(self.tab1, "异常检测")
        else:
            # 如果模块还未实现，添加占位符
            placeholder1 = QWidget()
            self.tab_widget.addTab(placeholder1, "异常检测")
            
        # Tab 2: 哨兵系统 - 模型运行状态监控
        if Tab2Widget:
            self.tab2 = Tab2Widget()
            self.tab_widget.addTab(self.tab2, "哨兵系统")
        else:
            placeholder2 = QWidget()
            self.tab_widget.addTab(placeholder2, "状态监控")
            
        # Tab 3: 趋势预测
        if Tab3Widget:
            self.tab3 = Tab3Widget()
            self.tab_widget.addTab(self.tab3, "趋势预测")
        else:
            placeholder3 = QWidget()
            self.tab_widget.addTab(placeholder3, "趋势预测")
            
        # Tab 4: 模型剪枝
        if Tab4Widget:
            self.tab4 = Tab4Widget()
            self.tab_widget.addTab(self.tab4, "模型剪枝")
        else:
            placeholder4 = QWidget()
            self.tab_widget.addTab(placeholder4, "模型剪枝")
    
    def show_settings_menu(self):
        """显示设置菜单"""
        menu = QMenu(self)
        
        # 设置菜单样式
        menu.setStyleSheet("""
            QMenu {
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px;
            }
            QMenu::item {
                padding: 8px 20px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #e0e0e0;
            }
        """)
        
        # 添加设置选项
        settings_action = QAction("⚙️ 设置", self)
        settings_action.triggered.connect(self.open_settings)
        menu.addAction(settings_action)
        
        # 添加分隔线
        menu.addSeparator()
        
        # 添加清空缓存选项
        clear_cache_action = QAction("🗑️ 清空缓存", self)
        clear_cache_action.triggered.connect(self.clear_cache)
        menu.addAction(clear_cache_action)
        
        # 在按钮下方显示菜单
        menu.exec_(self.settings_button.mapToGlobal(
            self.settings_button.rect().bottomLeft()
        ))
    
    def open_settings(self):
        """打开设置对话框"""
        # TODO: 实现设置对话框
        QMessageBox.information(self, "设置", "设置功能正在开发中...")
        self.logger.info("用户点击了设置按钮")
    
    def clear_cache(self):
        """清空缓存"""
        reply = QMessageBox.question(
            self, 
            "确认清空缓存", 
            "确定要清空所有缓存吗？\n此操作将删除所有临时数据。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # TODO: 实现清空缓存功能
            try:
                # 这里调用后端的清空缓存服务
                # from app.services.cache_service import clear_all_cache
                # clear_all_cache()
                QMessageBox.information(self, "成功", "缓存已清空！")
                self.logger.info("缓存已清空")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"清空缓存失败：{str(e)}")
                self.logger.error(f"清空缓存失败：{str(e)}")
    
    def adjust_settings_button_position(self):
        """调整设置按钮位置"""
        if hasattr(self, 'settings_button') and hasattr(self, 'tab_widget'):
            # 将设置按钮定位到标签页区域的右上角，更靠近边缘
            self.settings_button.move(self.tab_widget.width() - 35, 3)
    
    def resizeEvent(self, event):
        """窗口大小改变事件 - 调整设置按钮位置"""
        super().resizeEvent(event)
        self.adjust_settings_button_position()
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        reply = QMessageBox.question(
            self,
            "确认退出",
            "确定要退出程序吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
            self.logger.info("程序正常退出")
        else:
            event.ignore()


def create_main_window():
    """创建并返回主窗口实例"""
    return MainWindow()
