from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QGroupBox, QSplitter, QHeaderView, QFileDialog,
                             QListWidget, QListWidgetItem, QComboBox, 
                             QCheckBox, QProgressBar, QMessageBox, QMenu,
                             QAction, QAbstractItemView, QTextEdit, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QPoint
from PyQt5.QtGui import QFont, QPixmap, QPainter, QPen, QColor, QBrush, QImage
import logging
from datetime import datetime
import os
import json
from pathlib import Path
import numpy as np


class ImageDetectionThread(QThread):
    """图像检测线程"""
    progress_updated = pyqtSignal(int)
    detection_finished = pyqtSignal(dict)
    log_message = pyqtSignal(str)
    
    def __init__(self, images, model_ids):
        super().__init__()
        self.images = images
        self.model_ids = model_ids
        self.is_running = True
        
    def run(self):
        """执行检测任务"""
        from app.services import detection_service
        
        # 定义进度回调函数
        def progress_callback(current, total):
            if self.is_running:
                progress = int(current / total * 100)
                self.progress_updated.emit(progress)
        
        # 使用检测服务进行批量检测
        try:
            results = detection_service.detect_batch_images(
                self.images, 
                self.model_ids,
                progress_callback=progress_callback
            )
            
            # 发送每个检测结果
            for result in results:
                if not self.is_running:
                    break
                    
                self.log_message.emit(f"完成检测: {result['image_name']}")
                self.detection_finished.emit(result)
                
        except Exception as e:
            self.log_message.emit(f"批量检测失败: {str(e)}")
            
    def stop(self):
        """停止检测"""
        self.is_running = False


class ZoomableImageLabel(QLabel):
    """支持缩放的图像标签"""
    
    def __init__(self):
        super().__init__()
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel { 
                background-color: #f0f0f0; 
                border: 1px solid #ddd; 
            }
            QLabel:focus { 
                border: 2px solid #4CAF50; 
            }
        """)
        self.setMinimumSize(QSize(400, 150))
        
        # 允许接收键盘焦点
        self.setFocusPolicy(Qt.ClickFocus)
        
        # 缩放相关属性
        self.original_pixmap = None
        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0
        
        # 启用鼠标追踪
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)
        
        # 拖拽相关
        self.dragging = False
        self.drag_start_pos = None
        self.pixmap_offset = QPoint(0, 0)
        
        # 右键菜单
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
    
    def set_image(self, pixmap):
        """设置图像"""
        self.original_pixmap = pixmap
        self.scale_factor = 1.0
        self.pixmap_offset = QPoint(0, 0)
        # 默认适应窗口
        QTimer.singleShot(50, self.fit_to_window)
    
    def wheelEvent(self, event):
        """鼠标滚轮事件 - 用于缩放"""
        if self.original_pixmap:
            # 获取滚轮角度
            delta = event.angleDelta().y()
            
            # 计算缩放因子
            if delta > 0:
                scale_delta = 1.1
            else:
                scale_delta = 0.9
            
            # 更新缩放
            new_scale = self.scale_factor * scale_delta
            self.scale_factor = max(self.min_scale, min(self.max_scale, new_scale))
            
            self.update_display()
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        # 设置焦点
        self.setFocus()
        
        if event.button() == Qt.LeftButton and self.original_pixmap:
            self.dragging = True
            self.drag_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.dragging and self.original_pixmap:
            # 计算移动距离
            delta = event.pos() - self.drag_start_pos
            self.pixmap_offset += delta
            self.drag_start_pos = event.pos()
            self.update_display()
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.setCursor(Qt.OpenHandCursor)
    
    def show_context_menu(self, pos):
        """显示右键菜单"""
        if not self.original_pixmap:
            return
            
        menu = QMenu(self)
        
        # 放大
        zoom_in_action = QAction("放大 (Ctrl++)", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(self.zoom_in)
        menu.addAction(zoom_in_action)
        
        # 缩小
        zoom_out_action = QAction("缩小 (Ctrl+-)", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.zoom_out)
        menu.addAction(zoom_out_action)
        
        menu.addSeparator()
        
        # 适应窗口
        fit_action = QAction("适应窗口 (Ctrl+0)", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self.fit_to_window)
        menu.addAction(fit_action)
        
        # 原始大小
        actual_size_action = QAction("原始大小 100% (Ctrl+1)", self)
        actual_size_action.setShortcut("Ctrl+1")
        actual_size_action.triggered.connect(self.actual_size)
        menu.addAction(actual_size_action)
        
        menu.addSeparator()
        
        # 显示当前缩放比例
        scale_info = QAction(f"当前缩放: {int(self.scale_factor * 100)}%", self)
        scale_info.setEnabled(False)
        menu.addAction(scale_info)
        
        menu.exec_(self.mapToGlobal(pos))
    
    def zoom_in(self):
        """放大"""
        self.scale_factor = min(self.scale_factor * 1.2, self.max_scale)
        self.update_display()
    
    def zoom_out(self):
        """缩小"""
        self.scale_factor = max(self.scale_factor * 0.8, self.min_scale)
        self.update_display()
    
    def fit_to_window(self):
        """适应窗口大小"""
        if self.original_pixmap:
            # 计算适应窗口的缩放比例
            label_size = self.size()
            pixmap_size = self.original_pixmap.size()
            
            scale_x = label_size.width() / pixmap_size.width()
            scale_y = label_size.height() / pixmap_size.height()
            
            self.scale_factor = min(scale_x, scale_y) * 0.95  # 留一点边距
            self.pixmap_offset = QPoint(0, 0)
            self.update_display()
    
    def actual_size(self):
        """显示原始大小"""
        self.scale_factor = 1.0
        self.pixmap_offset = QPoint(0, 0)
        self.update_display()
    
    def update_display(self):
        """更新显示"""
        if self.original_pixmap:
            # 计算缩放后的大小
            scaled_size = self.original_pixmap.size() * self.scale_factor
            
            # 创建缩放后的pixmap
            scaled_pixmap = self.original_pixmap.scaled(
                scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            # 创建一个与标签大小相同的pixmap
            display_pixmap = QPixmap(self.size())
            display_pixmap.fill(QColor(240, 240, 240))
            
            # 计算居中位置
            x = (self.width() - scaled_pixmap.width()) // 2 + self.pixmap_offset.x()
            y = (self.height() - scaled_pixmap.height()) // 2 + self.pixmap_offset.y()
            
            # 绘制图像
            painter = QPainter(display_pixmap)
            painter.drawPixmap(x, y, scaled_pixmap)
            painter.end()
            
            self.setPixmap(display_pixmap)
    
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        if self.original_pixmap:
            self.update_display()
    
    def setText(self, text):
        """设置文本（当没有图像时）"""
        self.original_pixmap = None
        super().setText(text)
    
    def clear(self):
        """清空内容"""
        self.original_pixmap = None
        super().clear()
    
    def keyPressEvent(self, event):
        """键盘事件处理"""
        if self.original_pixmap:
            if event.modifiers() == Qt.ControlModifier:
                if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
                    self.zoom_in()
                elif event.key() == Qt.Key_Minus:
                    self.zoom_out()
                elif event.key() == Qt.Key_0:
                    self.fit_to_window()
                elif event.key() == Qt.Key_1:
                    self.actual_size()
        super().keyPressEvent(event)


class Tab1Widget(QWidget):
    """基于图像的异常检测标签页"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.loaded_models = {}  # 存储已加载的模型
        self.current_image_list = []  # 当前图像列表
        self.detection_thread = None
        self.detection_results = {}  # 存储每张图像的检测结果
        self.init_ui()
        
    def init_ui(self):
        """初始化异常检测界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)  # 减少边距
        main_layout.setSpacing(5)  # 减少组件间距
        
        # 标题
        title_label = QLabel("图像异常检测")
        title_font = QFont()
        title_font.setPointSize(14)  # 减小字体大小
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setMaximumHeight(30)  # 限制标题高度
        main_layout.addWidget(title_label)
        
        # 创建主分割器（水平）
        main_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧面板（控制区）
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # 右侧面板（显示区）
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # 设置分割比例
        main_splitter.setSizes([400, 800])
        main_splitter.setStretchFactor(0, 0)  # 左侧面板不拉伸
        main_splitter.setStretchFactor(1, 1)  # 右侧面板可拉伸
        
        main_layout.addWidget(main_splitter)
        
        # 底部进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # 初始加载模型列表（所有UI组件创建完成后）
        self.refresh_model_list()
        
    def create_left_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)  # 减少边距
        layout.setSpacing(8)  # 减少组件间距
        
        # 图像导入区
        image_group = QGroupBox("图像管理")
        image_layout = QVBoxLayout(image_group)
        image_layout.setSpacing(5)  # 减少组内间距
        
        # 导入按钮
        import_layout = QHBoxLayout()
        self.import_image_btn = QPushButton("导入图像")
        self.import_image_btn.clicked.connect(self.import_images)
        self.import_folder_btn = QPushButton("导入文件夹")
        self.import_folder_btn.clicked.connect(self.import_folder)
        import_layout.addWidget(self.import_image_btn)
        import_layout.addWidget(self.import_folder_btn)
        image_layout.addLayout(import_layout)
        
        # 排序选项
        sort_layout = QHBoxLayout()
        sort_layout.addWidget(QLabel("排序:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["文件名", "修改时间", "文件大小", "自定义"])
        self.sort_combo.currentTextChanged.connect(self.sort_images)
        sort_layout.addWidget(self.sort_combo)
        image_layout.addLayout(sort_layout)
        
        # 图像列表
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.image_list.itemClicked.connect(self.on_image_selected)
        self.image_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_list.customContextMenuRequested.connect(self.show_image_context_menu)
        image_layout.addWidget(self.image_list)
        
        layout.addWidget(image_group)
        
        # 模型管理区
        model_group = QGroupBox("模型管理")
        model_layout = QVBoxLayout(model_group)
        
        # 刷新模型按钮
        self.refresh_model_btn = QPushButton("刷新模型列表")
        self.refresh_model_btn.clicked.connect(self.refresh_model_list)
        model_layout.addWidget(self.refresh_model_btn)
        
        # 模型列表（使用复选框）
        self.model_list = QListWidget()
        # 监听复选框状态变化
        self.model_list.itemChanged.connect(self.on_model_selection_changed)
        model_layout.addWidget(self.model_list)
        
        layout.addWidget(model_group)
        
        # 检测控制区
        control_group = QGroupBox("检测控制")
        control_layout = QVBoxLayout(control_group)
        
        # 检测按钮
        self.detect_btn = QPushButton("开始检测")
        self.detect_btn.setEnabled(False)
        self.detect_btn.clicked.connect(self.start_detection)
        self.detect_btn.setMinimumHeight(35)  # 设置最小高度
        self.detect_btn.setStyleSheet("""
            QPushButton {
                min-height: 25px;
            }
            QPushButton:enabled {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:enabled:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        control_layout.addWidget(self.detect_btn)
        
        # 停止按钮
        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setMinimumHeight(35)  # 设置最小高度
        self.stop_btn.setStyleSheet("""
            QPushButton {
                min-height: 20px;
            }
            QPushButton:enabled {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        control_layout.addWidget(self.stop_btn)
        
        # 导出按钮
        self.export_btn = QPushButton("导出结果")
        self.export_btn.setEnabled(False) # 初始不可用
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setStyleSheet("""
            QPushButton {
                min-height: 20px;
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1e88e5;
            }
            QPushButton:disabled {
                 background-color: #cccccc;
                 color: #666666;
            }
        """)
        control_layout.addWidget(self.export_btn)
        
        layout.addWidget(control_group)
        
        layout.addStretch()
        return panel
        
    def create_right_panel(self):
        """创建右侧显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)  # 减少边距
        layout.setSpacing(5)  # 减少组件间距
        
        # 创建垂直分割器
        splitter = QSplitter(Qt.Vertical)
        
        # 图像对比显示区
        image_compare_group = QGroupBox("图像对比")
        image_compare_layout = QVBoxLayout(image_compare_group)
        
        # 创建垂直分割器用于上下显示两个图像
        image_splitter = QSplitter(Qt.Vertical)
        
        # 原始图像显示
        original_frame = QWidget()
        original_layout = QVBoxLayout(original_frame)
        original_layout.setContentsMargins(0, 0, 0, 0)
        original_layout.setSpacing(5)

        original_header_layout = QHBoxLayout()
        original_header_layout.addWidget(QLabel("<b>原始图像</b>"))
        original_header_layout.addStretch()
        fit_original_btn = QPushButton("适应窗口")
        fit_original_btn.setToolTip("将图像调整到适应窗口大小 (Ctrl+0)")
        fit_original_btn.setFixedSize(90, 24)
        original_header_layout.addWidget(fit_original_btn)
        original_layout.addLayout(original_header_layout)
        
        scroll_area_original = QScrollArea()
        self.original_image_label = ZoomableImageLabel()
        fit_original_btn.clicked.connect(self.original_image_label.fit_to_window)
        scroll_area_original.setWidget(self.original_image_label)
        scroll_area_original.setWidgetResizable(True)
        scroll_area_original.setAlignment(Qt.AlignCenter)
        
        # 设置初始提示文本
        self.original_image_label.setText("请导入并选择图像")
        
        original_layout.addWidget(scroll_area_original)
        image_splitter.addWidget(original_frame)
        
        # 检测后图像显示
        detected_frame = QWidget()
        detected_layout = QVBoxLayout(detected_frame)
        detected_layout.setContentsMargins(0, 0, 0, 0)
        detected_layout.setSpacing(5)

        detected_header_layout = QHBoxLayout()
        detected_header_layout.addWidget(QLabel("<b>检测结果图像</b>"))
        detected_header_layout.addStretch()
        fit_detected_btn = QPushButton("适应窗口")
        fit_detected_btn.setToolTip("将图像调整到适应窗口大小 (Ctrl+0)")
        fit_detected_btn.setFixedSize(90, 24)
        detected_header_layout.addWidget(fit_detected_btn)
        detected_layout.addLayout(detected_header_layout)

        scroll_area_detected = QScrollArea()
        self.detected_image_label = ZoomableImageLabel()
        fit_detected_btn.clicked.connect(self.detected_image_label.fit_to_window)
        scroll_area_detected.setWidget(self.detected_image_label)
        scroll_area_detected.setWidgetResizable(True)
        scroll_area_detected.setAlignment(Qt.AlignCenter)
        
        # 设置初始提示文本
        self.detected_image_label.setText("请先选择图像并进行检测")
        
        detected_layout.addWidget(scroll_area_detected)
        image_splitter.addWidget(detected_frame)
        
        # 设置分割比例（上下各占一半）
        image_splitter.setSizes([200, 200])
        
        image_compare_layout.addWidget(image_splitter)
        splitter.addWidget(image_compare_group)
        
        # 结果显示区
        results_group = QGroupBox("检测结果")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "图像", "模型", "异常类型", "置信度", "位置"
        ])
        
        # 设置表格样式
        self.results_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """)
        
        # 设置列宽 - 使用可调整的比例分配
        header = self.results_table.horizontalHeader()
        header.setStretchLastSection(True)  # 拉伸最后一列以填满空间
        header.setSectionResizeMode(QHeaderView.Stretch) # 均匀拉伸所有列
        
        results_layout.addWidget(self.results_table)
        splitter.addWidget(results_group)
        
        # 日志显示区
        log_group = QGroupBox("检测日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        
        splitter.addWidget(log_group)
        
        # 设置分割比例
        splitter.setSizes([400, 200, 100])
        
        layout.addWidget(splitter)
        return panel
        
    def import_images(self):
        """导入图像文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self, 
            "选择图像文件", 
            "", 
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*.*)"
        )
        
        if files:
            for file in files:
                self.add_image_to_list(file)
            self.update_detection_button_state()
            self.log_message(f"已导入 {len(files)} 张图像")
            
    def import_folder(self):
        """导入整个文件夹的图像"""
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        
        if folder:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))
                        
            for file in image_files:
                self.add_image_to_list(file)
                
            self.update_detection_button_state()
            self.log_message(f"从文件夹导入 {len(image_files)} 张图像")
            
    def add_image_to_list(self, image_path):
        """添加图像到列表"""
        if image_path not in self.current_image_list:
            self.current_image_list.append(image_path)
            item = QListWidgetItem(os.path.basename(image_path))
            item.setData(Qt.UserRole, image_path)
            self.image_list.addItem(item)
            
    def sort_images(self, sort_type):
        """对图像列表进行排序"""
        if sort_type == "文件名":
            self.current_image_list.sort(key=lambda x: os.path.basename(x))
        elif sort_type == "修改时间":
            self.current_image_list.sort(key=lambda x: os.path.getmtime(x))
        elif sort_type == "文件大小":
            self.current_image_list.sort(key=lambda x: os.path.getsize(x))
        elif sort_type == "自定义":
            # TODO: 实现自定义排序对话框
            pass
            
        # 更新列表显示
        self.image_list.clear()
        for image_path in self.current_image_list:
            item = QListWidgetItem(os.path.basename(image_path))
            item.setData(Qt.UserRole, image_path)
            self.image_list.addItem(item)
            
    def show_image_context_menu(self, position):
        """显示图像列表的右键菜单"""
        menu = QMenu()
        
        remove_action = QAction("移除选中项", self)
        remove_action.triggered.connect(self.remove_selected_images)
        menu.addAction(remove_action)
        
        clear_action = QAction("清空列表", self)
        clear_action.triggered.connect(self.clear_image_list)
        menu.addAction(clear_action)
        
        menu.exec_(self.image_list.mapToGlobal(position))
        
    def remove_selected_images(self):
        """移除选中的图像"""
        for item in self.image_list.selectedItems():
            image_path = item.data(Qt.UserRole)
            self.current_image_list.remove(image_path)
            # 移除对应的检测结果
            if image_path in self.detection_results:
                del self.detection_results[image_path]
            self.image_list.takeItem(self.image_list.row(item))
        self.update_detection_button_state()
        
    def clear_image_list(self):
        """清空图像列表"""
        self.image_list.clear()
        self.current_image_list.clear()
        self.detection_results.clear()  # 清空检测结果缓存
        self.original_image_label.clear()
        self.detected_image_label.clear()
        self.update_detection_button_state()
        
    def on_image_selected(self, item):
        """处理图像选择事件"""
        image_path = item.data(Qt.UserRole)
        
        # 显示原始图像
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.original_image_label.set_image(pixmap)
            
            # 检查是否有该图像的检测结果
            if image_path in self.detection_results:
                # 显示之前的检测结果
                self.show_detection_result(image_path, self.detection_results[image_path])
            else:
                # 清空检测结果图像
                self.detected_image_label.clear()
                self.detected_image_label.setText("未检测")
            
        self.log_message(f"已选择图像: {os.path.basename(image_path)}")
        
    def refresh_model_list(self):
        """刷新模型列表"""
        from app.services import model_registry
        
        self.model_list.clear()
        self.loaded_models.clear()
        
        try:
            models = model_registry.list_models()
            
            if not models:
                item = QListWidgetItem("未找到可用模型")
                item.setFlags(item.flags() & ~Qt.ItemIsSelectable & ~Qt.ItemIsUserCheckable)
                self.model_list.addItem(item)
                self.log_message("未找到可用模型，请在models/目录下放置模型")
                return
                
            for model in models:
                model_id = model.get('model_id', '')
                model_name = model.get('name', model_id)
                model_type = model.get('model_type', 'unknown')
                
                # 创建带复选框的列表项
                display_text = f"{model_name} [{model_type}]"
                item = QListWidgetItem(display_text)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)  # 默认不选中
                item.setData(Qt.UserRole, model_id)
                
                self.model_list.addItem(item)
                self.loaded_models[model_id] = model
                
            self.update_detection_button_state()
            self.log_message(f"已加载 {len(models)} 个可用模型")
            
        except Exception as e:
            self.log_message(f"加载模型列表失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载模型列表失败: {str(e)}")
            
    def get_selected_models(self):
        """获取选中的模型列表"""
        selected_models = []
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            if item.checkState() == Qt.Checked:
                model_id = item.data(Qt.UserRole)
                if model_id:  # 排除提示项
                    selected_models.append(model_id)
        return selected_models
    
    def on_model_selection_changed(self, item):
        """处理模型选择变化"""
        self.update_detection_button_state()
        
    def update_detection_button_state(self):
        """更新检测按钮状态"""
        has_images = self.image_list.count() > 0
        
        # 计算可用模型数（排除提示项）
        available_models = 0
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            if item.data(Qt.UserRole):  # 有model_id的才是真正的模型
                available_models += 1
        
        has_models = available_models > 0
        has_selected_models = len(self.get_selected_models()) > 0
        
        # 启用条件：有图像、有可用模型、有选中的模型
        can_detect = has_images and has_models and has_selected_models
        self.detect_btn.setEnabled(can_detect)
        
        # 调试信息
        if not can_detect:
            reasons = []
            if not has_images:
                reasons.append("无图像")
            if not has_models:
                reasons.append("无可用模型")
            if not has_selected_models:
                reasons.append("未选择模型")
            self.log_message(f"检测按钮禁用原因: {', '.join(reasons)}")
        
    def start_detection(self):
        """开始检测"""
        
        # 修正：始终检测列表中的所有图像，以避免因自动选中而导致的混淆。
        # 旧逻辑是：如果用户有选中项，则只检测选中项。但这在检测后自动选中第一项时，
        # 会导致后续的"开始检测"操作只处理被自动选中的那一项，不符合用户预期。
        # 当前修改使"开始检测"按钮的行为始终是处理列表中的全部内容。
        selected_images = self.current_image_list
        
        self.last_detected_images = selected_images # 保存本次检测的图像列表
            
        selected_models = self.get_selected_models()
        
        if not selected_images:
            QMessageBox.warning(self, "警告", "请先导入图像")
            return
            
        if not selected_models:
            QMessageBox.warning(self, "警告", "请至少选择一个模型")
            return
            
        # 更新UI状态
        self.detect_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 清空之前的结果
        self.results_table.setRowCount(0)
        self.export_btn.setEnabled(False) # 检测期间禁用导出
        
        # 清空选中图像的旧检测结果
        for image_path in selected_images:
            if image_path in self.detection_results:
                del self.detection_results[image_path]
        
        # 创建并启动检测线程
        self.detection_thread = ImageDetectionThread(selected_images, selected_models)
        self.detection_thread.progress_updated.connect(self.update_progress)
        self.detection_thread.detection_finished.connect(self.on_detection_finished)
        self.detection_thread.log_message.connect(self.log_message)
        self.detection_thread.start()
        
        self.log_message(f"开始检测 {len(selected_images)} 张图像，使用 {len(selected_models)} 个模型")
        
    def stop_detection(self):
        """停止检测"""
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread.wait()
            
        self.detect_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        self.log_message("检测已停止")
        
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        
    def show_detection_result(self, image_path, detections):
        """显示检测结果图像"""
        from app.services import detection_service
        
        try:
            # 获取带有检测结果的图像
            detected_image = detection_service.draw_detections_on_image(
                image_path, 
                detections,
                line_thickness=40, # 设置为更明显的值
                font_size=200       # 设置为更明显的值
            )
            
            # 将numpy数组转换为QPixmap
            height, width, channel = detected_image.shape
            bytes_per_line = 3 * width
            
            # 确保数据是连续的
            if not detected_image.flags['C_CONTIGUOUS']:
                detected_image = np.ascontiguousarray(detected_image)
            
            # 创建QImage
            q_image = QImage(detected_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # 复制数据以确保稳定性
            q_image = q_image.copy()
            detected_pixmap = QPixmap.fromImage(q_image)
            
            # 显示检测结果图像
            self.detected_image_label.set_image(detected_pixmap)
            
        except Exception as e:
            self.log_message(f"显示检测结果失败: {str(e)}")
            self.detected_image_label.setText("显示失败")
    
    def on_detection_finished(self, results):
        """处理检测完成的结果"""
        from app.services import detection_service
        
        image_path = results['image_path']
        image_name = results['image_name']
        
        # 保存检测结果
        self.detection_results[image_path] = results['detections']
        
        # 如果当前显示的是这张图像，更新显示
        current_item = self.image_list.currentItem()
        if current_item and current_item.data(Qt.UserRole) == image_path:
            # 显示原始图像
            original_pixmap = QPixmap(image_path)
            if not original_pixmap.isNull():
                self.original_image_label.set_image(original_pixmap)
            
            # 显示检测结果
            self.show_detection_result(image_path, results['detections'])
        
        # 更新结果表格
        for detection in results['detections']:
            model_name = detection.get('model_name', detection.get('model_id', 'Unknown'))
            anomalies = detection.get('anomalies', [])
            
            if anomalies:
                for anomaly in anomalies:
                    row_position = self.results_table.rowCount()
                    self.results_table.insertRow(row_position)
                    
                    self.results_table.setItem(row_position, 0, QTableWidgetItem(image_name))
                    self.results_table.setItem(row_position, 1, QTableWidgetItem(model_name))
                    self.results_table.setItem(row_position, 2, QTableWidgetItem(anomaly['type']))
                    self.results_table.setItem(row_position, 3, QTableWidgetItem(f"{anomaly['confidence']:.2f}"))
                    
                    bbox = anomaly['bbox']
                    location = f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"
                    self.results_table.setItem(row_position, 4, QTableWidgetItem(location))
            else:
                # 没有检测到异常
                row_position = self.results_table.rowCount()
                self.results_table.insertRow(row_position)
                
                self.results_table.setItem(row_position, 0, QTableWidgetItem(image_name))
                self.results_table.setItem(row_position, 1, QTableWidgetItem(model_name))
                self.results_table.setItem(row_position, 2, QTableWidgetItem("无异常"))
                self.results_table.setItem(row_position, 3, QTableWidgetItem("-"))
                self.results_table.setItem(row_position, 4, QTableWidgetItem("-"))
                
        # 检查是否所有检测都完成
        if self.progress_bar.value() == 100:
            self.detect_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
            self.log_message("所有图像检测完成")

            # 自动显示第一张图像的结果
            if self.current_image_list:
                first_image_path = self.current_image_list[0]
                # 找到对应的QListWidgetItem并选中
                for i in range(self.image_list.count()):
                    item = self.image_list.item(i)
                    if item.data(Qt.UserRole) == first_image_path:
                        self.image_list.setCurrentItem(item)
                        self.on_image_selected(item) # 手动触发选中事件
                        break
            
            # 启用导出按钮
            if self.detection_results:
                self.export_btn.setEnabled(True)
            
    def log_message(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def on_column_resized(self, logical_index, old_size, new_size):
        """此方法不再需要，因为列宽是自动均匀分布的"""
        pass
    
    def update_column_widths(self):
        """此方法不再需要，因为列宽是自动均匀分布的"""
        pass
    
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        # 此处不再需要调用更新列宽的函数

    def export_results(self):
        """导出检测结果和图像"""
        if not self.detection_results:
            QMessageBox.information(self, "提示", "没有可导出的检测结果。")
            return

        # 选择导出目录
        export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not export_dir:
            return

        try:
            # 1. 导出JSON结果
            json_path = os.path.join(export_dir, "detection_results.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                # 准备一个更适合序列化的版本
                exportable_results = {}
                for img_path, detections in self.detection_results.items():
                    exportable_results[img_path] = detections
                json.dump(exportable_results, f, ensure_ascii=False, indent=4)
            self.log_message(f"检测结果已导出至: {json_path}")

            # 2. 导出带标注的图像
            images_dir = os.path.join(export_dir, "detected_images")
            os.makedirs(images_dir, exist_ok=True)
            
            from app.services import detection_service

            exported_count = 0
            for image_path, detections in self.detection_results.items():
                if not detections: # 如果某张图没有检测结果，则跳过
                    continue

                try:
                    base_name = os.path.basename(image_path)
                    output_path = os.path.join(images_dir, f"detected_{base_name}")
                    
                    # 调用服务进行绘制并直接保存
                    detection_service.draw_detections_on_image(
                        image_path,
                        detections,
                        output_path=output_path,
                        line_thickness=10,
                        font_size=50
                    )
                    exported_count += 1
                except Exception as e:
                    self.log_message(f"导出图像 {image_path} 失败: {e}")

            self.log_message(f"已成功导出 {exported_count} 张标注图像至: {images_dir}")
            QMessageBox.information(self, "导出成功", f"结果已成功导出到目录:\n{export_dir}")

        except Exception as e:
            self.log_message(f"导出失败: {e}")
            QMessageBox.critical(self, "导出失败", f"导出过程中发生错误: {e}")
