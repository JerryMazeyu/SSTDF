from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QGroupBox, QSplitter, QHeaderView, QFileDialog,
                             QListWidget, QListWidgetItem, QComboBox, 
                             QCheckBox, QProgressBar, QMessageBox, QMenu,
                             QAction, QAbstractItemView, QTextEdit, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QFont, QPixmap, QPainter, QPen, QColor, QBrush
import logging
from datetime import datetime
import os
import json
from pathlib import Path


class ImageDetectionThread(QThread):
    """图像检测线程"""
    progress_updated = pyqtSignal(int)
    detection_finished = pyqtSignal(dict)
    log_message = pyqtSignal(str)
    
    def __init__(self, images, models):
        super().__init__()
        self.images = images
        self.models = models
        self.is_running = True
        
    def run(self):
        """执行检测任务"""
        total_images = len(self.images)
        for idx, image_path in enumerate(self.images):
            if not self.is_running:
                break
                
            # 模拟检测过程
            self.log_message.emit(f"正在检测图像: {os.path.basename(image_path)}")
            
            # TODO: 调用实际的检测服务
            # results = detection_service.detect(image_path, self.models)
            
            # 模拟检测结果
            import random
            import time
            time.sleep(0.5)  # 模拟处理时间
            
            results = {
                'image_path': image_path,
                'detections': [
                    {
                        'model': model,
                        'anomalies': [
                            {
                                'type': '表面缺陷',
                                'confidence': random.uniform(0.8, 0.99),
                                'bbox': [random.randint(100, 300), random.randint(50, 150), 
                                       random.randint(50, 100), random.randint(50, 100)]
                            }
                        ] if random.random() > 0.7 else []
                    } for model in self.models
                ]
            }
            
            self.detection_finished.emit(results)
            progress = int((idx + 1) / total_images * 100)
            self.progress_updated.emit(progress)
            
    def stop(self):
        """停止检测"""
        self.is_running = False


class ImageLabel(QLabel):
    """自定义图像标签，支持绘制检测结果"""
    
    def __init__(self):
        super().__init__()
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
        self.detection_results = []
        self.original_pixmap = None
        
    def set_image(self, image_path):
        """设置显示的图像"""
        self.original_pixmap = QPixmap(image_path)
        self.update_display()
        
    def set_detection_results(self, results):
        """设置检测结果"""
        self.detection_results = results
        self.update_display()
        
    def update_display(self):
        """更新显示"""
        if self.original_pixmap:
            # 获取标签的大小
            label_size = self.size()
            
            # 计算缩放比例，保持宽高比
            pixmap = self.original_pixmap.scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            # 如果有检测结果，绘制边界框
            if self.detection_results:
                painter = QPainter(pixmap)
                
                for result in self.detection_results:
                    for anomaly in result.get('anomalies', []):
                        # 绘制边界框
                        bbox = anomaly['bbox']
                        confidence = anomaly['confidence']
                        
                        # 根据置信度设置颜色
                        if confidence > 0.9:
                            color = QColor(255, 0, 0)  # 红色
                        elif confidence > 0.7:
                            color = QColor(255, 165, 0)  # 橙色
                        else:
                            color = QColor(255, 255, 0)  # 黄色
                            
                        pen = QPen(color, 2)
                        painter.setPen(pen)
                        
                        # 计算缩放后的坐标
                        scale = pixmap.width() / self.original_pixmap.width()
                        x = int(bbox[0] * scale)
                        y = int(bbox[1] * scale)
                        w = int(bbox[2] * scale)
                        h = int(bbox[3] * scale)
                        
                        painter.drawRect(x, y, w, h)
                        
                        # 绘制标签背景
                        label_text = f"{anomaly['type']} ({confidence:.2f})"
                        painter.fillRect(x, y - 20, len(label_text) * 8, 20, QBrush(color))
                        
                        # 绘制文字
                        painter.setPen(QPen(Qt.white))
                        painter.drawText(x + 2, y - 5, label_text)
                        
                painter.end()
            
            self.setPixmap(pixmap)
            
    def resizeEvent(self, event):
        """窗口大小改变时更新图像"""
        super().resizeEvent(event)
        if self.original_pixmap:
            self.update_display()


class Tab1Widget(QWidget):
    """基于图像的异常检测标签页"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.loaded_models = {}  # 存储已加载的模型
        self.current_image_list = []  # 当前图像列表
        self.detection_thread = None
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
        
        # 导入模型按钮
        self.import_model_btn = QPushButton("导入模型")
        self.import_model_btn.clicked.connect(self.import_model)
        model_layout.addWidget(self.import_model_btn)
        
        # 模型列表（使用复选框）
        self.model_list = QListWidget()
        model_layout.addWidget(self.model_list)
        
        layout.addWidget(model_group)
        
        # 检测控制区
        control_group = QGroupBox("检测控制")
        control_layout = QVBoxLayout(control_group)
        
        # 检测按钮
        self.detect_btn = QPushButton("开始检测")
        self.detect_btn.setEnabled(False)
        self.detect_btn.clicked.connect(self.start_detection)
        self.detect_btn.setStyleSheet("""
            QPushButton:enabled {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
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
        self.stop_btn.setStyleSheet("""
            QPushButton:enabled {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        control_layout.addWidget(self.stop_btn)
        
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
        
        # 图像显示区
        image_group = QGroupBox("图像显示")
        image_layout = QVBoxLayout(image_group)
        
        # 使用滚动区域包装图像标签
        scroll_area = QScrollArea()
        self.image_label = ImageLabel()
        self.image_label.setMinimumSize(QSize(600, 200))  # 设置最小尺寸
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignCenter)
        
        image_layout.addWidget(scroll_area)
        splitter.addWidget(image_group)
        
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
        
        # 设置列宽
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        
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
            self.image_list.takeItem(self.image_list.row(item))
        self.update_detection_button_state()
        
    def clear_image_list(self):
        """清空图像列表"""
        self.image_list.clear()
        self.current_image_list.clear()
        self.image_label.clear()
        self.update_detection_button_state()
        
    def on_image_selected(self, item):
        """处理图像选择事件"""
        image_path = item.data(Qt.UserRole)
        self.image_label.set_image(image_path)
        self.log_message(f"已选择图像: {os.path.basename(image_path)}")
        
    def import_model(self):
        """导入检测模型"""
        file, _ = QFileDialog.getOpenFileName(
            self, 
            "选择模型文件", 
            "", 
            "模型文件 (*.pth *.pt *.onnx *.pb);;所有文件 (*.*)"
        )
        
        if file:
            model_name = os.path.basename(file)
            
            # 创建带复选框的列表项
            item = QListWidgetItem(model_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, file)
            
            self.model_list.addItem(item)
            self.loaded_models[model_name] = file
            
            self.update_detection_button_state()
            self.log_message(f"已导入模型: {model_name}")
            
    def get_selected_models(self):
        """获取选中的模型列表"""
        selected_models = []
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_models.append(item.text())
        return selected_models
        
    def update_detection_button_state(self):
        """更新检测按钮状态"""
        has_images = self.image_list.count() > 0
        has_models = self.model_list.count() > 0
        has_selected_models = len(self.get_selected_models()) > 0
        
        self.detect_btn.setEnabled(has_images and has_models and has_selected_models)
        
    def start_detection(self):
        """开始检测"""
        selected_images = []
        
        # 获取选中的图像
        if self.image_list.selectedItems():
            for item in self.image_list.selectedItems():
                selected_images.append(item.data(Qt.UserRole))
        else:
            # 如果没有选中，检测所有图像
            selected_images = self.current_image_list
            
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
        
    def on_detection_finished(self, results):
        """处理检测完成的结果"""
        image_path = results['image_path']
        image_name = os.path.basename(image_path)
        
        # 如果当前显示的是这张图像，更新显示
        current_item = self.image_list.currentItem()
        if current_item and current_item.data(Qt.UserRole) == image_path:
            # 合并所有模型的检测结果
            all_anomalies = []
            for detection in results['detections']:
                all_anomalies.extend(detection['anomalies'])
            self.image_label.set_detection_results(all_anomalies)
        
        # 更新结果表格
        for detection in results['detections']:
            model_name = detection['model']
            anomalies = detection['anomalies']
            
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
            
    def log_message(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
