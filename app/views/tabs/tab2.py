from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QGroupBox, QSplitter, QListWidget,
                             QListWidgetItem, QComboBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QGridLayout,
                             QProgressBar, QTextEdit, QCheckBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QPixmap, QPainter, QColor, QBrush, QPen
import pyqtgraph as pg
import numpy as np
import logging
from datetime import datetime
import psutil
import random


class ResourceMonitorThread(QThread):
    """资源监控线程"""
    cpu_updated = pyqtSignal(float)
    gpu_updated = pyqtSignal(list)  # [GPU使用率, 显存使用率]
    memory_updated = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        self.is_running = True
        
    def run(self):
        """持续监控系统资源"""
        while self.is_running:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_updated.emit(cpu_percent)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            self.memory_updated.emit(memory.percent)
            
            # GPU使用率（模拟，实际需要使用pynvml或其他GPU监控库）
            try:
                # import pynvml
                # pynvml.nvmlInit()
                # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                # gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                # memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                # gpu_percent = gpu_util.gpu
                # memory_percent = (memory_info.used / memory_info.total) * 100
                
                # 模拟GPU数据
                gpu_percent = random.uniform(20, 80)
                memory_percent = random.uniform(30, 70)
                self.gpu_updated.emit([gpu_percent, memory_percent])
            except:
                self.gpu_updated.emit([0, 0])
                
            self.msleep(1000)  # 每秒更新一次
            
    def stop(self):
        """停止监控"""
        self.is_running = False


class ModelMonitorThread(QThread):
    """模型监控线程"""
    feature_map_updated = pyqtSignal(dict)
    model_status_updated = pyqtSignal(str, dict)  # 模型名称, 状态信息
    log_message = pyqtSignal(str)
    
    def __init__(self, model_name, model_path=None):
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.is_running = True
        
    def run(self):
        """监控模型运行状态"""
        while self.is_running:
            # 模拟获取特征图
            # 实际应该调用模型服务获取真实的特征图
            feature_maps = self.simulate_feature_maps()
            self.feature_map_updated.emit(feature_maps)
            
            # 模拟模型状态
            status = {
                'running': True,
                'fps': random.uniform(25, 35),
                'latency': random.uniform(10, 50),
                'accuracy': random.uniform(0.85, 0.99),
                'error_rate': random.uniform(0, 0.05)
            }
            self.model_status_updated.emit(self.model_name, status)
            
            self.msleep(2000)  # 每2秒更新一次
            
    def simulate_feature_maps(self):
        """模拟生成特征图"""
        # 实际应该从模型中提取真实的特征图
        feature_maps = {}
        layers = ['conv1', 'conv2', 'conv3', 'fc1']
        
        for layer in layers:
            # 生成随机特征图数据
            if 'conv' in layer:
                # 卷积层特征图
                feature_map = np.random.randn(64, 64) * 255
            else:
                # 全连接层特征
                feature_map = np.random.randn(16, 16) * 255
                
            feature_maps[layer] = feature_map.astype(np.uint8)
            
        return feature_maps
        
    def stop(self):
        """停止监控"""
        self.is_running = False


class FeatureMapWidget(QWidget):
    """特征图显示组件"""
    
    def __init__(self):
        super().__init__()
        self.feature_maps = {}
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        layout = QGridLayout(self)
        layout.setSpacing(10)
        
        # 创建特征图显示标签
        self.feature_labels = {}
        self.layer_names = ['conv1', 'conv2', 'conv3', 'fc1']
        
        for i, layer_name in enumerate(self.layer_names):
            row = i // 2
            col = i % 2
            
            # 创建组框
            group = QGroupBox(f"Layer: {layer_name}")
            group_layout = QVBoxLayout(group)
            
            # 创建图像标签
            label = QLabel()
            label.setMinimumSize(200, 200)
            label.setScaledContents(True)
            label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
            label.setAlignment(Qt.AlignCenter)
            
            group_layout.addWidget(label)
            layout.addWidget(group, row, col)
            
            self.feature_labels[layer_name] = label
            
    def update_feature_maps(self, feature_maps):
        """更新特征图显示"""
        self.feature_maps = feature_maps
        
        for layer_name, feature_map in feature_maps.items():
            if layer_name in self.feature_labels:
                # 将numpy数组转换为QPixmap
                height, width = feature_map.shape
                
                # 归一化到0-255
                normalized = ((feature_map - feature_map.min()) / 
                             (feature_map.max() - feature_map.min() + 1e-8) * 255).astype(np.uint8)
                
                # 创建QImage
                from PyQt5.QtGui import QImage
                
                # 转换为RGB格式
                rgb_array = np.stack([normalized] * 3, axis=2)
                
                image = QImage(rgb_array.data, width, height, 3 * width, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(image)
                
                # 缩放到标签大小
                scaled_pixmap = pixmap.scaled(
                    self.feature_labels[layer_name].size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                
                self.feature_labels[layer_name].setPixmap(scaled_pixmap)


class Tab2Widget(QWidget):
    """哨兵系统 - 模型运行状态监控"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.resource_monitor_thread = None
        self.model_monitor_threads = {}  # 存储多个模型监控线程
        self.resource_history = {'cpu': [], 'gpu': [], 'memory': []}
        self.max_history_points = 60  # 保存60个数据点
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)  # 减少边距
        main_layout.setSpacing(5)  # 减少组件间距
        
        # 标题
        title_label = QLabel("哨兵系统 - 模型运行状态监控")
        title_font = QFont()
        title_font.setPointSize(14)  # 减小字体大小
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setMaximumHeight(30)  # 限制标题高度
        main_layout.addWidget(title_label)
        
        # 创建主分割器
        main_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧面板（模型选择和控制）
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # 右侧面板（监控显示）
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # 设置分割比例
        main_splitter.setSizes([300, 900])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(main_splitter)
        
        # 启动资源监控
        self.start_resource_monitoring()
        
    def create_left_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)  # 减少边距
        layout.setSpacing(8)  # 减少组件间距
        
        # 运行中的模型
        running_group = QGroupBox("运行中的模型")
        running_layout = QVBoxLayout(running_group)
        running_layout.setSpacing(5)  # 减少组内间距
        
        self.running_models_list = QListWidget()
        self.running_models_list.itemClicked.connect(self.on_model_selected)
        running_layout.addWidget(self.running_models_list)
        
        # 模拟添加一些运行中的模型
        running_models = ["YOLO-v5", "ResNet-50", "EfficientNet-B0"]
        for model in running_models:
            item = QListWidgetItem(f"🟢 {model}")
            item.setData(Qt.UserRole, {'name': model, 'status': 'running'})
            self.running_models_list.addItem(item)
        
        layout.addWidget(running_group)
        
        # 可用模型
        available_group = QGroupBox("可用模型")
        available_layout = QVBoxLayout(available_group)
        
        self.available_models_list = QListWidget()
        self.available_models_list.itemClicked.connect(self.on_model_selected)
        available_layout.addWidget(self.available_models_list)
        
        # 模拟添加一些可用模型
        available_models = ["MobileNet-v2", "VGG-16", "Inception-v3"]
        for model in available_models:
            item = QListWidgetItem(f"⚪ {model}")
            item.setData(Qt.UserRole, {'name': model, 'status': 'stopped'})
            self.available_models_list.addItem(item)
        
        layout.addWidget(available_group)
        
        # 控制按钮
        control_group = QGroupBox("控制")
        control_layout = QVBoxLayout(control_group)
        
        self.monitor_btn = QPushButton("开始监控")
        self.monitor_btn.setEnabled(False)
        self.monitor_btn.clicked.connect(self.toggle_model_monitoring)
        self.monitor_btn.setStyleSheet("""
            QPushButton:enabled {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:enabled:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        control_layout.addWidget(self.monitor_btn)
        
        self.refresh_btn = QPushButton("刷新模型列表")
        self.refresh_btn.clicked.connect(self.refresh_model_list)
        control_layout.addWidget(self.refresh_btn)
        
        layout.addWidget(control_group)
        
        layout.addStretch()
        return panel
        
    def create_right_panel(self):
        """创建右侧监控显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)  # 减少边距
        layout.setSpacing(5)  # 减少组件间距
        
        # 创建垂直分割器
        splitter = QSplitter(Qt.Vertical)
        
        # 上部分：特征图显示
        feature_group = QGroupBox("模型特征图")
        feature_layout = QVBoxLayout(feature_group)
        
        # 当前监控的模型名称
        self.current_model_label = QLabel("请选择要监控的模型")
        self.current_model_label.setAlignment(Qt.AlignCenter)
        feature_layout.addWidget(self.current_model_label)
        
        # 特征图显示组件
        self.feature_map_widget = FeatureMapWidget()
        feature_layout.addWidget(self.feature_map_widget)
        
        splitter.addWidget(feature_group)
        
        # 中部分：资源监控
        resource_group = QGroupBox("系统资源监控")
        resource_layout = QVBoxLayout(resource_group)
        
        # 创建资源监控图表
        self.create_resource_charts(resource_layout)
        
        splitter.addWidget(resource_group)
        
        # 下部分：模型状态信息
        status_group = QGroupBox("模型状态信息")
        status_layout = QVBoxLayout(status_group)
        
        self.status_table = QTableWidget()
        self.status_table.setColumnCount(5)
        self.status_table.setHorizontalHeaderLabels([
            "模型名称", "状态", "FPS", "延迟(ms)", "准确率"
        ])
        
        header = self.status_table.horizontalHeader()
        for i in range(5):
            header.setSectionResizeMode(i, QHeaderView.Stretch)
            
        status_layout.addWidget(self.status_table)
        
        splitter.addWidget(status_group)
        
        # 设置分割比例
        splitter.setSizes([400, 250, 150])
        
        layout.addWidget(splitter)
        return panel
        
    def create_resource_charts(self, parent_layout):
        """创建资源监控图表"""
        # 使用pyqtgraph创建实时图表
        chart_layout = QHBoxLayout()
        
        # CPU使用率图表
        cpu_widget = pg.PlotWidget(title="CPU使用率 (%)")
        cpu_widget.setLabel('left', '使用率', units='%')
        cpu_widget.setLabel('bottom', '时间', units='s')
        cpu_widget.setYRange(0, 100)
        cpu_widget.addLegend()
        
        self.cpu_curve = cpu_widget.plot(pen='y', name='CPU')
        chart_layout.addWidget(cpu_widget)
        
        # GPU使用率图表
        gpu_widget = pg.PlotWidget(title="GPU使用率 (%)")
        gpu_widget.setLabel('left', '使用率', units='%')
        gpu_widget.setLabel('bottom', '时间', units='s')
        gpu_widget.setYRange(0, 100)
        gpu_widget.addLegend()
        
        self.gpu_curve = gpu_widget.plot(pen='g', name='GPU')
        self.gpu_memory_curve = gpu_widget.plot(pen='r', name='显存')
        chart_layout.addWidget(gpu_widget)
        
        # 内存使用率图表
        memory_widget = pg.PlotWidget(title="内存使用率 (%)")
        memory_widget.setLabel('left', '使用率', units='%')
        memory_widget.setLabel('bottom', '时间', units='s')
        memory_widget.setYRange(0, 100)
        memory_widget.addLegend()
        
        self.memory_curve = memory_widget.plot(pen='b', name='内存')
        chart_layout.addWidget(memory_widget)
        
        parent_layout.addLayout(chart_layout)
        
        # 添加当前值显示
        values_layout = QHBoxLayout()
        
        self.cpu_label = QLabel("CPU: 0%")
        self.cpu_label.setAlignment(Qt.AlignCenter)
        values_layout.addWidget(self.cpu_label)
        
        self.gpu_label = QLabel("GPU: 0% / 显存: 0%")
        self.gpu_label.setAlignment(Qt.AlignCenter)
        values_layout.addWidget(self.gpu_label)
        
        self.memory_label = QLabel("内存: 0%")
        self.memory_label.setAlignment(Qt.AlignCenter)
        values_layout.addWidget(self.memory_label)
        
        parent_layout.addLayout(values_layout)
        
    def start_resource_monitoring(self):
        """启动资源监控"""
        self.resource_monitor_thread = ResourceMonitorThread()
        self.resource_monitor_thread.cpu_updated.connect(self.update_cpu)
        self.resource_monitor_thread.gpu_updated.connect(self.update_gpu)
        self.resource_monitor_thread.memory_updated.connect(self.update_memory)
        self.resource_monitor_thread.start()
        
    def update_cpu(self, value):
        """更新CPU使用率"""
        self.cpu_label.setText(f"CPU: {value:.1f}%")
        
        # 更新历史数据
        self.resource_history['cpu'].append(value)
        if len(self.resource_history['cpu']) > self.max_history_points:
            self.resource_history['cpu'].pop(0)
            
        # 更新图表
        x = list(range(len(self.resource_history['cpu'])))
        self.cpu_curve.setData(x, self.resource_history['cpu'])
        
    def update_gpu(self, values):
        """更新GPU使用率"""
        gpu_percent, memory_percent = values
        self.gpu_label.setText(f"GPU: {gpu_percent:.1f}% / 显存: {memory_percent:.1f}%")
        
        # 更新历史数据
        if 'gpu_util' not in self.resource_history:
            self.resource_history['gpu_util'] = []
            self.resource_history['gpu_memory'] = []
            
        self.resource_history['gpu_util'].append(gpu_percent)
        self.resource_history['gpu_memory'].append(memory_percent)
        
        if len(self.resource_history['gpu_util']) > self.max_history_points:
            self.resource_history['gpu_util'].pop(0)
            self.resource_history['gpu_memory'].pop(0)
            
        # 更新图表
        x = list(range(len(self.resource_history['gpu_util'])))
        self.gpu_curve.setData(x, self.resource_history['gpu_util'])
        self.gpu_memory_curve.setData(x, self.resource_history['gpu_memory'])
        
    def update_memory(self, value):
        """更新内存使用率"""
        self.memory_label.setText(f"内存: {value:.1f}%")
        
        # 更新历史数据
        self.resource_history['memory'].append(value)
        if len(self.resource_history['memory']) > self.max_history_points:
            self.resource_history['memory'].pop(0)
            
        # 更新图表
        x = list(range(len(self.resource_history['memory'])))
        self.memory_curve.setData(x, self.resource_history['memory'])
        
    def on_model_selected(self, item):
        """处理模型选择事件"""
        model_data = item.data(Qt.UserRole)
        self.selected_model = model_data['name']
        self.monitor_btn.setEnabled(True)
        
        # 清除其他列表的选择
        if self.running_models_list.currentItem() == item:
            self.available_models_list.clearSelection()
        else:
            self.running_models_list.clearSelection()
            
        self.logger.info(f"选择了模型: {self.selected_model}")
        
    def toggle_model_monitoring(self):
        """开始/停止模型监控"""
        if not hasattr(self, 'selected_model'):
            return
            
        if self.selected_model in self.model_monitor_threads:
            # 停止监控
            thread = self.model_monitor_threads[self.selected_model]
            thread.stop()
            thread.wait()
            del self.model_monitor_threads[self.selected_model]
            
            self.monitor_btn.setText("开始监控")
            self.current_model_label.setText("请选择要监控的模型")
            
            # 清除该模型的状态信息
            for row in range(self.status_table.rowCount()):
                if self.status_table.item(row, 0).text() == self.selected_model:
                    self.status_table.removeRow(row)
                    break
                    
        else:
            # 开始监控
            thread = ModelMonitorThread(self.selected_model)
            thread.feature_map_updated.connect(self.update_feature_maps)
            thread.model_status_updated.connect(self.update_model_status)
            thread.log_message.connect(self.log_message)
            thread.start()
            
            self.model_monitor_threads[self.selected_model] = thread
            self.monitor_btn.setText("停止监控")
            self.current_model_label.setText(f"正在监控: {self.selected_model}")
            
    def update_feature_maps(self, feature_maps):
        """更新特征图显示"""
        self.feature_map_widget.update_feature_maps(feature_maps)
        
    def update_model_status(self, model_name, status):
        """更新模型状态信息"""
        # 查找是否已存在该模型的行
        row_exists = False
        for row in range(self.status_table.rowCount()):
            if self.status_table.item(row, 0).text() == model_name:
                row_exists = True
                break
                
        if not row_exists:
            row = self.status_table.rowCount()
            self.status_table.insertRow(row)
        
        # 更新状态信息
        self.status_table.setItem(row, 0, QTableWidgetItem(model_name))
        
        status_text = "运行中" if status['running'] else "已停止"
        status_item = QTableWidgetItem(status_text)
        if status['running']:
            status_item.setBackground(QColor(76, 175, 80))  # 绿色
        else:
            status_item.setBackground(QColor(244, 67, 54))  # 红色
        status_item.setForeground(Qt.white)
        self.status_table.setItem(row, 1, status_item)
        
        self.status_table.setItem(row, 2, QTableWidgetItem(f"{status['fps']:.1f}"))
        self.status_table.setItem(row, 3, QTableWidgetItem(f"{status['latency']:.1f}"))
        self.status_table.setItem(row, 4, QTableWidgetItem(f"{status['accuracy']:.2%}"))
        
    def refresh_model_list(self):
        """刷新模型列表"""
        # TODO: 从后端服务获取最新的模型列表
        self.logger.info("刷新模型列表")
        
    def log_message(self, message):
        """记录日志"""
        self.logger.info(message)
        
    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        # 停止资源监控线程
        if self.resource_monitor_thread:
            self.resource_monitor_thread.stop()
            self.resource_monitor_thread.wait()
            
        # 停止所有模型监控线程
        for thread in self.model_monitor_threads.values():
            thread.stop()
            thread.wait()
            
        event.accept()
