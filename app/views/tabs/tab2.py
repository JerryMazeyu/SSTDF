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
from utils.ssh_client import SSHClient

class ResourceMonitorThread(QThread):
    """资源监控线程"""
    cpu_updated = pyqtSignal(float)
    gpu_updated = pyqtSignal(list)  # [GPU索引, GPU使用率, 显存使用率]
    memory_updated = pyqtSignal(float)
    gpu_count_updated = pyqtSignal(int)  # 新增信号，用于通知GPU数量
    
    def __init__(self):
        super().__init__()
        self.is_running = True
        self.ssh_client = SSHClient()
        self.ssh_client.connect()
        self.current_gpu_index = 0  # 当前选中的GPU索引
        
    def run(self):
        """持续监控系统资源"""
        while self.is_running:
            # 获取远程服务器硬件资源
            remote_hardware_resources = self.ssh_client.get_remote_server_hardware_resources()
            # CPU使用率
            cpu_percent = remote_hardware_resources['cpu_usage']
            self.cpu_updated.emit(cpu_percent)
            
            # 内存使用率
            memory_percent = remote_hardware_resources['memory_usage']
            self.memory_updated.emit(memory_percent)
            
            # 获取GPU数量并通知UI
            gpu_count = len(remote_hardware_resources['gpu_info'])
            self.gpu_count_updated.emit(gpu_count)
            
            # 如果当前GPU索引超出范围，重置为0
            if self.current_gpu_index >= gpu_count:
                self.current_gpu_index = 0
                
            # 发送当前选中GPU的信息
            if gpu_count > 0:
                gpu_info = remote_hardware_resources['gpu_info'][self.current_gpu_index]
                gpu_percent = gpu_info['gpu_load']
                gpu_memory_percent = gpu_info['gpu_memory_used']
                # 作为列表发送数据
                self.gpu_updated.emit([self.current_gpu_index, gpu_percent, gpu_memory_percent])
                
            self.msleep(1000)  # 每秒更新一次
            
    def set_gpu_index(self, index):
        """设置要监控的GPU索引"""
        self.current_gpu_index = index
            
    def stop(self):
        """停止监控"""
        self.is_running = False
        self.ssh_client.close()

class ModelMonitorThread(QThread):
    """模型监控线程"""
    feature_map_updated = pyqtSignal(dict)
    model_status_updated = pyqtSignal(str, dict)  # 模型名称, 状态信息
    log_message = pyqtSignal(str)
    
    def __init__(self, model_name, model_path=None):
        super().__init__()
        self.model_name = model_name
        self.model_id = model_name  # 使用模型名称作为模型ID
        self.is_running = True
        self.ssh_client = SSHClient()
        self.ssh_client.connect()
        
    def run(self):
        """监控模型运行状态"""
        while self.is_running:
            try:
                # 获取远程服务器上的特征图
                feature_maps = self.fetch_feature_maps()
                self.feature_map_updated.emit(feature_maps)
                
                # 模拟模型状态（实际应该从服务端获取）
                status = {
                    'running': True,
                    'fps': random.uniform(25, 35),
                    'latency': random.uniform(10, 50),
                    'accuracy': random.uniform(0.85, 0.99),
                    'error_rate': random.uniform(0, 0.05)
                }
                self.model_status_updated.emit(self.model_name, status)
                
            except Exception as e:
                self.log_message.emit(f"获取特征图失败: {str(e)}")
                # 发送空特征图数据
                self.feature_map_updated.emit({'backbone': None, 'neck': None})
                
            self.msleep(5000)  # 每3秒更新一次
        
    def fetch_feature_maps(self):
        """从远程服务器获取特征图"""
        feature_maps = {'backbone': None, 'neck': None}
        
        # 特征图路径
        backbone_path = f"/home/zentek/MMDetection/api_server/monitoring/features_map/{self.model_id}/backbone.png"
        neck_path = f"/home/zentek/MMDetection/api_server/monitoring/features_map/{self.model_id}/neck.png"
        
        # 临时保存路径
        local_backbone_path = f"temp_backbone_{self.model_id}.png"
        local_neck_path = f"temp_neck_{self.model_id}.png"
        
        try:
            # 尝试通过SFTP从远程服务器下载特征图
            # backbone特征图
            try:
                self.ssh_client.sftp.get(backbone_path, local_backbone_path)
                pixmap = QPixmap(local_backbone_path)
                if not pixmap.isNull():
                    feature_maps['backbone'] = pixmap
                    self.log_message.emit(f"获取backbone特征图成功")
                else:
                    self.log_message.emit(f"backbone特征图加载失败")
            except Exception as e:
                self.log_message.emit(f"未找到backbone特征图: {str(e)}")
                
            # neck特征图
            try:
                self.ssh_client.sftp.get(neck_path, local_neck_path)
                pixmap = QPixmap(local_neck_path)
                if not pixmap.isNull():
                    feature_maps['neck'] = pixmap
                    self.log_message.emit(f"获取neck特征图成功")
                else:
                    self.log_message.emit(f"neck特征图加载失败")
            except Exception as e:
                self.log_message.emit(f"未找到neck特征图: {str(e)}")
                
            # 删除临时文件
            try:
                import os
                if os.path.exists(local_backbone_path):
                    os.remove(local_backbone_path)
                if os.path.exists(local_neck_path):
                    os.remove(local_neck_path)
            except Exception as e:
                self.log_message.emit(f"删除临时文件失败: {str(e)}")
                
        except Exception as e:
            self.log_message.emit(f"获取特征图发生异常: {str(e)}")
            
        return feature_maps
        
    def stop(self):
        """停止监控"""
        self.is_running = False
        if hasattr(self, 'ssh_client'):
            try:
                self.ssh_client.close()
            except:
                pass


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
        self.layer_names = ['backbone', 'neck']
        
        for i, layer_name in enumerate(self.layer_names):
            col = i
            
            # 创建组框
            group = QGroupBox(f"特征图: {layer_name}")
            group_layout = QVBoxLayout(group)
            
            # 创建图像标签
            label = QLabel("无数据")
            label.setMinimumSize(350, 350)  # 调整大小使特征图更明显
            label.setScaledContents(True)
            label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
            label.setAlignment(Qt.AlignCenter)
            
            group_layout.addWidget(label)
            layout.addWidget(group, 0, col)
            
            self.feature_labels[layer_name] = label
            
    def update_feature_maps(self, feature_maps):
        """更新特征图显示"""
        self.feature_maps = feature_maps
        
        for layer_name, pixmap in feature_maps.items():
            if layer_name in self.feature_labels:
                if pixmap and not pixmap.isNull():
                    # 缩放到标签大小
                    scaled_pixmap = pixmap.scaled(
                        self.feature_labels[layer_name].size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.feature_labels[layer_name].setPixmap(scaled_pixmap)
                    self.feature_labels[layer_name].setText("")  # 清除文本
                else:
                    self.feature_labels[layer_name].clear()  # 清除现有的pixmap
                    self.feature_labels[layer_name].setText("无数据")
                    self.feature_labels[layer_name].setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")


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
        
        # 初始加载模型列表
        self.refresh_model_list()
        
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
        
        layout.addWidget(running_group)
        
        # 可用模型
        available_group = QGroupBox("可用模型")
        available_layout = QVBoxLayout(available_group)
        
        self.available_models_list = QListWidget()
        self.available_models_list.itemClicked.connect(self.on_model_selected)
        available_layout.addWidget(self.available_models_list)
        
        layout.addWidget(available_group)
        
        # 控制按钮
        control_group = QGroupBox("控制")
        control_layout = QVBoxLayout(control_group)
        
        self.monitor_btn = QPushButton("开始监控")
        self.monitor_btn.setEnabled(False)
        self.monitor_btn.clicked.connect(self.toggle_model_monitoring)
        self.monitor_btn.setStyleSheet("""
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
        
        # 当前值显示
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
        
        # GPU选择下拉列表（单独放在GPU信息下方）
        gpu_select_layout = QHBoxLayout()
        gpu_select_layout.addStretch()
        gpu_select_layout.addWidget(QLabel("选择GPU:"))
        self.gpu_combo = QComboBox()
        self.gpu_combo.setEnabled(False)  # 初始禁用，等待GPU信息
        self.gpu_combo.currentIndexChanged.connect(self.on_gpu_selected)
        self.gpu_combo.setMaximumWidth(120)  # 限制下拉列表宽度
        gpu_select_layout.addWidget(self.gpu_combo)
        gpu_select_layout.addStretch()
        
        parent_layout.addLayout(gpu_select_layout)
        
    def on_gpu_selected(self, index):
        """处理GPU下拉列表选择事件"""
        if self.resource_monitor_thread:
            self.resource_monitor_thread.set_gpu_index(index)
        
    def update_gpu_count(self, count):
        """更新GPU数量"""
        # 更新下拉列表
        current_index = self.gpu_combo.currentIndex()
        self.gpu_combo.clear()
        
        if count > 0:
            for i in range(count):
                self.gpu_combo.addItem(f"GPU {i}", i)
            
            self.gpu_combo.setEnabled(True)
            
            # 保持原来选中的GPU，如果可能
            if current_index >= 0 and current_index < count:
                self.gpu_combo.setCurrentIndex(current_index)
            else:
                self.gpu_combo.setCurrentIndex(0)
        else:
            self.gpu_combo.setEnabled(False)
    
    def start_resource_monitoring(self):
        """启动资源监控"""
        self.resource_monitor_thread = ResourceMonitorThread()
        self.resource_monitor_thread.cpu_updated.connect(self.update_cpu)
        self.resource_monitor_thread.gpu_updated.connect(self.update_gpu)
        self.resource_monitor_thread.memory_updated.connect(self.update_memory)
        self.resource_monitor_thread.gpu_count_updated.connect(self.update_gpu_count)
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
        index, gpu_percent, gpu_memory_percent = values
        self.gpu_label.setText(f"GPU {index}: {gpu_percent:.1f}% / 显存: {gpu_memory_percent:.1f}%")
        
        # 更新历史数据
        if 'gpu_util' not in self.resource_history:
            self.resource_history['gpu_util'] = []
            self.resource_history['gpu_memory'] = []
            
        self.resource_history['gpu_util'].append(gpu_percent)
        self.resource_history['gpu_memory'].append(gpu_memory_percent)
        
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
            self.monitor_btn.setEnabled(False)  # 暂时禁用按钮，防止重复点击
            self.monitor_btn.setText("正在停止...")
            
            thread = self.model_monitor_threads[self.selected_model]
            thread.stop()  # 设置停止标志
            
            # 确保在主线程中清理UI
            self.current_model_label.setText("请选择要监控的模型")
            
            # 清除该模型的状态信息
            for row in range(self.status_table.rowCount()):
                if self.status_table.item(row, 0).text() == self.selected_model:
                    self.status_table.removeRow(row)
                    break
            
            # 清空特征图显示
            self.feature_map_widget.update_feature_maps({'backbone': None, 'neck': None})
            
            # 等待线程终止（最多等待3秒）
            if not thread.wait(3000):
                self.logger.warning(f"监控线程未能在预期时间内终止，正在强制终止")
                thread.terminate()  # 强制终止线程
                thread.wait()
                
            # 移除线程引用
            del self.model_monitor_threads[self.selected_model]
            
            self.monitor_btn.setText("开始监控")
            self.monitor_btn.setEnabled(True)
                    
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
        from app.services import model_registry
        
        # 清空现有列表
        self.running_models_list.clear()
        self.available_models_list.clear()
        
        try:
            models = model_registry.list_models()
            
            if not models:
                # 如果没有模型，添加提示项
                item = QListWidgetItem("未找到可用模型")
                item.setData(Qt.UserRole, None)
                item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
                self.available_models_list.addItem(item)
                self.logger.info("未找到可用模型")
                return
                
            for model in models:
                model_id = model.get('model_id', '')
                model_name = model.get('name', model_id)
                model_type = model.get('model_type', 'unknown')
                status = model.get('status', 'stopped')
                
                display_text = f"{model_name}\n[{model_type}]"
                
                item = QListWidgetItem()
                item.setData(Qt.UserRole, {
                    'name': model_name,
                    'model_id': model_id,
                    'status': status,
                    'type': model_type
                })
                
                # 根据状态分配到不同列表
                if status == 'running':
                    item.setText(f"🟢 {display_text}")
                    self.running_models_list.addItem(item)
                else:
                    item.setText(f"⚪ {display_text}")
                    self.available_models_list.addItem(item)
                    
            self.logger.info(f"已刷新模型列表，共 {len(models)} 个模型")
            
        except Exception as e:
            self.logger.error(f"刷新模型列表失败: {str(e)}")
            # 添加错误提示项
            error_item = QListWidgetItem(f"加载失败: {str(e)}")
            error_item.setData(Qt.UserRole, None)
            error_item.setFlags(error_item.flags() & ~Qt.ItemIsSelectable)
            self.available_models_list.addItem(error_item)
        
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
