# -*- coding: utf-8 -*-
"""
模型管理对话框 - 展示可用模型并支持测试
"""

import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QPushButton, QLabel,
    QTextEdit, QGroupBox, QComboBox, QMessageBox,
    QProgressBar, QFrame, QWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QPixmap, QIcon
from app.services import model_registry


class ModelTestThread(QThread):
    """模型测试线程"""
    
    # 定义信号
    test_started = pyqtSignal(str, str)  # model_id, test_image
    test_finished = pyqtSignal(str, dict)  # model_id, result
    test_error = pyqtSignal(str, str)  # model_id, error_msg
    
    def __init__(self, model_id: str, test_image_path: str):
        super().__init__()
        self.model_id = model_id
        self.test_image_path = test_image_path
        
    def run(self):
        """执行模型测试"""
        try:
            self.test_started.emit(self.model_id, self.test_image_path)
            result = model_registry.test_model(self.model_id, self.test_image_path)
            self.test_finished.emit(self.model_id, result)
        except Exception as e:
            self.test_error.emit(self.model_id, str(e))


class ModelManagerDialog(QDialog):
    """模型管理对话框"""
    
    # 定义信号
    models_updated = pyqtSignal()  # 模型列表更新信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("模型管理")
        self.setModal(True)
        self.setMinimumSize(1000, 700)
        self.current_test_thread = None
        self.sample_images = []
        self.init_ui()
        self.load_sample_images()
        self.refresh_model_list()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("模型管理")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 主分割器
        main_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：模型列表
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # 右侧：模型详情和测试
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # 设置分割比例
        main_splitter.setSizes([400, 600])
        layout.addWidget(main_splitter)
        
        # 底部按钮
        button_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("刷新模型列表")
        self.refresh_btn.clicked.connect(self.refresh_model_list)
        button_layout.addWidget(self.refresh_btn)
        
        button_layout.addStretch()
        
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
    def create_left_panel(self):
        """创建左侧模型列表面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 模型列表
        models_group = QGroupBox("可用模型")
        models_layout = QVBoxLayout(models_group)
        
        self.model_list = QListWidget()
        self.model_list.itemClicked.connect(self.on_model_selected)
        models_layout.addWidget(self.model_list)
        
        layout.addWidget(models_group)
        
        return panel
        
    def create_right_panel(self):
        """创建右侧详情和测试面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 模型详情
        details_group = QGroupBox("模型详情")
        details_layout = QVBoxLayout(details_group)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(250)
        details_layout.addWidget(self.details_text)
        
        layout.addWidget(details_group)
        
        # 模型测试
        test_group = QGroupBox("模型测试")
        test_layout = QVBoxLayout(test_group)
        
        # 测试图像选择
        image_layout = QHBoxLayout()
        image_layout.addWidget(QLabel("测试图像:"))
        
        self.image_combo = QComboBox()
        self.image_combo.setMinimumWidth(200)
        image_layout.addWidget(self.image_combo)
        
        self.test_btn = QPushButton("开始测试")
        self.test_btn.setEnabled(False)
        self.test_btn.clicked.connect(self.start_model_test)
        self.test_btn.setStyleSheet("""
            QPushButton:enabled {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 2px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        image_layout.addWidget(self.test_btn)
        image_layout.addStretch()
        
        test_layout.addLayout(image_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        test_layout.addWidget(self.progress_bar)
        
        # 测试结果
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("请选择模型和测试图像，然后点击开始测试")
        test_layout.addWidget(self.result_text)
        
        layout.addWidget(test_group)
        
        return panel
        
    def load_sample_images(self):
        """加载样例图像"""
        resources_dir = "app/resources"
        if os.path.exists(resources_dir):
            for file in os.listdir(resources_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(resources_dir, file)
                    self.sample_images.append(image_path)
                    display_name = file.replace('sample_', '').replace('.jpg', '').replace('_', ' ').title()
                    self.image_combo.addItem(display_name, image_path)
        
        if not self.sample_images:
            self.image_combo.addItem("未找到样例图像", "")
            
    def refresh_model_list(self):
        """刷新模型列表"""
        self.model_list.clear()
        
        try:
            models = model_registry.list_models()
            
            if not models:
                item = QListWidgetItem("未找到可用模型")
                item.setData(Qt.UserRole, None)
                item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
                self.model_list.addItem(item)
                return
                
            for model in models:
                # 创建模型项
                model_name = model.get('name', model.get('model_id', 'Unknown'))
                model_type = model.get('model_type', 'unknown')
                framework = model.get('framework', 'unknown')
                
                display_text = f"{model_name}\n[{model_type}] - {framework}"
                
                item = QListWidgetItem(display_text)
                item.setData(Qt.UserRole, model)
                
                # 根据模型类型设置图标颜色
                if model.get('status') == 'running':
                    item.setText(f"🟢 {display_text}")
                else:
                    item.setText(f"⚪ {display_text}")
                    
                self.model_list.addItem(item)
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"刷新模型列表失败: {str(e)}")
            
    def on_model_selected(self, item):
        """处理模型选择事件"""
        model_data = item.data(Qt.UserRole)
        
        if model_data is None:
            self.details_text.clear()
            self.test_btn.setEnabled(False)
            return
            
        # 显示模型详情
        self.show_model_details(model_data)
        
        # 启用测试按钮
        has_images = len(self.sample_images) > 0
        self.test_btn.setEnabled(has_images)
        
        # 清除之前的测试结果
        self.result_text.clear()
        
    def show_model_details(self, model_data):
        """显示模型详情"""
        details = []
        
        details.append(f"模型名称: {model_data.get('name', 'Unknown')}")
        details.append(f"模型ID: {model_data.get('model_id', 'Unknown')}")
        details.append(f"描述: {model_data.get('description', '无描述')}")
        details.append(f"版本: {model_data.get('version', 'Unknown')}")
        details.append(f"类型: {model_data.get('model_type', 'unknown')}")
        details.append(f"框架: {model_data.get('framework', 'unknown')}")
        details.append(f"状态: {model_data.get('status', 'unknown')}")
        details.append(f"路径: {model_data.get('path', 'Unknown')}")
        
        # 添加性能信息（如果有）
        config = model_data.get('config', {})
        performance = config.get('performance', {})
        if performance:
            details.append("\n性能指标:")
            for key, value in performance.items():
                if isinstance(value, float):
                    details.append(f"  {key}: {value:.3f}")
                else:
                    details.append(f"  {key}: {value}")
                    
        # 添加标签信息（如果有）
        tags = config.get('tags', [])
        if tags:
            details.append(f"\n标签: {', '.join(tags)}")
            
        self.details_text.setPlainText("\n".join(details))
        
    def start_model_test(self):
        """开始模型测试"""
        # 获取选中的模型
        current_item = self.model_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "请先选择一个模型！")
            return
            
        model_data = current_item.data(Qt.UserRole)
        if not model_data:
            return
            
        # 获取选中的测试图像
        current_index = self.image_combo.currentIndex()
        if current_index < 0:
            QMessageBox.warning(self, "警告", "请选择测试图像！")
            return
            
        test_image_path = self.image_combo.itemData(current_index)
        if not test_image_path or not os.path.exists(test_image_path):
            QMessageBox.warning(self, "警告", "测试图像不存在！")
            return
            
        # 禁用测试按钮，显示进度条
        self.test_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.result_text.clear()
        self.result_text.append("🔄 正在测试模型...")
        
        # 启动测试线程
        model_id = model_data.get('model_id')
        self.current_test_thread = ModelTestThread(model_id, test_image_path)
        self.current_test_thread.test_started.connect(self.on_test_started)
        self.current_test_thread.test_finished.connect(self.on_test_finished)
        self.current_test_thread.test_error.connect(self.on_test_error)
        self.current_test_thread.start()
        
    @pyqtSlot(str, str)
    def on_test_started(self, model_id, test_image):
        """测试开始"""
        self.result_text.append(f"📋 模型ID: {model_id}")
        self.result_text.append(f"🖼️ 测试图像: {os.path.basename(test_image)}")
        self.result_text.append("")
        
    @pyqtSlot(str, dict)
    def on_test_finished(self, model_id, result):
        """测试完成"""
        self.progress_bar.setVisible(False)
        self.test_btn.setEnabled(True)
        
        self.result_text.append("✅ 测试完成!")
        self.result_text.append("=" * 50)
        
        if result.get('success', False):
            # 显示成功结果
            self.result_text.append("🎯 推理结果:")
            
            # 通用信息
            if 'inference_time_ms' in result:
                self.result_text.append(f"⏱️ 推理时间: {result['inference_time_ms']:.2f} ms")
                
            if 'image_size' in result:
                self.result_text.append(f"📐 图像尺寸: {result['image_size']}")
            
            self.result_text.append("")
            
            # 根据模型类型显示特定结果
            if 'detections' in result:
                # 目标检测结果
                num_detections = result.get('num_detections', 0)
                self.result_text.append(f"🔍 检测到 {num_detections} 个目标:")
                
                for i, detection in enumerate(result.get('detections', []), 1):
                    class_name = detection.get('class_name', 'Unknown')
                    confidence = detection.get('confidence', 0)
                    bbox = detection.get('bbox', [])
                    self.result_text.append(f"  {i}. {class_name} (置信度: {confidence:.3f})")
                    if bbox:
                        self.result_text.append(f"     位置: ({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})")
                        
            elif 'anomaly_score' in result:
                # 异常检测结果
                score = result.get('anomaly_score', 0)
                is_anomaly = result.get('is_anomaly', False)
                confidence = result.get('confidence', 0)
                severity = result.get('severity', 'unknown')
                
                self.result_text.append(f"🔬 异常检测结果:")
                self.result_text.append(f"  异常分数: {score:.4f}")
                self.result_text.append(f"  是否异常: {'是' if is_anomaly else '否'}")
                self.result_text.append(f"  置信度: {confidence:.3f}")
                self.result_text.append(f"  严重程度: {severity}")
                
                anomaly_regions = result.get('anomaly_regions', [])
                if anomaly_regions:
                    self.result_text.append(f"  异常区域数量: {len(anomaly_regions)}")
                    for i, region in enumerate(anomaly_regions, 1):
                        region_type = region.get('type', 'unknown')
                        region_score = region.get('score', 0)
                        self.result_text.append(f"    {i}. {region_type} (分数: {region_score:.3f})")
                        
        else:
            # 显示错误结果
            error_msg = result.get('error', '未知错误')
            self.result_text.append(f"❌ 测试失败: {error_msg}")
            
        self.result_text.append("")
        self.result_text.append(f"🕒 测试时间: {result.get('tested_at', 'Unknown')}")
        
    @pyqtSlot(str, str)
    def on_test_error(self, model_id, error_msg):
        """测试出错"""
        self.progress_bar.setVisible(False)
        self.test_btn.setEnabled(True)
        
        self.result_text.append(f"❌ 测试失败: {error_msg}")
        
    def closeEvent(self, event):
        """关闭事件"""
        # 停止正在运行的测试线程
        if self.current_test_thread and self.current_test_thread.isRunning():
            self.current_test_thread.terminate()
            self.current_test_thread.wait()
            
        self.models_updated.emit()
        event.accept() 