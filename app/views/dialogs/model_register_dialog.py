# -*- coding: utf-8 -*-
"""
模型注册对话框
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QComboBox, QFileDialog,
    QDialogButtonBox, QMessageBox, QLabel
)
from PyQt5.QtCore import Qt, pyqtSignal
from app.services import model_registry


class ModelRegisterDialog(QDialog):
    """模型注册对话框"""
    
    # 定义信号
    model_registered = pyqtSignal(str, str, str)  # name, path, framework
    models_updated = pyqtSignal()  # 模型列表更新信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("注册新模型")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 表单布局
        form_layout = QFormLayout()
        
        # 模型名称输入
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("请输入模型名称")
        form_layout.addRow("模型名称:", self.name_input)
        
        # 模型路径选择
        path_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("请选择模型文件路径")
        self.path_input.setReadOnly(True)
        
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_model_file)
        
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_btn)
        form_layout.addRow("模型路径:", path_layout)
        
        # 模型框架选择
        self.framework_combo = QComboBox()
        self.framework_combo.addItems([
            "PyTorch",
            "TensorFlow",
            "ONNX",
            "TensorRT",
            "OpenVINO",
            "其他"
        ])
        form_layout.addRow("模型框架:", self.framework_combo)
        
        # 模型描述（可选）
        self.description_input = QLineEdit()
        self.description_input.setPlaceholderText("可选：输入模型描述")
        form_layout.addRow("模型描述:", self.description_input)
        
        layout.addLayout(form_layout)
        
        # 提示信息
        info_label = QLabel("注意：请确保模型文件路径正确，且模型格式与所选框架匹配。")
        info_label.setStyleSheet("color: #666; font-size: 11px; padding: 10px 0;")
        layout.addWidget(info_label)
        
        # 按钮
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal
        )
        button_box.accepted.connect(self.register_model)
        button_box.rejected.connect(self.reject)
        
        # 自定义按钮文本
        button_box.button(QDialogButtonBox.Ok).setText("注册")
        button_box.button(QDialogButtonBox.Cancel).setText("取消")
        
        layout.addWidget(button_box)
        
    def browse_model_file(self):
        """浏览选择模型文件"""
        file_filter = "Model Files (*.pth *.pt *.onnx *.pb *.h5 *.tflite *.xml);;All Files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            "",
            file_filter
        )
        
        if file_path:
            self.path_input.setText(file_path)
            
            # 自动填充模型名称（如果为空）
            if not self.name_input.text():
                import os
                model_name = os.path.splitext(os.path.basename(file_path))[0]
                self.name_input.setText(model_name)
                
            # 尝试自动检测框架
            self.auto_detect_framework(file_path)
            
    def auto_detect_framework(self, file_path):
        """根据文件扩展名自动检测框架"""
        import os
        ext = os.path.splitext(file_path)[1].lower()
        
        framework_map = {
            '.pth': 'PyTorch',
            '.pt': 'PyTorch',
            '.onnx': 'ONNX',
            '.pb': 'TensorFlow',
            '.h5': 'TensorFlow',
            '.tflite': 'TensorFlow',
            '.xml': 'OpenVINO',
            '.bin': 'OpenVINO',
        }
        
        if ext in framework_map:
            framework = framework_map[ext]
            index = self.framework_combo.findText(framework)
            if index >= 0:
                self.framework_combo.setCurrentIndex(index)
                
    def register_model(self):
        """注册模型"""
        # 验证输入
        name = self.name_input.text().strip()
        path = self.path_input.text().strip()
        framework = self.framework_combo.currentText()
        
        if not name:
            QMessageBox.warning(self, "警告", "请输入模型名称！")
            self.name_input.setFocus()
            return
            
        if not path:
            QMessageBox.warning(self, "警告", "请选择模型文件！")
            self.browse_btn.setFocus()
            return
            
        # 检查模型是否已存在
        existing_models = model_registry.list_models()
        if any(model['name'] == name for model in existing_models):
            reply = QMessageBox.question(
                self, 
                "确认",
                f"模型 '{name}' 已存在，是否覆盖？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
                
        # 注册模型
        success = model_registry.register_model(
            name=name,
            path=path,
            framework=framework.lower()
        )
        
        if success:
            # 发射信号
            self.model_registered.emit(name, path, framework.lower())
            self.models_updated.emit()  # 发射模型列表更新信号
            
            QMessageBox.information(
                self,
                "成功",
                f"模型 '{name}' 注册成功！"
            )
            self.accept()
        else:
            QMessageBox.critical(
                self,
                "错误",
                f"模型注册失败，请检查日志获取详细信息。"
            )
            
    def get_model_info(self):
        """获取模型信息"""
        return {
            'name': self.name_input.text().strip(),
            'path': self.path_input.text().strip(),
            'framework': self.framework_combo.currentText().lower(),
            'description': self.description_input.text().strip()
        } 