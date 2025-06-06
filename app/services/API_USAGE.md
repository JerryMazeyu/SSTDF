# 后端接口使用指南

## 接口分类

### 1. 共用接口 (common.py)

这些接口可以被多个tab使用：

#### 模型管理
```python
from app.services import model_registry

# 注册模型
model_registry.register_model("YOLOv5", "/path/to/yolov5.pt", "pytorch")

# 获取所有已注册模型
models = model_registry.list_models()

# 获取特定模型信息
model_info = model_registry.get_model_info("YOLOv5")

# 更新模型状态
model_registry.update_model_status("YOLOv5", "running", "cuda:0")
```

#### 系统资源监控
```python
from app.services import get_resource_usage, resource_monitor_stream

# 获取当前资源使用情况
usage = get_resource_usage()
print(f"CPU: {usage['cpu']['percent']}%")
print(f"Memory: {usage['memory']['percent']}%")
if 'gpu' in usage:
    print(f"GPU: {usage['gpu']['devices'][0]['utilization']}%")

# 流式监控资源（用于实时更新）
for usage_data in resource_monitor_stream(interval=1.0):
    # 处理实时数据
    update_ui(usage_data)
```

#### 模型统计信息
```python
from app.services import get_model_statistics, load_model_from_path

# 加载模型
model = load_model_from_path("/path/to/model.pth")

# 获取模型统计信息
stats = get_model_statistics(model)
print(f"总参数量: {stats['total_parameters']:,}")
print(f"模型大小: {stats['model_size_mb']:.2f} MB")
```

### 2. 哨兵系统专用接口 (monitor.py)

这些接口专门用于Tab2（哨兵系统）：

#### 模型监控
```python
from app.services import (
    start_model_monitoring,
    stop_model_monitoring,
    check_model_status
)

# 开始监控模型
result = start_model_monitoring("YOLOv5", model_path="/path/to/model.pth")
if result['success']:
    print(result['message'])

# 检查模型状态
status = check_model_status("YOLOv5")
print(f"模型状态: {status['status']}")
print(f"运行时长: {status['uptime']}")
print(f"当前FPS: {status['performance']['current']['fps']}")

# 停止监控
stop_model_monitoring("YOLOv5")
```

#### 特征图获取
```python
from app.services import get_model_feature_maps

# 获取模型特征图
feature_maps = get_model_feature_maps("YOLOv5")
for layer_name, feature_map in feature_maps.items():
    print(f"层 {layer_name}: {feature_map.shape}")
    # 在UI中显示特征图
    display_feature_map(layer_name, feature_map)
```

## 在前端UI中的使用示例

### Tab2 - 哨兵系统集成
```python
# 在 tab2.py 中

from app.services import (
    model_registry,
    resource_monitor_stream,
    start_model_monitoring,
    get_model_feature_maps
)

class Tab2Widget(QWidget):
    def __init__(self):
        super().__init__()
        # 初始化资源监控线程
        self.start_resource_monitoring()
        
    def start_resource_monitoring(self):
        """启动资源监控"""
        self.resource_thread = QThread()
        self.resource_worker = ResourceWorker()
        self.resource_worker.moveToThread(self.resource_thread)
        self.resource_worker.data_updated.connect(self.update_resource_charts)
        self.resource_thread.started.connect(self.resource_worker.run)
        self.resource_thread.start()
        
    def refresh_model_list(self):
        """刷新模型列表"""
        # 从后端获取模型列表
        models = model_registry.list_models()
        
        # 更新UI
        self.running_models_list.clear()
        self.available_models_list.clear()
        
        for model in models:
            if model['status'] == 'running':
                item = QListWidgetItem(f"🟢 {model['name']}")
                self.running_models_list.addItem(item)
            else:
                item = QListWidgetItem(f"⚪ {model['name']}")
                self.available_models_list.addItem(item)
                
    def on_start_monitoring(self):
        """开始监控选中的模型"""
        model_name = self.selected_model
        result = start_model_monitoring(model_name)
        
        if result['success']:
            # 开始获取特征图
            self.feature_map_timer = QTimer()
            self.feature_map_timer.timeout.connect(self.update_feature_maps)
            self.feature_map_timer.start(2000)  # 每2秒更新一次
            
    def update_feature_maps(self):
        """更新特征图显示"""
        feature_maps = get_model_feature_maps(self.selected_model)
        self.feature_map_widget.update_feature_maps(feature_maps)
```

## 注意事项

1. **错误处理**：所有接口都可能抛出异常，请在调用时进行适当的错误处理
2. **资源管理**：使用流式接口时要注意正确关闭和清理资源
3. **线程安全**：在UI线程中调用后端接口时，建议使用QThread避免阻塞
4. **GPU支持**：GPU相关功能需要安装pynvml库，否则只能获取基础信息 