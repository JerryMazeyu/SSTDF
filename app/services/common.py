"""
通用服务模块 - 提供各个tab共用的后端接口
"""

import os
import sys
import platform
import psutil
import torch
import torch.nn as nn
import json
import time
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """模型信息数据类"""
    name: str
    path: str
    framework: str  # 'pytorch', 'tensorflow', 'onnx', etc.
    status: str  # 'running', 'stopped', 'error'
    loaded_time: Optional[datetime] = None
    device: Optional[str] = None  # 'cuda:0', 'cpu', etc.
    
    def to_dict(self):
        return {
            'name': self.name,
            'path': self.path,
            'framework': self.framework,
            'status': self.status,
            'loaded_time': self.loaded_time.isoformat() if self.loaded_time else None,
            'device': self.device
        }


class ModelRegistry:
    """模型注册管理器"""
    
    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._running_models: Dict[str, object] = {}  # 存储实际加载的模型对象
        
    def register_model(self, name: str, path: str, framework: str = 'pytorch') -> bool:
        """注册一个模型"""
        try:
            if name in self._models:
                logger.warning(f"模型 {name} 已存在，将覆盖原有注册")
                
            model_info = ModelInfo(
                name=name,
                path=path,
                framework=framework,
                status='stopped'
            )
            self._models[name] = model_info
            logger.info(f"成功注册模型: {name}")
            return True
            
        except Exception as e:
            logger.error(f"注册模型失败: {str(e)}")
            return False
            
    def list_models(self) -> List[Dict]:
        """列出所有已注册的模型"""
        return [model.to_dict() for model in self._models.values()]
        
    def get_model_info(self, name: str) -> Optional[Dict]:
        """获取指定模型的信息"""
        if name in self._models:
            return self._models[name].to_dict()
        return None
        
    def update_model_status(self, name: str, status: str, device: Optional[str] = None):
        """更新模型状态"""
        if name in self._models:
            self._models[name].status = status
            if status == 'running':
                self._models[name].loaded_time = datetime.now()
                if device:
                    self._models[name].device = device
            elif status == 'stopped':
                self._models[name].loaded_time = None
                self._models[name].device = None


# 全局模型注册器实例
model_registry = ModelRegistry()


def get_system_info() -> Dict:
    """获取系统信息"""
    return {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'python_version': sys.version,
        'torch_version': torch.__version__ if torch else 'Not installed',
        'cuda_available': torch.cuda.is_available() if torch else False,
        'cuda_version': torch.version.cuda if torch and torch.cuda.is_available() else None,
    }


def get_resource_usage() -> Dict:
    """获取当前系统资源使用情况"""
    # CPU信息
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    
    # 内存信息
    memory = psutil.virtual_memory()
    
    # 构建返回数据
    usage_data = {
        'timestamp': time.time(),
        'cpu': {
            'percent': cpu_percent,
            'count': cpu_count,
            'frequency': cpu_freq.current if cpu_freq else 0,
        },
        'memory': {
            'percent': memory.percent,
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
        }
    }
    
    # GPU信息（根据系统类型）
    gpu_info = get_gpu_info()
    if gpu_info:
        usage_data['gpu'] = gpu_info
        
    return usage_data


def get_gpu_info() -> Optional[Dict]:
    """获取GPU信息"""
    gpu_data = {}
    
    # 尝试使用torch获取CUDA信息
    if torch and torch.cuda.is_available():
        try:
            gpu_data['cuda_available'] = True
            gpu_data['device_count'] = torch.cuda.device_count()
            gpu_data['devices'] = []
            
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                gpu_data['devices'].append({
                    'index': i,
                    'name': device_props.name,
                    'total_memory': device_props.total_memory,
                    'major': device_props.major,
                    'minor': device_props.minor,
                })
                
        except Exception as e:
            logger.error(f"获取CUDA信息失败: {str(e)}")
    
    # 尝试使用nvidia-ml-py获取更详细的信息
    try:
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        if 'devices' not in gpu_data:
            gpu_data['devices'] = []
            
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # 获取GPU使用率
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # 获取内存信息
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # 获取温度
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = None
                
            # 更新或添加设备信息
            if i < len(gpu_data['devices']):
                gpu_data['devices'][i].update({
                    'utilization': util.gpu,
                    'memory_util': util.memory,
                    'memory_used': mem_info.used,
                    'memory_free': mem_info.free,
                    'temperature': temp,
                })
            else:
                gpu_data['devices'].append({
                    'index': i,
                    'utilization': util.gpu,
                    'memory_util': util.memory,
                    'memory_total': mem_info.total,
                    'memory_used': mem_info.used,
                    'memory_free': mem_info.free,
                    'temperature': temp,
                })
                
        pynvml.nvmlShutdown()
        
    except ImportError:
        logger.debug("pynvml未安装，无法获取详细GPU信息")
    except Exception as e:
        logger.error(f"获取GPU信息失败: {str(e)}")
        
    return gpu_data if gpu_data else None


def resource_monitor_stream(interval: float = 1.0) -> Generator[Dict, None, None]:
    """
    以流的形式返回系统资源占用情况
    
    Args:
        interval: 监控间隔（秒）
        
    Yields:
        资源使用情况字典
    """
    while True:
        try:
            yield get_resource_usage()
            time.sleep(interval)
        except GeneratorExit:
            break
        except Exception as e:
            logger.error(f"资源监控出错: {str(e)}")
            yield {'error': str(e)}
            time.sleep(interval)


def get_model_statistics(model: nn.Module) -> Dict:
    """
    获取PyTorch模型的统计信息
    
    Args:
        model: PyTorch模型
        
    Returns:
        包含模型统计信息的字典
    """
    def count_parameters(model):
        """计算模型参数量"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def get_model_size(model):
        """获取模型大小（字节）"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        return param_size + buffer_size
    
    try:
        total_params, trainable_params = count_parameters(model)
        model_size = get_model_size(model)
        
        # 获取模型结构信息
        layers_info = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 只统计叶子节点
                layers_info.append({
                    'name': name,
                    'type': module.__class__.__name__,
                    'params': sum(p.numel() for p in module.parameters())
                })
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_bytes': model_size,
            'model_size_mb': model_size / (1024 * 1024),
            'layers_count': len(layers_info),
            'layers_info': layers_info[:10],  # 只返回前10层信息
        }
        
    except Exception as e:
        logger.error(f"获取模型统计信息失败: {str(e)}")
        return {'error': str(e)}


def load_model_from_path(model_path: str, model_class: Optional[type] = None) -> Optional[nn.Module]:
    """
    从路径加载模型
    
    Args:
        model_path: 模型文件路径
        model_class: 模型类（如果需要先实例化）
        
    Returns:
        加载的模型对象
    """
    try:
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return None
            
        # 根据文件扩展名判断加载方式
        ext = os.path.splitext(model_path)[1].lower()
        
        if ext in ['.pth', '.pt']:
            # PyTorch模型
            if model_class:
                model = model_class()
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
            else:
                model = torch.load(model_path, map_location='cpu')
            return model
            
        elif ext == '.onnx':
            # ONNX模型（这里只返回路径，实际使用需要onnxruntime）
            logger.info(f"ONNX模型: {model_path}")
            return model_path
            
        else:
            logger.error(f"不支持的模型格式: {ext}")
            return None
            
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return None


# 初始化时注册一些示例模型（用于测试）
def init_sample_models():
    """初始化示例模型（仅用于测试）"""
    sample_models = [
        ("YOLO-v5", "/models/yolov5.pt", "pytorch"),
        ("ResNet-50", "/models/resnet50.pth", "pytorch"),
        ("EfficientNet-B0", "/models/efficientnet_b0.pth", "pytorch"),
        ("MobileNet-v2", "/models/mobilenet_v2.onnx", "onnx"),
    ]
    
    for name, path, framework in sample_models:
        model_registry.register_model(name, path, framework)
        # 模拟一些模型正在运行
        if name in ["YOLO-v5", "ResNet-50"]:
            model_registry.update_model_status(name, "running", "cuda:0")


# 初始化
init_sample_models()
