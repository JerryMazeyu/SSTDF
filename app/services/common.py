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
import importlib.util
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
    """基于文件系统的模型管理器"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = models_dir
        self._running_models: Dict[str, object] = {}  # 存储实际加载的模型对象
        self._model_status: Dict[str, str] = {}  # 存储模型状态
        self._scan_models()
        
    def _scan_models(self):
        """扫描models目录，发现可用的模型"""
        self._available_models = {}
        
        if not os.path.exists(self.models_dir):
            logger.warning(f"模型目录不存在: {self.models_dir}")
            return
            
        try:
            for model_id in os.listdir(self.models_dir):
                model_path = os.path.join(self.models_dir, model_id)
                if os.path.isdir(model_path):
                    conf_file = os.path.join(model_path, "conf.json")
                    model_file = os.path.join(model_path, "model.py")
                    
                    if os.path.exists(conf_file) and os.path.exists(model_file):
                        try:
                            with open(conf_file, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                            
                            model_info = ModelInfo(
                                name=config.get('name', model_id),
                                path=model_path,
                                framework=config.get('framework', 'unknown'),
                                status=self._model_status.get(model_id, 'stopped')
                            )
                            
                            # 添加额外信息
                            model_info.model_id = model_id
                            model_info.config = config
                            model_info.description = config.get('description', '')
                            model_info.version = config.get('version', '1.0.0')
                            model_info.model_type = config.get('model_type', 'unknown')
                            
                            self._available_models[model_id] = model_info
                            logger.debug(f"发现模型: {model_info.name} ({model_id})")
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"解析模型配置失败 {conf_file}: {e}")
                        except Exception as e:
                            logger.error(f"加载模型信息失败 {model_path}: {e}")
                            
        except Exception as e:
            logger.error(f"扫描模型目录失败: {e}")
            
    def list_models(self) -> List[Dict]:
        """列出所有可用的模型"""
        self._scan_models()  # 重新扫描以获取最新状态
        models = []
        for model_info in self._available_models.values():
            model_dict = model_info.to_dict()
            model_dict.update({
                'model_id': model_info.model_id,
                'description': model_info.description,
                'version': model_info.version,
                'model_type': model_info.model_type,
                'config': model_info.config
            })
            models.append(model_dict)
        return models
        
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """获取指定模型的信息"""
        self._scan_models()
        if model_id in self._available_models:
            model_info = self._available_models[model_id]
            model_dict = model_info.to_dict()
            model_dict.update({
                'model_id': model_info.model_id,
                'description': model_info.description,
                'version': model_info.version,
                'model_type': model_info.model_type,
                'config': model_info.config
            })
            return model_dict
        return None
        
    def update_model_status(self, model_id: str, status: str, device: Optional[str] = None):
        """更新模型状态"""
        self._model_status[model_id] = status
        if model_id in self._available_models:
            self._available_models[model_id].status = status
            if status == 'running':
                self._available_models[model_id].loaded_time = datetime.now()
                if device:
                    self._available_models[model_id].device = device
            elif status == 'stopped':
                self._available_models[model_id].loaded_time = None
                self._available_models[model_id].device = None
                
    def is_empty(self) -> bool:
        """检查是否没有可用的模型"""
        self._scan_models()
        return len(self._available_models) == 0
        
    def load_model_inference(self, model_id: str):
        """动态加载模型的推理函数"""
        if model_id not in self._available_models:
            raise ValueError(f"模型 {model_id} 不存在")
            
        model_path = self._available_models[model_id].path
        model_file = os.path.join(model_path, "model.py")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
            
        # 动态导入模型模块
        import importlib.util
        import sys
        
        spec = importlib.util.spec_from_file_location(f"model_{model_id}", model_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模型模块: {model_file}")
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"model_{model_id}"] = module
        spec.loader.exec_module(module)
        
        if not hasattr(module, 'inference'):
            raise AttributeError(f"模型 {model_id} 缺少 inference 函数")
            
        return module.inference
        
    def test_model(self, model_id: str, test_image_path: str) -> Dict:
        """测试模型推理功能"""
        try:
            inference_func = self.load_model_inference(model_id)
            result = inference_func(test_image_path)
            
            # 添加测试信息
            result['test_image'] = test_image_path
            result['model_id'] = model_id
            result['tested_at'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"模型测试失败 {model_id}: {e}")
            return {
                "success": False,
                "error": f"模型测试失败: {str(e)}",
                "model_id": model_id,
                "test_image": test_image_path,
                "tested_at": datetime.now().isoformat()
            }


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



