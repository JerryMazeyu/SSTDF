"""
哨兵系统服务模块 - 提供模型运行状态监控相关的后端接口
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import random
import threading
from collections import deque
import logging
from datetime import datetime

from .common import model_registry, get_model_statistics, load_model_from_path

logger = logging.getLogger(__name__)


class ModelMonitor:
    """模型监控器"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_monitoring = False
        self.monitor_thread = None
        self.feature_maps = {}
        self.performance_history = deque(maxlen=100)  # 保存最近100条性能数据
        self.hooks = []
        
    def start_monitoring(self, model: Optional[nn.Module] = None):
        """开始监控模型"""
        if self.is_monitoring:
            logger.warning(f"模型 {self.model_name} 已在监控中")
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(model,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"开始监控模型: {self.model_name}")
        
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        self._remove_hooks()
        logger.info(f"停止监控模型: {self.model_name}")
        
    def _monitor_loop(self, model: Optional[nn.Module]):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 收集性能数据
                perf_data = self._collect_performance_data(model)
                self.performance_history.append(perf_data)
                
                # 更新模型状态
                model_registry.update_model_status(
                    self.model_name,
                    'running' if perf_data['fps'] > 0 else 'error'
                )
                
                time.sleep(1)  # 每秒更新一次
                
            except Exception as e:
                logger.error(f"监控模型 {self.model_name} 时出错: {str(e)}")
                time.sleep(1)
                
    def _collect_performance_data(self, model: Optional[nn.Module]) -> Dict:
        """收集模型性能数据（模拟）"""
        # 实际应用中，这里应该收集真实的性能数据
        return {
            'timestamp': time.time(),
            'fps': random.uniform(25, 35),
            'latency': random.uniform(10, 50),
            'accuracy': random.uniform(0.85, 0.99),
            'throughput': random.uniform(100, 200),
            'memory_usage': random.uniform(1000, 2000),  # MB
        }
        
    def register_feature_hooks(self, model: nn.Module, layer_names: List[str]):
        """注册特征图提取钩子"""
        self._remove_hooks()
        
        def get_activation(name):
            def hook(model, input, output):
                self.feature_maps[name] = output.detach()
            return hook
            
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)
                logger.debug(f"在层 {name} 注册了特征提取钩子")
                
    def _remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.feature_maps.clear()
        
    def get_feature_maps(self) -> Dict[str, np.ndarray]:
        """获取特征图"""
        # 如果有真实的特征图，返回处理后的数据
        if self.feature_maps:
            processed_maps = {}
            for name, tensor in self.feature_maps.items():
                # 转换为numpy数组，并选择合适的通道进行可视化
                if tensor.dim() == 4:  # batch x channels x height x width
                    # 选择第一个样本的第一个通道
                    feature_map = tensor[0, 0].cpu().numpy()
                elif tensor.dim() == 3:  # channels x height x width
                    feature_map = tensor[0].cpu().numpy()
                elif tensor.dim() == 2:  # height x width
                    feature_map = tensor.cpu().numpy()
                else:
                    continue
                    
                processed_maps[name] = feature_map
            return processed_maps
            
        # 模拟特征图数据
        return simulate_feature_maps()
        
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        if not self.performance_history:
            return {
                'current': {},
                'average': {},
                'history': []
            }
            
        # 当前性能
        current = self.performance_history[-1] if self.performance_history else {}
        
        # 计算平均值
        avg_fps = np.mean([p['fps'] for p in self.performance_history])
        avg_latency = np.mean([p['latency'] for p in self.performance_history])
        avg_accuracy = np.mean([p['accuracy'] for p in self.performance_history])
        
        return {
            'current': current,
            'average': {
                'fps': avg_fps,
                'latency': avg_latency,
                'accuracy': avg_accuracy,
            },
            'history': list(self.performance_history)[-20:]  # 返回最近20条记录
        }


# 全局监控器管理
_model_monitors: Dict[str, ModelMonitor] = {}


def get_model_monitor(model_name: str) -> ModelMonitor:
    """获取或创建模型监控器"""
    if model_name not in _model_monitors:
        _model_monitors[model_name] = ModelMonitor(model_name)
    return _model_monitors[model_name]


def check_model_status(model_name: str) -> Dict:
    """
    检查模型运行状态
    
    Returns:
        包含模型状态信息的字典
    """
    model_info = model_registry.get_model_info(model_name)
    if not model_info:
        return {'error': f'模型 {model_name} 未注册'}
        
    # 获取监控器
    monitor = get_model_monitor(model_name)
    
    # 获取性能统计
    perf_stats = monitor.get_performance_stats()
    
    # 构建状态信息
    status_info = {
        'model_name': model_name,
        'status': model_info['status'],
        'device': model_info.get('device', 'cpu'),
        'loaded_time': model_info.get('loaded_time'),
        'is_monitoring': monitor.is_monitoring,
        'performance': perf_stats,
    }
    
    # 如果模型正在运行，添加更多信息
    if model_info['status'] == 'running':
        status_info.update({
            'uptime': calculate_uptime(model_info.get('loaded_time')),
            'health': 'healthy' if perf_stats['current'].get('fps', 0) > 20 else 'degraded',
        })
        
    return status_info


def start_model_monitoring(model_name: str, model_path: Optional[str] = None) -> Dict:
    """
    开始监控指定模型
    
    Args:
        model_name: 模型名称
        model_path: 模型路径（如果需要加载）
        
    Returns:
        操作结果
    """
    try:
        # 获取模型信息
        model_info = model_registry.get_model_info(model_name)
        if not model_info:
            return {'success': False, 'error': f'模型 {model_name} 未注册'}
            
        # 获取监控器
        monitor = get_model_monitor(model_name)
        
        # 如果提供了模型路径，尝试加载模型
        model = None
        if model_path:
            model = load_model_from_path(model_path)
            
        # 开始监控
        monitor.start_monitoring(model)
        
        # 更新模型状态
        model_registry.update_model_status(model_name, 'running')
        
        return {
            'success': True,
            'message': f'开始监控模型 {model_name}',
            'monitor_id': id(monitor)
        }
        
    except Exception as e:
        logger.error(f"启动模型监控失败: {str(e)}")
        return {'success': False, 'error': str(e)}


def stop_model_monitoring(model_name: str) -> Dict:
    """停止监控指定模型"""
    try:
        monitor = get_model_monitor(model_name)
        monitor.stop_monitoring()
        
        # 更新模型状态
        model_registry.update_model_status(model_name, 'stopped')
        
        return {
            'success': True,
            'message': f'停止监控模型 {model_name}'
        }
        
    except Exception as e:
        logger.error(f"停止模型监控失败: {str(e)}")
        return {'success': False, 'error': str(e)}


def get_model_feature_maps(model_name: str) -> Dict[str, np.ndarray]:
    """
    获取模型的特征图
    
    Returns:
        特征图字典，键为层名称，值为numpy数组
    """
    monitor = get_model_monitor(model_name)
    return monitor.get_feature_maps()


def simulate_feature_maps() -> Dict[str, np.ndarray]:
    """模拟生成特征图（用于测试）"""
    feature_maps = {}
    
    # 模拟不同层的特征图
    layers = [
        ('conv1', (64, 64)),
        ('conv2', (32, 32)),
        ('conv3', (16, 16)),
        ('fc1', (8, 8)),
    ]
    
    for layer_name, size in layers:
        # 生成随机特征图
        if 'conv' in layer_name:
            # 卷积层：生成类似边缘检测的特征
            x = np.linspace(-3, 3, size[0])
            y = np.linspace(-3, 3, size[1])
            X, Y = np.meshgrid(x, y)
            Z = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-0.1 * (X**2 + Y**2))
            feature_map = (Z * 128 + 128).astype(np.uint8)
        else:
            # 全连接层：生成随机激活
            feature_map = np.random.randn(size[0], size[1]) * 50 + 128
            feature_map = np.clip(feature_map, 0, 255).astype(np.uint8)
            
        # 添加一些随机噪声
        noise = np.random.randn(*size) * 20
        feature_map = np.clip(feature_map + noise, 0, 255).astype(np.uint8)
        
        feature_maps[layer_name] = feature_map
        
    return feature_maps


def calculate_uptime(loaded_time: Optional[str]) -> str:
    """计算模型运行时间"""
    if not loaded_time:
        return "未知"
        
    try:
        if isinstance(loaded_time, str):
            loaded_dt = datetime.fromisoformat(loaded_time)
        else:
            loaded_dt = loaded_time
            
        uptime = datetime.now() - loaded_dt
        
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if days > 0:
            return f"{days}天 {hours}小时 {minutes}分钟"
        elif hours > 0:
            return f"{hours}小时 {minutes}分钟"
        else:
            return f"{minutes}分钟 {seconds}秒"
            
    except Exception as e:
        logger.error(f"计算运行时间失败: {str(e)}")
        return "未知"


def get_model_inference_stats(model_name: str, test_input: Optional[torch.Tensor] = None) -> Dict:
    """
    获取模型推理统计信息
    
    Args:
        model_name: 模型名称
        test_input: 测试输入（如果不提供则使用随机输入）
        
    Returns:
        推理统计信息
    """
    # 这里应该实现真实的推理测试
    # 现在返回模拟数据
    return {
        'model_name': model_name,
        'inference_time_ms': random.uniform(10, 50),
        'preprocessing_time_ms': random.uniform(1, 5),
        'postprocessing_time_ms': random.uniform(1, 5),
        'total_time_ms': random.uniform(15, 60),
        'throughput_fps': random.uniform(20, 60),
        'batch_size': 1,
        'input_shape': [1, 3, 224, 224],
        'output_shape': [1, 1000],
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    }


def get_all_monitoring_models() -> List[Dict]:
    """获取所有正在监控的模型列表"""
    monitoring_models = []
    
    for model_name, monitor in _model_monitors.items():
        if monitor.is_monitoring:
            model_info = model_registry.get_model_info(model_name)
            if model_info:
                monitoring_models.append({
                    'name': model_name,
                    'status': model_info['status'],
                    'device': model_info.get('device', 'cpu'),
                    'monitoring_time': calculate_uptime(model_info.get('loaded_time')),
                    'performance': monitor.get_performance_stats()['current']
                })
                
    return monitoring_models
