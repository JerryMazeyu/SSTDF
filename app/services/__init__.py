# -*- coding: utf-8 -*-
"""
服务模块 - 提供后端接口
"""

# 从common模块导出常用接口
from .common import (
    model_registry,
    get_system_info,
    get_resource_usage,
    resource_monitor_stream,
    get_model_statistics,
    load_model_from_path,
    ModelInfo
)

# 从monitor模块导出哨兵系统接口
from .monitor import (
    check_model_status,
    start_model_monitoring,
    stop_model_monitoring,
    get_model_feature_maps,
    get_model_inference_stats,
    get_all_monitoring_models,
    ModelMonitor,
    get_model_monitor
)

__all__ = [
    # Common
    'model_registry',
    'get_system_info',
    'get_resource_usage', 
    'resource_monitor_stream',
    'get_model_statistics',
    'load_model_from_path',
    'ModelInfo',
    
    # Monitor
    'check_model_status',
    'start_model_monitoring',
    'stop_model_monitoring',
    'get_model_feature_maps',
    'get_model_inference_stats',
    'get_all_monitoring_models',
    'ModelMonitor',
    'get_model_monitor'
] 