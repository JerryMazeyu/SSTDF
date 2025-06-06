"""
Monitor服务模块测试示例
演示如何使用 app.services.monitor 中的哨兵系统功能
"""

import sys
import os
import time
import threading
import torch
import torch.nn as nn
from typing import Dict, List

# 添加项目根目录到path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from app.services.monitor import (
        ModelMonitor,
        get_model_monitor,
        check_model_status,
        start_model_monitoring,
        stop_model_monitoring,
        get_model_feature_maps,
        simulate_feature_maps,
        calculate_uptime,
        get_model_inference_stats,
        get_all_monitoring_models
    )
    from app.services.common import model_registry
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"项目根目录: {project_root}")
    print(f"当前Python路径: {sys.path[:3]}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)


def test_model_monitor_basic():
    """测试ModelMonitor基本功能"""
    print("=" * 60)
    print("测试ModelMonitor基本功能")
    print("=" * 60)
    
    # 1. 创建模型监控器
    print("\n1. 创建模型监控器:")
    monitor = ModelMonitor("TestModel")
    print(f"  监控器已创建，模型名称: {monitor.model_name}")
    print(f"  监控状态: {monitor.is_monitoring}")
    
    # 2. 开始监控
    print("\n2. 开始监控:")
    monitor.start_monitoring()
    print(f"  监控已启动: {monitor.is_monitoring}")
    
    # 3. 等待一些性能数据收集
    print("\n3. 收集性能数据 (等待3秒):")
    time.sleep(3)
    
    # 4. 获取性能统计
    print("\n4. 性能统计信息:")
    perf_stats = monitor.get_performance_stats()
    
    if perf_stats['current']:
        current = perf_stats['current']
        print(f"  当前FPS: {current['fps']:.2f}")
        print(f"  当前延迟: {current['latency']:.2f} ms")
        print(f"  当前精度: {current['accuracy']:.3f}")
        print(f"  内存使用: {current['memory_usage']:.1f} MB")
    
    if perf_stats['average']:
        avg = perf_stats['average']
        print(f"  平均FPS: {avg['fps']:.2f}")
        print(f"  平均延迟: {avg['latency']:.2f} ms")
        print(f"  平均精度: {avg['accuracy']:.3f}")
    
    print(f"  历史记录数量: {len(perf_stats['history'])}")
    
    # 5. 停止监控
    print("\n5. 停止监控:")
    monitor.stop_monitoring()
    print(f"  监控已停止: {monitor.is_monitoring}")


def test_feature_maps():
    """测试特征图功能"""
    print("\n" + "=" * 60)
    print("测试特征图功能")
    print("=" * 60)
    
    # 1. 模拟特征图
    print("\n1. 生成模拟特征图:")
    feature_maps = simulate_feature_maps()
    
    for layer_name, feature_map in feature_maps.items():
        print(f"  {layer_name}: {feature_map.shape} - 类型: {feature_map.dtype}")
        print(f"    数值范围: [{feature_map.min():.1f}, {feature_map.max():.1f}]")
        print(f"    平均值: {feature_map.mean():.2f}")
    
    # 2. 使用真实模型测试特征图提取
    print("\n2. 真实模型特征图提取:")
    
    # 创建一个简单的CNN模型
    class TestCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
            self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 10)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = TestCNN()
    model.eval()
    
    # 创建监控器并注册钩子
    monitor = ModelMonitor("TestCNN")
    layer_names = ['conv1', 'conv2']
    monitor.register_feature_hooks(model, layer_names)
    
    # 运行一次前向传播
    test_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(test_input)
    
    # 获取特征图
    feature_maps = monitor.get_feature_maps()
    print(f"  提取到 {len(feature_maps)} 个特征图:")
    for layer_name, feature_map in feature_maps.items():
        print(f"    {layer_name}: {feature_map.shape}")
    
    # 清理
    monitor._remove_hooks()


def test_monitoring_workflow():
    """测试完整的监控工作流程"""
    print("\n" + "=" * 60)
    print("测试完整的监控工作流程")
    print("=" * 60)
    
    # 1. 注册测试模型
    print("\n1. 注册测试模型:")
    test_models = [
        ("TestModel1", "/models/test1.pth", "pytorch"),
        ("TestModel2", "/models/test2.pt", "pytorch"),
        ("TestModel3", "/models/test3.onnx", "onnx"),
    ]
    
    for name, path, framework in test_models:
        success = model_registry.register_model(name, path, framework)
        print(f"  注册 {name}: {'成功' if success else '失败'}")
    
    # 2. 开始监控
    print("\n2. 开始监控模型:")
    for model_name, _, _ in test_models[:2]:  # 只监控前两个
        result = start_model_monitoring(model_name)
        if result['success']:
            print(f"  ✓ {model_name} 监控已启动")
        else:
            print(f"  ✗ {model_name} 监控启动失败: {result['error']}")
    
    # 3. 等待一些数据收集
    print("\n3. 收集监控数据 (等待5秒)...")
    time.sleep(5)
    
    # 4. 检查模型状态
    print("\n4. 检查模型运行状态:")
    for model_name, _, _ in test_models:
        status = check_model_status(model_name)
        if 'error' in status:
            print(f"  {model_name}: 错误 - {status['error']}")
        else:
            print(f"  {model_name}:")
            print(f"    状态: {status['status']}")
            print(f"    设备: {status.get('device', 'Unknown')}")
            print(f"    监控中: {status['is_monitoring']}")
            if status.get('uptime'):
                print(f"    运行时长: {status['uptime']}")
            if status['performance']['current']:
                perf = status['performance']['current']
                print(f"    当前FPS: {perf['fps']:.2f}")
                print(f"    健康状态: {status.get('health', 'Unknown')}")
    
    # 5. 获取所有监控中的模型
    print("\n5. 所有监控中的模型:")
    monitoring_models = get_all_monitoring_models()
    if monitoring_models:
        for model in monitoring_models:
            print(f"  {model['name']} - {model['status']}")
            print(f"    设备: {model['device']}")
            print(f"    监控时长: {model['monitoring_time']}")
            if model['performance']:
                print(f"    当前FPS: {model['performance']['fps']:.2f}")
    else:
        print("  没有正在监控的模型")
    
    # 6. 获取特征图
    print("\n6. 获取模型特征图:")
    for model_name, _, _ in test_models[:1]:  # 只测试第一个
        feature_maps = get_model_feature_maps(model_name)
        print(f"  {model_name} 特征图:")
        for layer_name, feature_map in feature_maps.items():
            print(f"    {layer_name}: {feature_map.shape}")
    
    # 7. 停止监控
    print("\n7. 停止监控:")
    for model_name, _, _ in test_models:
        result = stop_model_monitoring(model_name)
        if result['success']:
            print(f"  ✓ {model_name} 监控已停止")
        else:
            print(f"  ✗ {model_name} 监控停止失败: {result['error']}")


def test_inference_stats():
    """测试推理统计功能"""
    print("\n" + "=" * 60)
    print("测试推理统计功能")
    print("=" * 60)
    
    # 1. 获取推理统计信息
    print("\n1. 获取模型推理统计:")
    test_models = ["YOLOv5", "ResNet-50", "EfficientNet-B0"]
    
    for model_name in test_models:
        stats = get_model_inference_stats(model_name)
        print(f"\n  {model_name}:")
        print(f"    推理时间: {stats['inference_time_ms']:.2f} ms")
        print(f"    预处理时间: {stats['preprocessing_time_ms']:.2f} ms")
        print(f"    后处理时间: {stats['postprocessing_time_ms']:.2f} ms")
        print(f"    总时间: {stats['total_time_ms']:.2f} ms")
        print(f"    吞吐量: {stats['throughput_fps']:.2f} FPS")
        print(f"    批大小: {stats['batch_size']}")
        print(f"    输入形状: {stats['input_shape']}")
        print(f"    输出形状: {stats['output_shape']}")
        print(f"    设备: {stats['device']}")


def test_uptime_calculation():
    """测试运行时间计算"""
    print("\n" + "=" * 60)
    print("测试运行时间计算")
    print("=" * 60)
    
    from datetime import datetime, timedelta
    
    # 1. 测试不同的时间差
    print("\n1. 测试运行时间计算:")
    
    test_cases = [
        ("1分钟前", datetime.now() - timedelta(minutes=1)),
        ("30分钟前", datetime.now() - timedelta(minutes=30)),
        ("2小时前", datetime.now() - timedelta(hours=2)),
        ("1天前", datetime.now() - timedelta(days=1)),
        ("3天前", datetime.now() - timedelta(days=3)),
        ("无效时间", "invalid_time"),
        ("空时间", None),
    ]
    
    for description, start_time in test_cases:
        if isinstance(start_time, datetime):
            time_str = start_time.isoformat()
        else:
            time_str = start_time
            
        uptime = calculate_uptime(time_str)
        print(f"  {description}: {uptime}")


def test_concurrent_monitoring():
    """测试并发监控"""
    print("\n" + "=" * 60)
    print("测试并发监控")
    print("=" * 60)
    
    # 1. 创建多个监控器
    print("\n1. 创建多个并发监控器:")
    monitors = []
    model_names = ["ConcurrentModel1", "ConcurrentModel2", "ConcurrentModel3"]
    
    for name in model_names:
        monitor = get_model_monitor(name)
        monitor.start_monitoring()
        monitors.append(monitor)
        print(f"  启动监控器: {name}")
    
    # 2. 让它们运行一段时间
    print("\n2. 并发运行监控 (8秒)...")
    time.sleep(8)
    
    # 3. 检查所有监控器的状态
    print("\n3. 检查并发监控状态:")
    for monitor in monitors:
        perf_stats = monitor.get_performance_stats()
        print(f"  {monitor.model_name}:")
        print(f"    监控状态: {monitor.is_monitoring}")
        print(f"    历史记录: {len(perf_stats['history'])} 条")
        if perf_stats['current']:
            print(f"    当前FPS: {perf_stats['current']['fps']:.2f}")
    
    # 4. 停止所有监控
    print("\n4. 停止所有并发监控:")
    for monitor in monitors:
        monitor.stop_monitoring()
        print(f"  停止监控器: {monitor.model_name}")


def test_error_handling():
    """测试错误处理"""
    print("\n" + "=" * 60)
    print("测试错误处理")
    print("=" * 60)
    
    # 1. 测试未注册模型的监控
    print("\n1. 尝试监控未注册的模型:")
    result = start_model_monitoring("NonexistentModel")
    print(f"  结果: {result}")
    
    # 2. 测试检查不存在模型的状态
    print("\n2. 检查不存在模型的状态:")
    status = check_model_status("NonexistentModel")
    print(f"  结果: {status}")
    
    # 3. 测试停止未运行的监控
    print("\n3. 停止未运行的监控:")
    result = stop_model_monitoring("NonexistentModel")
    print(f"  结果: {result}")
    
    # 4. 测试重复启动监控
    print("\n4. 测试重复启动监控:")
    model_registry.register_model("DuplicateTest", "/test/path.pt", "pytorch")
    
    result1 = start_model_monitoring("DuplicateTest")
    print(f"  第一次启动: {result1}")
    
    result2 = start_model_monitoring("DuplicateTest")
    print(f"  第二次启动: {result2}")
    
    # 清理
    stop_model_monitoring("DuplicateTest")


def performance_benchmark():
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("性能基准测试")
    print("=" * 60)
    
    # 1. 测试监控器创建速度
    print("\n1. 监控器创建性能测试:")
    start_time = time.time()
    monitors = []
    
    for i in range(100):
        monitor = ModelMonitor(f"BenchmarkModel{i}")
        monitors.append(monitor)
    
    creation_time = time.time() - start_time
    print(f"  创建100个监控器耗时: {creation_time:.3f} 秒")
    print(f"  平均每个: {creation_time/100*1000:.2f} ms")
    
    # 2. 测试特征图生成速度
    print("\n2. 特征图生成性能测试:")
    start_time = time.time()
    
    for _ in range(50):
        feature_maps = simulate_feature_maps()
    
    generation_time = time.time() - start_time
    print(f"  生成50次特征图耗时: {generation_time:.3f} 秒")
    print(f"  平均每次: {generation_time/50*1000:.2f} ms")
    
    # 3. 测试并发性能数据收集
    print("\n3. 并发性能数据收集测试:")
    monitor = ModelMonitor("PerformanceTest")
    monitor.start_monitoring()
    
    start_time = time.time()
    time.sleep(3)  # 让监控器运行3秒
    
    perf_stats = monitor.get_performance_stats()
    collection_time = time.time() - start_time
    
    print(f"  收集3秒性能数据，实际用时: {collection_time:.3f} 秒")
    print(f"  收集到数据点: {len(perf_stats['history'])}")
    
    monitor.stop_monitoring()


def main():
    """运行所有测试"""
    print("Monitor服务模块功能测试")
    print("=" * 80)
    
    try:
        test_model_monitor_basic()
        test_feature_maps()
        test_monitoring_workflow()
        test_inference_stats()
        test_uptime_calculation()
        test_concurrent_monitoring()
        test_error_handling()
        performance_benchmark()
        
        print("\n" + "=" * 80)
        print("所有测试完成!")
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理：停止所有可能还在运行的监控
        print("\n清理资源...")
        try:
            # 获取所有监控器并停止
            from app.services.monitor import _model_monitors
            for monitor in _model_monitors.values():
                if monitor.is_monitoring:
                    monitor.stop_monitoring()
            print("所有监控器已停止")
        except:
            pass


if __name__ == "__main__":
    main() 