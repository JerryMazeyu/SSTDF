"""
Common服务模块测试示例
演示如何使用 app.services.common 中的各种接口
"""

import sys
import os
import time
import torch
import torch.nn as nn

# 添加项目根目录到path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from app.services.common import (
        model_registry,
        get_system_info,
        get_resource_usage,
        get_gpu_info,
        resource_monitor_stream,
        get_model_statistics,
        load_model_from_path,
        ModelInfo
    )
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"项目根目录: {project_root}")
    print(f"当前Python路径: {sys.path[:3]}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)


def test_model_registry():
    """测试基于文件系统的模型管理器功能"""
    print("=" * 60)
    print("测试基于文件系统的模型管理器功能")
    print("=" * 60)
    
    # 1. 扫描并列出所有可用模型
    print("\n1. 扫描models/目录中的可用模型:")
    models = model_registry.list_models()
    
    if not models:
        print("  未找到任何模型，请确保在models/目录下有正确格式的模型")
        return
        
    for model in models:
        print(f"  - {model['name']} ({model['framework']}) - 状态: {model['status']}")
        print(f"    模型ID: {model['model_id']}")
        print(f"    路径: {model['path']}")
        print(f"    类型: {model['model_type']}")
        print(f"    版本: {model['version']}")
        print(f"    描述: {model['description']}")
        if model.get('device'):
            print(f"    设备: {model['device']}")
        print()
    
    # 2. 获取特定模型信息
    print("\n2. 获取特定模型信息:")
    first_model_id = models[0]['model_id'] if models else None
    if first_model_id:
        model_info = model_registry.get_model_info(first_model_id)
        if model_info:
            print(f"  模型名称: {model_info['name']}")
            print(f"  模型ID: {model_info['model_id']}")
            print(f"  模型路径: {model_info['path']}")
            print(f"  框架: {model_info['framework']}")
            print(f"  状态: {model_info['status']}")
            print(f"  版本: {model_info['version']}")
            print(f"  性能指标: {model_info['config'].get('performance', {})}")
    else:
        print("  没有可用模型进行测试")
    
    # 3. 更新模型状态
    print("\n3. 更新模型状态:")
    if models:
        test_model_id = models[0]['model_id']
        model_registry.update_model_status(test_model_id, "running", "cuda:0")
        print(f"  已将模型 {test_model_id} 状态更新为 running")
        
        if len(models) > 1:
            test_model_id2 = models[1]['model_id']
            model_registry.update_model_status(test_model_id2, "error")
            print(f"  已将模型 {test_model_id2} 状态更新为 error")
    
    # 4. 测试模型推理功能
    print("\n4. 测试模型推理功能:")
    if models:
        test_model_id = models[0]['model_id']
        # 使用样例图像测试
        sample_image = "app/resources/sample_part1.jpg"
        if os.path.exists(sample_image):
            print(f"  使用测试图像: {sample_image}")
            result = model_registry.test_model(test_model_id, sample_image)
            
            if result.get('success', False):
                print(f"  ✅ 推理成功!")
                print(f"    模型: {result.get('model_name', 'Unknown')}")
                print(f"    推理时间: {result.get('inference_time_ms', 0):.2f} ms")
                
                # 根据不同类型显示结果
                if 'detections' in result:
                    print(f"    检测到 {result.get('num_detections', 0)} 个目标")
                elif 'anomaly_score' in result:
                    print(f"    异常分数: {result.get('anomaly_score', 0):.4f}")
                    print(f"    是否异常: {'是' if result.get('is_anomaly', False) else '否'}")
            else:
                print(f"  ❌ 推理失败: {result.get('error', '未知错误')}")
        else:
            print(f"  测试图像不存在: {sample_image}")
    else:
        print("  没有可用模型进行推理测试")


def test_system_info():
    """测试系统信息获取"""
    print("\n" + "=" * 60)
    print("测试系统信息获取")
    print("=" * 60)
    
    # 1. 获取系统基本信息
    print("\n1. 系统基本信息:")
    sys_info = get_system_info()
    print(f"  操作系统: {sys_info['platform']} {sys_info['platform_release']}")
    print(f"  架构: {sys_info['architecture']}")
    print(f"  处理器: {sys_info['processor']}")
    print(f"  Python版本: {sys_info['python_version'].split()[0]}")
    print(f"  PyTorch版本: {sys_info['torch_version']}")
    print(f"  CUDA可用: {sys_info['cuda_available']}")
    if sys_info['cuda_version']:
        print(f"  CUDA版本: {sys_info['cuda_version']}")


def test_resource_monitoring():
    """测试资源监控功能"""
    print("\n" + "=" * 60)
    print("测试资源监控功能")
    print("=" * 60)
    
    # 1. 获取当前资源使用情况
    print("\n1. 当前资源使用情况:")
    usage = get_resource_usage()
    
    print(f"  时间戳: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(usage['timestamp']))}")
    
    # CPU信息
    cpu = usage['cpu']
    print(f"  CPU使用率: {cpu['percent']:.1f}%")
    print(f"  CPU核心数: {cpu['count']}")
    print(f"  CPU频率: {cpu['frequency']:.0f} MHz")
    
    # 内存信息
    memory = usage['memory']
    print(f"  内存使用率: {memory['percent']:.1f}%")
    print(f"  总内存: {memory['total'] / (1024**3):.2f} GB")
    print(f"  已用内存: {memory['used'] / (1024**3):.2f} GB")
    print(f"  可用内存: {memory['available'] / (1024**3):.2f} GB")
    
    # GPU信息
    if 'gpu' in usage:
        gpu = usage['gpu']
        print(f"  GPU可用: {gpu.get('cuda_available', False)}")
        if 'devices' in gpu:
            for i, device in enumerate(gpu['devices']):
                print(f"  GPU {i}: {device.get('name', 'Unknown')}")
                if 'utilization' in device:
                    print(f"    使用率: {device['utilization']:.1f}%")
                    print(f"    内存使用率: {device['memory_util']:.1f}%")
                if 'temperature' in device and device['temperature']:
                    print(f"    温度: {device['temperature']}°C")
    else:
        print("  未检测到GPU或GPU信息获取失败")


def test_gpu_info():
    """测试GPU信息获取"""
    print("\n" + "=" * 60)
    print("测试GPU信息获取")
    print("=" * 60)
    
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"  CUDA可用: {gpu_info.get('cuda_available', False)}")
        print(f"  设备数量: {gpu_info.get('device_count', 0)}")
        
        if 'devices' in gpu_info:
            for device in gpu_info['devices']:
                print(f"\n  GPU {device['index']}:")
                print(f"    名称: {device.get('name', 'Unknown')}")
                if 'total_memory' in device:
                    print(f"    显存: {device['total_memory'] / (1024**3):.2f} GB")
                if 'utilization' in device:
                    print(f"    使用率: {device['utilization']}%")
                    print(f"    显存使用率: {device['memory_util']}%")
                if 'memory_used' in device:
                    print(f"    已用显存: {device['memory_used'] / (1024**3):.2f} GB")
                    print(f"    可用显存: {device['memory_free'] / (1024**3):.2f} GB")
                if 'temperature' in device and device['temperature']:
                    print(f"    温度: {device['temperature']}°C")
    else:
        print("  未检测到GPU信息")


def test_model_statistics():
    """测试模型统计信息获取"""
    print("\n" + "=" * 60)
    print("测试模型统计信息获取")
    print("=" * 60)
    
    # 创建一个简单的测试模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    
    print("\n1. 简单CNN模型统计:")
    stats = get_model_statistics(model)
    
    if 'error' in stats:
        print(f"  错误: {stats['error']}")
    else:
        print(f"  总参数量: {stats['total_parameters']:,}")
        print(f"  可训练参数: {stats['trainable_parameters']:,}")
        print(f"  不可训练参数: {stats['non_trainable_parameters']:,}")
        print(f"  模型大小: {stats['model_size_mb']:.2f} MB")
        print(f"  层数: {stats['layers_count']}")
        
        print("\n  前几层详细信息:")
        for layer in stats['layers_info'][:5]:
            print(f"    {layer['name']} ({layer['type']}): {layer['params']:,} 参数")


def test_stream_monitoring():
    """测试流式资源监控"""
    print("\n" + "=" * 60)
    print("测试流式资源监控 (运行5秒)")
    print("=" * 60)
    
    print("\n实时监控数据:")
    print("时间戳               CPU%   内存%   GPU%")
    print("-" * 50)
    
    count = 0
    for usage_data in resource_monitor_stream(interval=1.0):
        if 'error' in usage_data:
            print(f"错误: {usage_data['error']}")
            break
            
        timestamp = time.strftime('%H:%M:%S', time.localtime(usage_data['timestamp']))
        cpu_percent = usage_data['cpu']['percent']
        memory_percent = usage_data['memory']['percent']
        
        gpu_percent = "N/A"
        if 'gpu' in usage_data and usage_data['gpu']['devices']:
            gpu_percent = f"{usage_data['gpu']['devices'][0].get('utilization', 0)}"
        
        print(f"{timestamp}        {cpu_percent:5.1f}  {memory_percent:5.1f}  {gpu_percent:>4}")
        
        count += 1
        if count >= 5:  # 只运行5次
            break


def test_model_loading():
    """测试模型加载功能"""
    print("\n" + "=" * 60)
    print("测试模型加载功能")
    print("=" * 60)
    
    print("\n1. 测试不存在的模型文件:")
    model = load_model_from_path("/path/to/nonexistent/model.pth")
    print(f"  结果: {model}")
    
    print("\n2. 创建并保存一个简单模型:")
    # 创建一个简单模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
            
        def forward(self, x):
            return self.linear(x)
    
    test_model = TestModel()
    
    # 保存模型到临时位置
    temp_path = "temp_model.pth"
    torch.save(test_model.state_dict(), temp_path)
    print(f"  模型已保存到: {temp_path}")
    
    # 尝试加载模型
    print("\n3. 加载保存的模型:")
    try:
        loaded_model = TestModel()
        state_dict = torch.load(temp_path, map_location='cpu')
        loaded_model.load_state_dict(state_dict)
        print("  模型加载成功!")
        
        # 获取模型统计信息
        stats = get_model_statistics(loaded_model)
        print(f"  参数量: {stats['total_parameters']}")
        
        # 清理临时文件
        os.remove(temp_path)
        print("  临时文件已清理")
        
    except Exception as e:
        print(f"  模型加载失败: {str(e)}")


def main():
    """运行所有测试"""
    print("Common服务模块功能测试")
    print("=" * 80)
    
    try:
        test_model_registry()
        test_system_info()
        test_resource_monitoring()
        test_gpu_info()
        test_model_statistics()
        test_stream_monitoring()
        test_model_loading()
        
        print("\n" + "=" * 80)
        print("所有测试完成!")
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 