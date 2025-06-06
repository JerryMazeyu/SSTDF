"""
运行所有服务模块测试示例的脚本
"""

import sys
import os
import subprocess
import time

def run_example(script_name, description):
    """运行单个示例脚本"""
    print(f"\n{'='*80}")
    print(f"开始运行: {description}")
    print(f"脚本: {script_name}")
    print(f"{'='*80}")
    
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 运行脚本
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5分钟超时
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        # 输出结果
        print(f"\n运行时间: {duration:.2f} 秒")
        
        if result.returncode == 0:
            print("✅ 运行成功!")
            if result.stdout.strip():
                print("\n--- 输出内容 ---")
                print(result.stdout)
        else:
            print("❌ 运行失败!")
            print(f"返回码: {result.returncode}")
            if result.stderr.strip():
                print("\n--- 错误信息 ---")
                print(result.stderr)
            if result.stdout.strip():
                print("\n--- 输出内容 ---")
                print(result.stdout)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ 运行超时!")
        return False
    except Exception as e:
        print(f"💥 运行异常: {str(e)}")
        return False


def main():
    """主函数"""
    print("服务模块测试示例运行器")
    print("="*80)
    
    # 检查当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"当前目录: {current_dir}")
    
    # 要运行的示例列表
    examples = [
        ("common_example.py", "Common服务模块功能测试"),
        ("monitor_example.py", "Monitor服务模块功能测试"),
    ]
    
    results = {}
    
    print(f"\n准备运行 {len(examples)} 个测试示例...")
    
    # 依次运行每个示例
    for script_name, description in examples:
        script_path = os.path.join(current_dir, script_name)
        
        if not os.path.exists(script_path):
            print(f"\n❌ 文件不存在: {script_path}")
            results[script_name] = False
            continue
        
        success = run_example(script_path, description)
        results[script_name] = success
    
    # 总结报告
    print(f"\n{'='*80}")
    print("测试总结报告")
    print(f"{'='*80}")
    
    total_tests = len(examples)
    success_count = sum(1 for success in results.values() if success)
    failure_count = total_tests - success_count
    
    print(f"总测试数: {total_tests}")
    print(f"成功: {success_count}")
    print(f"失败: {failure_count}")
    print(f"成功率: {success_count/total_tests*100:.1f}%")
    
    print("\n详细结果:")
    for script_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {script_name}: {status}")
    
    if failure_count == 0:
        print("\n🎉 所有测试都通过了!")
    else:
        print(f"\n⚠️  有 {failure_count} 个测试失败，请检查错误信息")
    
    return failure_count == 0


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 运行器出现异常: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 