#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能模型监控与优化系统
Main entry point for the application
"""

import sys
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        # logging.FileHandler('app.log', encoding='utf-8')  # 可选：保存到文件
    ]
)

def setup_application():
    """设置应用程序的全局配置"""
    # 启用高DPI支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # 创建应用程序实例
    app = QApplication(sys.argv)
    
    # 设置应用程序名称和组织
    app.setApplicationName("智能模型监控与优化系统")
    app.setOrganizationName("YourCompany")
    
    # 设置默认字体
    font = QFont("Microsoft YaHei", 9)  # Windows下使用微软雅黑
    app.setFont(font)
    
    # 设置应用程序样式
    app.setStyle('Fusion')  # 使用Fusion风格，跨平台一致性好
    
    return app

def main():
    """主函数"""
    try:
        print("正在启动应用程序...")
        # 设置应用程序
        app = setup_application()
        print("应用程序设置完成")
        
        # 导入并创建主窗口
        print("正在导入主窗口...")
        from app.views.main_window import create_main_window
        print("主窗口导入成功")
        
        main_window = create_main_window()
        print("主窗口创建成功")
        
        # 显示主窗口
        main_window.show()
        # 确保窗口最大化
        main_window.showMaximized()
        print("主窗口已显示")
        
        # 运行应用程序
        print("开始运行事件循环...")
        sys.exit(app.exec_())
        
    except Exception as e:
        logging.error(f"应用程序启动失败: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 