"""
配置管理器 - 管理应用程序的配置和缓存
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器类"""
    
    # 默认配置
    DEFAULT_CONFIG = {
        'general': {
            'language': 'zh_CN',
            'theme': 'light',
            'auto_save': True,
            'auto_save_interval': 300,  # 秒
        },
        'detection': {
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'max_detections': 100,
            'default_model': None,
            'batch_size': 1,
        },
        'monitor': {
            'update_interval': 1.0,  # 秒
            'max_history_points': 100,
            'auto_start_monitoring': False,
            'show_gpu_metrics': True,
        },
        'prediction': {
            'history_window': 100,
            'prediction_horizon': 20,
            'update_interval': 5.0,
            'confidence_interval': 0.95,
        },
        'paths': {
            'models_dir': './models',
            'cache_dir': './cache',
            'logs_dir': './logs',
            'results_dir': './results',
        },
        'ui': {
            'window_state': 'normal',
            'window_geometry': None,
            'splitter_sizes': {},
            'recent_files': [],
            'max_recent_files': 10,
        }
    }
    
    def __init__(self, config_file: str = 'config.json'):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件名
        """
        self.config_dir = Path.home() / '.sstdf'
        self.config_file = self.config_dir / config_file
        self.config = self.DEFAULT_CONFIG.copy()
        
        # 确保配置目录存在
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.load_config()
        
        # 确保必要的目录存在
        self._ensure_directories()
        
    def _ensure_directories(self):
        """确保必要的目录存在"""
        for key, path in self.config['paths'].items():
            Path(path).mkdir(parents=True, exist_ok=True)
            
    def load_config(self) -> bool:
        """
        从文件加载配置
        
        Returns:
            是否成功加载
        """
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    
                # 深度合并配置（保留默认值中新增的项）
                self.config = self._deep_merge(self.DEFAULT_CONFIG, loaded_config)
                logger.info(f"成功加载配置文件: {self.config_file}")
                return True
            else:
                logger.info("配置文件不存在，使用默认配置")
                self.save_config()  # 保存默认配置
                return False
                
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            return False
            
    def save_config(self) -> bool:
        """
        保存配置到文件
        
        Returns:
            是否成功保存
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            logger.info(f"成功保存配置文件: {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {str(e)}")
            return False
            
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key_path: 配置键路径，如 'general.language' 或 'detection.confidence_threshold'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key_path: str, value: Any) -> bool:
        """
        设置配置值
        
        Args:
            key_path: 配置键路径
            value: 配置值
            
        Returns:
            是否成功设置
        """
        keys = key_path.split('.')
        config = self.config
        
        try:
            # 导航到父级
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
                
            # 设置值
            config[keys[-1]] = value
            
            # 自动保存
            if self.get('general.auto_save', True):
                self.save_config()
                
            return True
            
        except Exception as e:
            logger.error(f"设置配置失败: {str(e)}")
            return False
            
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        深度合并两个字典
        
        Args:
            base: 基础字典
            update: 更新字典
            
        Returns:
            合并后的字典
        """
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def reset_to_default(self, section: Optional[str] = None) -> bool:
        """
        重置配置到默认值
        
        Args:
            section: 要重置的部分，如果为None则重置全部
            
        Returns:
            是否成功重置
        """
        try:
            if section:
                if section in self.DEFAULT_CONFIG:
                    self.config[section] = self.DEFAULT_CONFIG[section].copy()
                    logger.info(f"重置配置部分: {section}")
                else:
                    logger.warning(f"未知的配置部分: {section}")
                    return False
            else:
                self.config = self.DEFAULT_CONFIG.copy()
                logger.info("重置所有配置到默认值")
                
            self.save_config()
            return True
            
        except Exception as e:
            logger.error(f"重置配置失败: {str(e)}")
            return False
            
    def add_recent_file(self, file_path: str):
        """添加最近使用的文件"""
        recent_files = self.get('ui.recent_files', [])
        
        # 如果文件已存在，先移除
        if file_path in recent_files:
            recent_files.remove(file_path)
            
        # 添加到开头
        recent_files.insert(0, file_path)
        
        # 限制数量
        max_files = self.get('ui.max_recent_files', 10)
        recent_files = recent_files[:max_files]
        
        self.set('ui.recent_files', recent_files)
        
    def get_recent_files(self) -> list:
        """获取最近使用的文件列表"""
        return self.get('ui.recent_files', [])
        
    def clear_cache(self) -> bool:
        """
        清除缓存
        
        Returns:
            是否成功清除
        """
        try:
            cache_dir = Path(self.get('paths.cache_dir', './cache'))
            
            if cache_dir.exists():
                # 备份缓存目录路径
                backup_dir = cache_dir.parent / f"cache_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # 移动到备份目录（以防需要恢复）
                shutil.move(str(cache_dir), str(backup_dir))
                
                # 重新创建空的缓存目录
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                # 删除备份（可选）
                shutil.rmtree(backup_dir)
                
                logger.info("成功清除缓存")
                return True
            else:
                logger.info("缓存目录不存在")
                return True
                
        except Exception as e:
            logger.error(f"清除缓存失败: {str(e)}")
            return False
            
    def get_cache_size(self) -> int:
        """
        获取缓存大小（字节）
        
        Returns:
            缓存大小
        """
        cache_dir = Path(self.get('paths.cache_dir', './cache'))
        
        if not cache_dir.exists():
            return 0
            
        total_size = 0
        for item in cache_dir.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
                
        return total_size
        
    def export_config(self, export_path: str) -> bool:
        """
        导出配置到指定文件
        
        Args:
            export_path: 导出路径
            
        Returns:
            是否成功导出
        """
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            logger.info(f"成功导出配置到: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出配置失败: {str(e)}")
            return False
            
    def import_config(self, import_path: str) -> bool:
        """
        从指定文件导入配置
        
        Args:
            import_path: 导入路径
            
        Returns:
            是否成功导入
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
                
            # 验证配置格式
            if not isinstance(imported_config, dict):
                logger.error("导入的配置格式无效")
                return False
                
            # 合并配置
            self.config = self._deep_merge(self.DEFAULT_CONFIG, imported_config)
            self.save_config()
            
            logger.info(f"成功导入配置从: {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"导入配置失败: {str(e)}")
            return False


# 创建全局配置管理器实例
config_manager = ConfigManager()
