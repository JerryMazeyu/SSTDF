# -*- coding: utf-8 -*-
"""
检测服务 - 提供图像异常检测的后端逻辑
"""

import os
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import platform

from app.services import model_registry


class DetectionService:
    """检测服务类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def detect_single_image(self, image_path: str, model_ids: List[str]) -> Dict[str, Any]:
        """
        对单张图像进行检测
        
        Args:
            image_path: 图像路径
            model_ids: 要使用的模型ID列表
            
        Returns:
            检测结果字典
        """
        result = {
            'image_path': image_path,
            'image_name': os.path.basename(image_path),
            'detections': [],
            'success': True,
            'error': None
        }
        
        try:
            # 确保图像存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
            # 使用每个模型进行检测
            for model_id in model_ids:
                try:
                    # 获取模型信息
                    model_info = model_registry.get_model_info(model_id)
                    if not model_info:
                        self.logger.warning(f"模型 {model_id} 未找到")
                        continue
                    
                    # 执行模型推理
                    model_result = model_registry.test_model(model_id, image_path)
                    
                    # 整理检测结果
                    detection = {
                        'model_id': model_id,
                        'model_name': model_info.get('name', model_id),
                        'model_type': model_info.get('model_type', 'unknown'),
                        'success': model_result.get('success', False),
                        'error': model_result.get('error', None),
                        'anomalies': []
                    }
                    
                    if model_result.get('success', False):
                        # 根据不同的模型类型处理结果
                        if 'detections' in model_result:
                            # 目标检测类型
                            for det in model_result.get('detections', []):
                                anomaly = {
                                    'type': det.get('class_name', 'object'),
                                    'confidence': det.get('confidence', 0),
                                    'bbox': det.get('bbox', [0, 0, 100, 100]),
                                    'area': self._calculate_bbox_area(det.get('bbox', [0, 0, 100, 100]))
                                }
                                detection['anomalies'].append(anomaly)
                                
                        elif 'anomaly_regions' in model_result:
                            # 异常检测类型
                            for region in model_result.get('anomaly_regions', []):
                                anomaly = {
                                    'type': region.get('type', 'anomaly'),
                                    'confidence': region.get('score', 0),
                                    'bbox': region.get('bbox', [0, 0, 100, 100]),
                                    'area': self._calculate_bbox_area(region.get('bbox', [0, 0, 100, 100]))
                                }
                                detection['anomalies'].append(anomaly)
                                
                        elif model_result.get('is_anomaly', False):
                            # 整图异常判断
                            anomaly = {
                                'type': 'anomaly',
                                'confidence': model_result.get('anomaly_score', 0),
                                'bbox': [50, 50, 200, 200],  # 默认框
                                'area': 40000
                            }
                            detection['anomalies'].append(anomaly)
                    
                    result['detections'].append(detection)
                    
                except Exception as e:
                    self.logger.error(f"模型 {model_id} 检测失败: {str(e)}")
                    detection = {
                        'model_id': model_id,
                        'model_name': model_id,
                        'model_type': 'unknown',
                        'success': False,
                        'error': str(e),
                        'anomalies': []
                    }
                    result['detections'].append(detection)
                    
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            self.logger.error(f"检测图像 {image_path} 失败: {str(e)}")
            
        return result
    
    def detect_batch_images(self, image_paths: List[str], model_ids: List[str], 
                          progress_callback=None) -> List[Dict[str, Any]]:
        """
        批量检测图像
        
        Args:
            image_paths: 图像路径列表
            model_ids: 要使用的模型ID列表
            progress_callback: 进度回调函数，接收(current, total)参数
            
        Returns:
            检测结果列表
        """
        results = []
        total = len(image_paths)
        
        # 提交所有检测任务
        futures = {}
        for idx, image_path in enumerate(image_paths):
            future = self.executor.submit(self.detect_single_image, image_path, model_ids)
            futures[future] = idx
            
        # 收集结果
        for idx, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                
                # 调用进度回调
                if progress_callback:
                    progress_callback(idx + 1, total)
                    
            except Exception as e:
                self.logger.error(f"批量检测出错: {str(e)}")
                # 创建错误结果
                result = {
                    'image_path': image_paths[futures[future]],
                    'image_name': os.path.basename(image_paths[futures[future]]),
                    'detections': [],
                    'success': False,
                    'error': str(e)
                }
                results.append(result)
                
        return results
    
    def draw_detections_on_image(self, image_path: str, detections: List[Dict[str, Any]], 
                               output_path: Optional[str] = None,
                               line_thickness: int = 4,
                               font_size: int = 24) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image_path: 原始图像路径
            detections: 检测结果列表
            output_path: 输出图像路径（可选）
            line_thickness: 边界框线条粗细
            font_size: 标签字体大小
            
        Returns:
            绘制了检测结果的图像数组
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        # 转换为RGB格式（OpenCV默认是BGR）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建PIL图像用于绘制
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # 尝试加载中文字体，增加健壮性
        font_path = "simhei.ttf"
        if platform.system() == "Windows" and not os.path.exists(font_path):
            font_path = "C:/Windows/Fonts/simhei.ttf"
            
        try:
            font = ImageFont.truetype(font_path, font_size)
            # self.logger.info(f"成功加载字体 {font_path}，大小 {font_size}")
        except IOError:
            self.logger.warning(f"字体文件 '{font_path}' 未找到，回退到默认字体。字体大小设置将不生效。")
            font = ImageFont.load_default()
        
        # 为不同模型定义颜色
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
        
        # 绘制每个检测结果
        for model_idx, detection in enumerate(detections):
            if not detection.get('success', False):
                continue
                
            color = colors[model_idx % len(colors)]
            
            for anomaly in detection.get('anomalies', []):
                bbox = anomaly['bbox']
                confidence = anomaly['confidence']
                anomaly_type = anomaly['type']
                
                # 绘制边界框
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                
                # 绘制矩形框，使用传入的线宽
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_thickness)
                
                # 绘制标签背景
                label_text = f"{anomaly_type} ({confidence:.2f})"
                text_bbox = draw.textbbox((x1, y1), label_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # 绘制标签背景矩形
                draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], 
                             fill=color, outline=color)
                
                # 绘制文字
                draw.text((x1 + 2, y1 - text_height - 2), label_text, 
                         fill='white', font=font)
                
        # 转换回numpy数组
        result_image = np.array(pil_image)
        
        # 如果指定了输出路径，保存图像
        if output_path:
            # 转换回BGR格式保存
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)
            
        return result_image
    
    def _calculate_bbox_area(self, bbox: List[int]) -> int:
        """计算边界框面积"""
        if len(bbox) >= 4:
            return bbox[2] * bbox[3]
        return 0
    
    def shutdown(self):
        """关闭检测服务"""
        self.executor.shutdown(wait=True)


# 创建全局检测服务实例
detection_service = DetectionService()
