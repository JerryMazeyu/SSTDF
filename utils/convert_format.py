import logging
import json

logger = logging.getLogger(__name__)

def convert_format(input_file: str, confidence_threshold: float = 0.3) -> list:
    # 读取输入文件
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 获取labels、scores和bboxes
    labels = data['labels']
    scores = data['scores']
    bboxes = data['bboxes']
    
    # 转换格式
    converted_data = []
    for label, score, bbox in zip(labels, scores, bboxes):
        # 过滤低置信度的bbox
        if score < confidence_threshold:
            continue
            
        # 转换bbox格式：从x1,x2,y1,y2转换为x,y,w,h
        x1, y1, x2, y2 = bbox
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
        
        converted_item = {
            "class_name": label,
            "confidence": score,
            "bbox": [x, y, w, h]
        }
        converted_data.append(converted_item)
    return converted_data
