import json

def convert_format(input_file:str)->list:
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
        converted_item = {
            "class_name": label,  # 如果找不到映射，使用默认格式
            "confidence": score,
            "bbox": bbox
        }
        converted_data.append(converted_item)
    return converted_data
