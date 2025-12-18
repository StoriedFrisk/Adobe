import json
import os
import shutil
import random
from tqdm import tqdm
import numpy as np

# ================= 配置区域 =================
# 1. 你的 JSON 根目录 (ISAT 生成的 json 文件夹)
json_root_dir = r'D:\Downloads\Compressed\bamboo_saw\labeled\Final' 

# 2. 那个存放所有原图的“混乱”文件夹
images_source_dir = r'D:\Downloads\Compressed\bamboo_saw\labeled\images'

# 3. 输出的 YOLO 数据集路径
output_dir = r'D:\Documaents\Adobe\data'

# 4. 类别名称映射
# ISAT json 中的 "category" -> YOLO ID
class_map = {
    '霉变': 0,
    '严重开裂': 1,  # 请确认 ISAT 里用的是不是这个词
    '虫眼': 2,
    '边壁缺失': 3
}

# 数据集划分比例
split_ratio = 0.8
# ===========================================

def convert_to_yolo_bbox(points, img_w, img_h):
    # 将 list 转为 numpy 方便计算
    points = np.array(points)
    
    # 提取矩形框 (xmin, ymin, xmax, ymax)
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    # 限制坐标在图片范围内 (防止画出界)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)

    # 转换为 YOLO 中心点格式 (x_center, y_center, w, h) 并归一化
    dw = 1. / img_w
    dh = 1. / img_h
    
    w = x_max - x_min
    h = y_max - y_min
    x = (x_min + x_max) / 2.0
    y = (y_min + y_max) / 2.0

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def make_dirs():
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

def main():
    make_dirs()
    
    # 收集所有 JSON 文件
    json_files = []
    for root, dirs, files in os.walk(json_root_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    random.shuffle(json_files)
    
    print(f"找到 {len(json_files)} 个 JSON 文件，开始转换 (ISAT 模式)...")
    
    processed_count = 0
    missing_img_count = 0

    for i, json_path in tqdm(enumerate(json_files), total=len(json_files)):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"无法读取 JSON: {json_path}, 错误: {e}")
            continue

        # ================= 修改点 1: 读取宽高信息 =================
        # ISAT 的宽高在 'info' 字典里
        info = data.get('info', {})
        img_w = info.get('width')
        img_h = info.get('height')
        img_name_in_json = info.get('name') # ISAT 通常会记录原始文件名

        if img_w is None or img_h is None:
            print(f"[跳过] JSON 缺少 info.width/height 信息: {json_path}")
            continue
        
        # ================= 修改点 2: 寻找图片文件 =================
        # 优先使用 JSON 里记录的文件名，其次尝试用 JSON 文件名推断
        image_found_path = None
        
        # 策略A: 尝试用 info['name'] 找 (例如 "20250917008027.jpg")
        if img_name_in_json:
            temp_path = os.path.join(images_source_dir, img_name_in_json)
            if os.path.exists(temp_path):
                image_found_path = temp_path

        # 策略B: 如果A找不到，尝试用 json 文件名匹配常见后缀
        if not image_found_path:
            file_base_name = os.path.splitext(os.path.basename(json_path))[0]
            valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
            for ext in valid_exts:
                temp_path = os.path.join(images_source_dir, file_base_name + ext)
                if os.path.exists(temp_path):
                    image_found_path = temp_path
                    break
        
        if not image_found_path:
            missing_img_count += 1
            if missing_img_count < 5:
                # 仅打印前5个错误，避免刷屏
                target_name = img_name_in_json if img_name_in_json else os.path.basename(json_path)
                print(f"[警告] 找不到对应的图片: {target_name}")
            continue

        # 划分训练/验证集
        split = 'train' if i < len(json_files) * split_ratio else 'val'

        # ================= 修改点 3: 解析 objects =================
        label_str = ""
        has_valid_object = False
        
        # ISAT 使用 'objects' 列表，而不是 'shapes'
        objects = data.get('objects', [])
        
        for obj in objects:
            # ISAT 使用 'category' 存放标签名
            label_name = obj.get('category')
            
            if label_name in class_map:
                class_id = class_map[label_name]
                
                # ISAT 使用 'segmentation' 存放点坐标 [[x1,y1], [x2,y2]...]
                points = obj.get('segmentation')
                
                # 如果没有 segmentation，尝试直接用 bbox (ISAT 也有 bbox 字段)
                # ISAT bbox 通常是 [xmin, ymin, xmax, ymax]
                if not points and 'bbox' in obj:
                    raw_bbox = obj['bbox']
                    # 构造一个伪 points 传给函数处理 (或者你可以单独写个 bbox 处理逻辑)
                    points = [
                        [raw_bbox[0], raw_bbox[1]], # 左上
                        [raw_bbox[2], raw_bbox[3]]  # 右下
                    ]

                if points:
                    bbox = convert_to_yolo_bbox(points, img_w, img_h)
                    label_str += f"{class_id} {' '.join(f'{x:.6f}' for x in bbox)}\n"
                    has_valid_object = True

        if has_valid_object:
            # 复制图片
            dst_img_path = os.path.join(output_dir, 'images', split, os.path.basename(image_found_path))
            shutil.copy2(image_found_path, dst_img_path)
            
            # 保存 TXT
            # 使用图片名作为 txt 文件名，防止 json 和 img 名字不一致的问题
            txt_base_name = os.path.splitext(os.path.basename(image_found_path))[0]
            dst_txt_path = os.path.join(output_dir, 'labels', split, txt_base_name + '.txt')
            
            with open(dst_txt_path, 'w', encoding='utf-8') as f:
                f.write(label_str)
            
            processed_count += 1

    print(f"\n转换完成！")
    print(f"成功转换: {processed_count} 张")
    print(f"丢失图片: {missing_img_count} 张")
    print(f"数据已保存在: {output_dir}")

if __name__ == '__main__':
    main()