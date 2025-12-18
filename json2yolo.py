import json
import os
import shutil
import random
from tqdm import tqdm
import numpy as np

# ================= 配置区域 =================
# 1. 你的 JSON 根目录 (里面包含 霉变/ 严重开裂/ 等子文件夹)
json_root_dir = r'D:\Downloads\\Compressed\\bamboo_saw\\labeled\\X-L' 

# 2. 那个存放所有原图的“混乱”文件夹
images_source_dir = r'D:\\Downloads\\Compressed\\bamboo_saw\\labeled\\images'

# 3. 输出的 YOLO 数据集路径
output_dir = r'D:\\Documaents\\Adobe\\data'

# 4. 类别名称 (必须和你 LabelMe 打标签时选的名字一模一样！)
# 注意：你的 JSON 里写的可能是 "严重开裂"，但你想让 YOLO 显示 "crack"
# 这里我们用一个字典来做映射： 'LabelMe里的名字': ID
class_map = {
    '霉变': 0,
    '严重开裂': 1,  # 如果你JSON里是'裂纹'，这里就改'裂纹'
    '虫眼': 2,      # 如果你JSON里是'虫洞'，这里就改'虫洞'
    '边壁缺失': 3
}

# 数据集划分比例
split_ratio = 0.8
# ===========================================

def convert_to_yolo_bbox(points, img_w, img_h):
    # LabelMe 的 points 可能是 [[x1,y1], [x2,y2]] (矩形) 或 [[x1,y1], [x2,y2], ...] (多边形)
    # 我们统一取所有点的最小/最大值来生成外接矩形
    points = np.array(points)
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    # 限制坐标在图片范围内
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)

    # 计算中心点和宽高
    dw = 1. / img_w
    dh = 1. / img_h
    w = x_max - x_min
    h = y_max - y_min
    x = (x_min + x_max) / 2.0
    y = (y_min + y_max) / 2.0

    # 归一化
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
    # 遍历 labeled 文件夹下的所有子文件夹
    for root, dirs, files in os.walk(json_root_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    # 打乱顺序
    random.shuffle(json_files)
    
    print(f"找到 {len(json_files)} 个 JSON 文件，开始转换...")
    
    processed_count = 0
    missing_img_count = 0

    for i, json_path in tqdm(enumerate(json_files), total=len(json_files)):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"无法读取 JSON: {json_path}, 错误: {e}")
            continue

        # 1. 获取图片尺寸
        img_w = data.get('imageWidth')
        img_h = data.get('imageHeight')

        # [新增] 安全检查：如果读不到宽高，直接跳过这张图
        if img_w is None or img_h is None:
            print(f"[跳过] JSON 缺少宽高信息: {json_path}")
            continue
        
        # 2. 确定图片文件名 (优先用 JSON 文件名，因为 imagePath 可能是绝对路径会出错)
        file_base_name = os.path.splitext(os.path.basename(json_path))[0]
        
        # 3. 在混乱文件夹里找对应的图片
        image_found_path = None
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        for ext in valid_exts:
            temp_path = os.path.join(images_source_dir, file_base_name + ext)
            if os.path.exists(temp_path):
                image_found_path = temp_path
                break
        
        if not image_found_path:
            # 尝试使用 json 里的 imagePath 字段
            json_img_path = data.get('imagePath')
            if json_img_path:
                 # 仅仅取文件名
                 temp_path = os.path.join(images_source_dir, os.path.basename(json_img_path))
                 if os.path.exists(temp_path):
                     image_found_path = temp_path

        if not image_found_path:
            missing_img_count += 1
            if missing_img_count < 5:
                print(f"[警告] 找不到对应的图片: {file_base_name}")
            continue

        # 4. 划分训练/验证集
        if i < len(json_files) * split_ratio:
            split = 'train'
        else:
            split = 'val'

        # 5. 转换标签
        label_str = ""
        has_valid_object = False
        
        # 修改：使用 .get('shapes', []) 
        # 意思是：尝试获取 shapes，如果没有，就当作是一个空列表 [] 处理，这样就不会报错了
        for shape in data.get('shapes', []):
            label_name = shape.get('label') # 为了保险，这里也可以加个 .get
            # 检查这个标签是否在我们的白名单里
            if label_name in class_map:
                class_id = class_map[label_name]
                points = shape['points']
                
                # 转换坐标
                bbox = convert_to_yolo_bbox(points, img_w, img_h)
                label_str += f"{class_id} {' '.join(f'{x:.6f}' for x in bbox)}\n"
                has_valid_object = True
            else:
                # 这是一个不在名单里的标签，比如 'end', 'outter' 等，跳过
                pass

        # 6. 如果这张图里有有效目标，才保存
        if has_valid_object:
            # 复制图片
            dst_img_path = os.path.join(output_dir, 'images', split, os.path.basename(image_found_path))
            shutil.copy2(image_found_path, dst_img_path)
            
            # 保存 TXT
            dst_txt_path = os.path.join(output_dir, 'labels', split, file_base_name + '.txt')
            with open(dst_txt_path, 'w', encoding='utf-8') as f:
                f.write(label_str)
            
            processed_count += 1

    print(f"\n转换完成！")
    print(f"成功转换: {processed_count} 张 (含标签)")
    print(f"找不到原图: {missing_img_count} 张")
    print(f"数据已保存在: {output_dir}")
    print("请记得更新 bamboo.yaml 中的 path 为上面的输出路径！")

if __name__ == '__main__':
    main()