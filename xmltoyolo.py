import xml.etree.ElementTree as ET
import os
import shutil
import random
from tqdm import tqdm

classes = ['霉变', '严重开裂', '虫眼', '边壁缺失'] 

input_dir = 'D:\\Downloads\\Compressed\\bamboo_saw\\labeled\\annotations'  # XML文件夹
input_images_dir = 'D:\\Downloads\\Compressed\\bamboo_saw\\labeled\\images'  # 图片文件夹
output_dir = 'D:\\Documents\\Adobe\\data'  # 输出路径

split_ratios = 0.8

def xml2yolo(size, box):
    """ 将 VOC 坐标转换为 YOLO 坐标 """
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)

def conv_annotation(xml_file, output_txt_path):
    """ 读取 XML 并转换 """
    # encoding='utf-8'
    in_file = open(xml_file, encoding='utf-8')
    try:
        tree = ET.parse(in_file)
    except ET.ParseError:
        print(f"\n文件可能损坏..?: {xml_file}")
        in_file.close()
        return False
        
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    out_file = open(output_txt_path, 'w', encoding='utf-8')

    found_box = False
    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        difficult = int(difficult.text) if difficult is not None else 0
        
        cls = obj.find('name').text
        # 去除可能存在的首尾空格
        cls = cls.strip()
        
        if cls not in classes or difficult == 1:
            continue
            
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = xml2yolo((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        found_box = True
        
    in_file.close()
    out_file.close()
    return found_box

def make_dir():
    # 如果文件夹存在，先删除再创建，保持环境干净（可选，这里我保留了创建逻辑）
    for type_ in ['images', 'labels']:
        for split in ['train', 'val']:
            path = os.path.join(output_dir, type_, split)
            if not os.path.exists(path):
                os.makedirs(path)

def main():
    make_dir()
    
    # 过滤出图片文件
    image_files = [f for f in os.listdir(input_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    random.shuffle(image_files)

    train_count = 0
    val_count = 0
    missing_xml_count = 0

    print(f"找到 {len(image_files)} 张图片，开始处理...")

    for i, image_file in tqdm(enumerate(image_files), total=len(image_files)):
        # 处理类似 123.456.jpg
        file_name = os.path.splitext(image_file)[0]
        
        #xml.XML
        xml_file = os.path.join(input_dir, file_name + '.xml')
        if not os.path.exists(xml_file):
             xml_file = os.path.join(input_dir, file_name + '.XML')
        
        if not os.path.exists(xml_file):
            # 调试信息：只打印前 3 个找不到的，防止刷屏
            if missing_xml_count < 3:
                print(f"找不到对应的xml文档，跳过了: {file_name}.jpg...")
            missing_xml_count += 1
            continue

        # 3. 划分数据集
        if i < len(image_files) * split_ratios:
            split = 'train'
            train_count += 1
        else:
            split = 'val'
            val_count += 1

        # 4. 复制图片
        src_img = os.path.join(input_images_dir, image_file)
        dst_img = os.path.join(output_dir, 'images', split, image_file)
        shutil.copyfile(src_img, dst_img)

        # 5. 转换标签 (这是之前报错的地方，已经修复)
        dst_label = os.path.join(output_dir, 'labels', split, file_name + '.txt')
        conv_annotation(xml_file, dst_label)

    print(f"\n处理完毕！Summary:")
    print(f"  训练集: {train_count}")
    print(f"  验证集: {val_count}")
    print(f"  未找到XML跳过: {missing_xml_count}")
    print(f"数据已保存在: {output_dir}")

if __name__ == '__main__':
    main()