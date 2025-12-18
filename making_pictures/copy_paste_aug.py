import cv2
import numpy as np
import os
import random
from tqdm import tqdm

# ================= 配置区域 =================
# 建议路径结构：
# make_data/
#   ├── backgrounds/ (放正常的竹子图)
#   ├── patches/
#       ├── holes/ (放抠出来的虫眼png)
#       └── missing/ (放抠出来的边壁缺失png)
BG_DIR = r'make_data/backgrounds'       
PATCH_HOLES_DIR = r'make_data/patches/holes'     
PATCH_MISSING_DIR = r'make_data/patches/missing' 
OUTPUT_DIR = r'make_data/output'        

# 生成数量
NUM_TO_GENERATE = 500  # 建议生成多一点，因为它们现在形态各异

# 类别 ID (必须与您的 bamboo.yaml 一致)
CLS_ID_HOLE = 2     # 虫眼
CLS_ID_MISSING = 3  # 边壁缺失

# 虫眼生成的数量范围
HOLES_PER_IMAGE_MIN = 3
HOLES_PER_IMAGE_MAX = 8
# ===========================================

def load_images_from_folder(folder):
    images = []
    if not os.path.exists(folder):
        return []
    for filename in os.listdir(folder):
        # 读取图片，IMREAD_UNCHANGED 确保读取 Alpha 透明通道
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)
    return images

def rotate_image(image, angle):
    """
    旋转带透明通道的图片，并自动扩大画布以防被裁剪
    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # 1. 计算旋转矩阵
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 2. 计算旋转后的新画布宽高 (如果不算这个，旋转后图片会被切掉角)
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 3. 调整旋转矩阵的平移量，防止图像跑出画布
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # 4. 执行旋转 (注意 borderValue=(0,0,0,0) 填充透明色)
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

def overlay_image_alpha(img, img_overlay, x, y):
    """把带透明通道的 patch 贴到背景 img 上"""
    h, w = img.shape[:2]
    h_ov, w_ov = img_overlay.shape[:2]

    # 边界检查：如果贴纸完全跑出去了，直接返回
    if x >= w or y >= h or x + w_ov <= 0 or y + h_ov <= 0:
        return img

    # 裁剪贴纸和背景的重叠区域 (处理贴纸一部分在画布外的情况)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + w_ov), min(h, y + h_ov)
    
    # 对应的贴纸内部坐标
    ox1, oy1 = max(0, -x), max(0, -y)
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    # 提取重叠区域
    bg_crop = img[y1:y2, x1:x2]
    ov_crop = img_overlay[oy1:oy2, ox1:ox2]

    # 提取 Alpha 通道并归一化到 0-1
    if ov_crop.shape[2] == 4:
        alpha = ov_crop[:, :, 3] / 255.0
        ov_rgb = ov_crop[:, :, :3]
    else:
        alpha = np.ones((ov_crop.shape[0], ov_crop.shape[1]))
        ov_rgb = ov_crop

    # 扩展 alpha 维度以匹配 RGB (H, W, 1) -> (H, W, 3)
    alpha = alpha[:, :, np.newaxis]

    # 混合运算: Output = Alpha * Overlay + (1 - Alpha) * Background
    img[y1:y2, x1:x2, :3] = (alpha * ov_rgb + (1 - alpha) * bg_crop[:, :, :3]).astype(np.uint8)
    
    return img

def process_patch(patch):
    """对单个素材进行随机变换：翻转 -> 旋转 -> 缩放"""
    # 1. 随机翻转 (Flip)
    if random.random() > 0.5:
        patch = cv2.flip(patch, 1) # 水平翻转
    if random.random() > 0.5:
        patch = cv2.flip(patch, 0) # 垂直翻转

    # 2. 随机旋转 (Rotate)
    angle = random.uniform(0, 360)
    patch = rotate_image(patch, angle)

    # 3. 随机缩放 + 形变 (Scale & Aspect Ratio)
    # 让宽高缩放比例不同，圆形虫眼变椭圆，模拟不同视角
    scale_x = random.uniform(0.5, 1.2)
    scale_y = random.uniform(0.5, 1.2) 
    
    new_h = int(patch.shape[0] * scale_y)
    new_w = int(patch.shape[1] * scale_x)
    
    # 保护一下，防止缩得太小没了
    new_h = max(10, new_h)
    new_w = max(10, new_w)
    
    patch = cv2.resize(patch, (new_w, new_h))
    return patch

def main():
    # 1. 准备目录
    os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels'), exist_ok=True)

    # 2. 加载素材
    bg_files = [os.path.join(BG_DIR, f) for f in os.listdir(BG_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    hole_patches = load_images_from_folder(PATCH_HOLES_DIR)
    missing_patches = load_images_from_folder(PATCH_MISSING_DIR)

    if not bg_files:
        print("❌ 错误：背景文件夹是空的！请在 make_data/backgrounds 放入竹子原图。")
        return
    if not hole_patches and not missing_patches:
        print("❌ 错误：没有找到素材贴纸！请在 make_data/patches/holes 或 missing 下放入 png 文件。")
        return

    print(f"🚀 开始生成 {NUM_TO_GENERATE} 张“超级增强版”合成数据...")

    for i in tqdm(range(NUM_TO_GENERATE)):
        # A. 随机选一张背景
        bg_path = random.choice(bg_files)
        bg_img = cv2.imread(bg_path)
        if bg_img is None: continue
        
        h_bg, w_bg = bg_img.shape[:2]
        labels = [] 
        
        # B. 随机决定造什么缺陷
        mode = random.choice(['hole', 'missing'])
        if not hole_patches: mode = 'missing'
        if not missing_patches: mode = 'hole'

        if mode == 'hole':
            # --- 造虫眼 (随机生成多个) ---
            num_holes = random.randint(HOLES_PER_IMAGE_MIN, HOLES_PER_IMAGE_MAX)
            for _ in range(num_holes):
                # 随机选一个素材并进行变换
                patch = random.choice(hole_patches)
                patch_processed = process_patch(patch)
                
                ph, pw = patch_processed.shape[:2]
                
                # 随机位置 (避开最边缘，防止贴出去太多)
                if w_bg > pw and h_bg > ph:
                    x = random.randint(0, w_bg - pw)
                    y = random.randint(0, h_bg - ph)
                    
                    # 贴上去
                    bg_img = overlay_image_alpha(bg_img, patch_processed, x, y)
                    
                    # 计算 YOLO 标签 (归一化中心坐标)
                    xc = (x + pw / 2) / w_bg
                    yc = (y + ph / 2) / h_bg
                    nw = pw / w_bg
                    nh = ph / h_bg
                    labels.append(f"{CLS_ID_HOLE} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        elif mode == 'missing':
            # --- 造边壁缺失 (通常只造一个大的) ---
            patch = random.choice(missing_patches)
            patch_processed = process_patch(patch)
            
            # 对于边壁缺失，稍微放大一点点范围 (0.8 - 1.5)
            # process_patch 里默认是 0.5-1.2，这里如果您觉得不够大，可以再乘个系数，或者改上面的参数
            
            ph, pw = patch_processed.shape[:2]
            
            if w_bg > pw and h_bg > ph:
                x = random.randint(0, w_bg - pw)
                y = random.randint(0, h_bg - ph)
                
                bg_img = overlay_image_alpha(bg_img, patch_processed, x, y)
                
                xc = (x + pw / 2) / w_bg
                yc = (y + ph / 2) / h_bg
                nw = pw / w_bg
                nh = ph / h_bg
                labels.append(f"{CLS_ID_MISSING} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        # C. 保存结果
        out_name = f"aug_adv_{i:04d}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'images', f"{out_name}.jpg"), bg_img)
        # 只有当 labels 不为空时才保存 txt (防止生成空标签文件)
        if labels:
            with open(os.path.join(OUTPUT_DIR, 'labels', f"{out_name}.txt"), 'w') as f:
                f.write('\n'.join(labels))

    print(f"✅ 完成！生成数据已保存在: {OUTPUT_DIR}")
    print("💡 下一步：请将 output/images 和 output/labels 里的文件复制到您的训练集中。")

if __name__ == "__main__":
    main()