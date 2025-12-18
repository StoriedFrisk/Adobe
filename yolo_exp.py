import torch
import torch.nn as nn
# 1. 基础卷积模块 (Conv Module)
# YOLO用Conv+BatchNorm+SiLU的组合
def autopad(k, p=None):
    #计算padding，确保输出尺寸与输入相同
    if p is None:
        p=k//2 if isinstance(k,int) else [x//2 for x in k]
        return p
class Conv(nn.Module):
    #Conv:Conv2d+BatchNorm+SiLU
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        '''
        param c1: 输入通道数
        param c2: 输出通道数
        param k: 卷积核大小
        param s: 步长
        param p: padding，如果为 None 则自动计算
        param g: 分组卷积的组数
        param d: 膨胀率
        param act: 激活函数
        '''
        super().__init__()
        #卷积层
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        '''
        groups:分组卷积的组数
        dilation:膨胀率，控制卷积核元素之间的间距
        '''
        #批归一化层，二维卷积
        self.bn = nn.BatchNorm2d(c2)
        # Sigmoid Linear Unit，平滑的激活函数，相比ReLu更适合深层网络
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())#nn.Identity()表示不使用激活函数
        #前向传播，返回激活后的输出
        def forward(self, x):
            #bn = batch normalization
            return self.act(self.bn(self.conv(x)))
# 2. 核心特征提取模块 (C2f Module)
class Bottleneck(nn.Module):
    # 标准瓶颈层：ResNet 结构 (x + f(x))
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        '''
        shortcut: 是否使用残差连接
        g: 分组卷积的组数
        e: 中间层通道数的扩展比例
        '''
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1) # 输入分路
        self.cv2 = Conv((2 + n) * self.c, c2, 1) # 输出汇合
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
    def forward(self, x):
        # 这里的 split 和 extend 就是 C2f 丰富的梯度流来源
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
# 3. 空间金字塔池化 (SPPF Module)
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
# 4. 完整的网络定义 (YOLOv8 Class)
#是伪代码,主要由主干\颈部和检测组成。forward 函数定义了数据在网络中的前向传播路径。
class YOLOv8_Model(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        
        # 1. Backbone (主干网络): 负责提取特征
        self.backbone = nn.Sequential(
            Conv(3, 16, 3, 2),      # Layer 0
            Conv(16, 32, 3, 2),     # Layer 1
            C2f(32, 32, 1, True),   # Layer 2
            Conv(32, 64, 3, 2),     # Layer 3
            C2f(64, 64, 2, True),   # Layer 4
            Conv(64, 128, 3, 2),    # Layer 5
            C2f(128, 128, 2, True), # Layer 6
            Conv(128, 256, 3, 2),   # Layer 7
            C2f(256, 256, 1, True), # Layer 8
            SPPF(256, 256, 5)       # Layer 9
        )
        
        # 2. Head (检测头): 负责输出类别和坐标
        # YOLOv8 使用解耦头 (Decoupled Head)
        self.detect_head = Detect(nc=num_classes) 

    def forward(self, x):
        # 提取不同尺度的特征
        features = self.backbone(x)
        # 输入检测头得到结果
        return self.detect_head(features)
    
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')
# 这一行会把底层的 nn.Module 结构全部打印出来
print(model.model)