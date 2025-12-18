# Copilot 使用说明 — Bamboo/YOLO 项目

简短目标：帮助 AI 代码代理快速上手此仓库，理解核心架构、常用工作流、项目约定与易错点。内容基于仓库可发现的脚本与配置（`bamboo.yaml`, `YoloTest1.py`, `xmltoyolo.py`, `val.py`, `data/`, `runs/` 等）。

- **代码类型**: 主要是基于 Ultralytics YOLO 的目标检测训练/验证脚本，以及一个用于 VOC XML→YOLO 标签转换的实用脚本。另有单独的 UNet 训练脚本（`__pycache__/Train.py`）。

- **核心文件与职责**:
  - `bamboo.yaml`: 数据集配置（路径、类别数量 `nc`、名称 `names`）。训练脚本与 `val.py` 使用它作为数据描述。
  - `YoloTest1.py`: 训练入口示例，使用 `from ultralytics import YOLO` 和 `model.train(...)`。这是常用的启动训练脚本（短、可直接运行）。
  - `val.py`: 使用 `YOLO(...).val(...)` 对训练产出（如 `runs/detect/bamboo_exp/weights/best.pt`）做验证并打印 mAP 指标。
  - `xmltoyolo.py`: 将 VOC 格式 XML 转为 YOLO 文本标签并把图像/标签按 `train`/`val` 划分到 `data/images/*`、`data/labels/*`。包含 `classes` 列表与 `input_dir`/`output_dir` 等路径配置（注意：脚本中使用 Windows 绝对路径）。
  - `data/`: 实际数据目录，期望结构为 `images/train`, `images/val`, `labels/train`, `labels/val`（当前仓库中已包含大量 `.txt` 标签文件）。
  - `runs/`: Ultralytics 训练/检测输出目录，训练时会在 `runs/detect/<name>/` 下创建 `weights/best.pt` 等。

- **重要项目约定 / 可被发现的模式**:
  - 采用 Ultralytics API（`YOLO` 类）进行训练、验证：`YOLO(model_path).train(...)` / `YOLO(path).val(...)`。
  - `bamboo.yaml` 中 `path` 通常为仓库内 `data` 的绝对路径（样例使用 Windows 路径 `D:\\Documaents\\Adobe\\data`）。训练脚本使用相对子路径 `images/train` 与 `images/val`。
  - 标签为常见 YOLO文本格式（每行 `class x_center y_center width height`，归一化坐标）。
  - 数据转换脚本 `xmltoyolo.py` 中有 **classes 列表**，必须与 `bamboo.yaml` 的 `names` 保持一致。当前仓库两处有轻微不一致（`bamboo.yaml` 的 `虫眼` vs `xmltoyolo.py` 的 `虫洞`） — 代理在改动前应提示人工确认并统一。
  - 路径风格：仓库内脚本多使用 Windows 绝对路径与反斜杠；在修改或生成路径时优先使用 `os.path` 或 Python 原生跨平台分隔符以提高健壮性。

- **常用运行/调试命令（可直接使用）**:
  - 激活虚拟环境（示例终端输出显示使用 `pytorch_env`）：
    - `conda activate pytorch_env`
  - 使用存在脚本训练（短）:
    - `python YoloTest1.py`
  - 使用 Ultralytics CLI（等价方式）:
    - `yolo task=detect mode=train model=yolov8n.pt data=bamboo.yaml epochs=50 imgsz=640`
  - 运行标签转换并生成 `data/` 目录结构：
    - `python xmltoyolo.py`（请先确认 `input_dir`、`input_images_dir`、`output_dir` 与 `classes` 设置）
  - 评估已训练模型：
    - `python val.py`（会加载 `runs/detect/bamboo_exp/weights/best.pt` 并打印 mAP）

- **易错点 / 变更前检查清单（代理在修改前应核对）**:
  - 确认 `bamboo.yaml` 的 `path` 是否指向仓库中的 `data`（路径须能被训练脚本访问）。
  - 检查 `xmltoyolo.py` 的 `classes` 与 `bamboo.yaml` 的 `names` 是否一致，若不一致需同步并重新转换标签。
  - 注意代码中存在若干手写模块（如 `yolo_exp.py` 中的自定义层），有时含缩进/实现错误：在自动改动这些模块前先运行语法检查与单元执行以避免引入中断性错误。
  - Windows 路径与 YAML：`bamboo.yaml` 示例使用双反斜线转义。若将路径改为单斜线（`/`）通常也能被 Python/Ultralytics 识别，推荐统一为相对路径或使用 `os.path` 生成绝对路径。

- **修改建议与优先任务（给代理的明确动作）**:
  - 在修改训练/数据脚本前，先运行 `python -m pyflakes <file>` 或 `python -m py_compile <file>` 做快速语法检查。
  - 若要更新类别列表, 先同步 `bamboo.yaml` 与 `xmltoyolo.py`，然后重新运行 `xmltoyolo.py` 生成标签并检查 `data/labels/*` 对应文件数是否与 `data/images/*` 对应。
  - 对 `runs/` 的输出引用（例如 `runs/detect/bamboo_exp/weights/best.pt`）应以参数化方式处理，不要硬编码路径在新工具中。

- **快速定位示例**:
  - 训练入口（短）： [YoloTest1.py](YoloTest1.py)
  - 数据集定义： [bamboo.yaml](bamboo.yaml)
  - 标签转换： [xmltoyolo.py](xmltoyolo.py)
  - 验证脚本： [val.py](val.py)
  - 数据目录： `data/images/`, `data/labels/`

如果你希望我把 `bamboo.yaml` 的路径改为相对路径、或修复 `xmltoyolo.py` 与 `bamboo.yaml` 的类别不一致问题，我可以继续修改并运行快速检查。请告诉我优先项或补充任何缺失的运行约定。
