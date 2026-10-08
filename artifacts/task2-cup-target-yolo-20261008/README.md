# 任务二：cup + target 直接检测模型

独立于任务一 exposed_cup 模型，使用用户
`labels_my-project-name_2026-10-08-08-16-04.zip` 标注训练。

- 类别 0 `cup`：用户标出的任务二杯子，包含杯口和可见杯身。
- 类别 1 `target`：用户标出的圆形承杯环/放杯槽，不是整台封口机。
- 直接在整张图片上检测两类，不使用工位配准、参考图库或固定坐标兜底。
- 使用类别区分的 NMS，杯子和杯托高度重叠时不会相互压掉。
- 任一类别可以单独检出；多个同类候选不自动选择最高分作为操作目标。
- cup 沿用用户圈定的任务对象；新画面中的其他杯状物没有全部标注，
  不能把此模型的结果解释为对所有杯子类型的通用识别能力。

## 数据审计

60 张选图中，ZIP 实际包含 56 张标签，共 56 个 cup 和 56 个有效 target。
006、010、026、040 没有标签，排除训练和计分，不擅自当作真实负例。
024 第三行是宽 0.000642、高 0 的无效 target 框，只在训练副本中剔除。
用户 ZIP、原图和任务一模型均未改动。详见 annotation-audit.json。

旧视频按整个 episode 划分，新图片的一组斜视角整体留作开发验证：
43 张原始训练图、13 张开发验证图。训练加上 43 张去除无关背景、只保留
两目标邻域的正例，以及 11 张合成移除两目标的负例，共 97 张训练输入。
所有增强均只来自训练侧，不把验证图增强后混入训练。

开发验证参与最佳权重选择，不是独立测试。新图来自同次拍摄、旧视频背景
相似，结果不能代表独立现场准确率。没有真实空场景的全面验证。

## 本次训练与验证结果

- CPU 完成 60 轮，采用第 52 轮最佳权重；训练约 11.2 分钟。
- 配套阈值：cup 0.20、target 0.25，只使用训练侧样本选择。
- 43 张训练原图，两类均正确定位 43/43。
- 13 张分组开发验证：cup 13/13，target 12/13（正确定位标准 IoU >= 0.5）。
- 其中新增视角 5 张，两类均为 5/5；旧视频 8 张为 cup 8/8、target 7/8。
- 两框不重叠的 9 张验证图，两类均正确；重叠的 4 张为 cup 4/4、target 3/4。
- 018 的杯托预测框偏下、偏小，与手动框 IoU 约 0.358；作为定位失败
  保留，未修改用户框或把它加入训练。见 failure-018-comparison.jpg。
- 验证侧 78 个合成变化：cup 正确 78/78、target 正确 72/78，同一 018
  的杯托定位仍失败；59 项合成负例/空白输入检查通过，不等于真实空场景验证。
- 16 项接口/数据边界测试通过；PyTorch/ONNX 在 56 张标注原图上输出一致。
- 单张 CPU 回放中位耗时约 34 ms，仅是本机测试，不保证另一台电脑帧率。

![完整原图回放示例](full-frame-grid-3.jpg)

## 运行

在包含 requirements.txt 依赖的环境中执行：

```bash
python recognize.py /path/to/current-full-frame.jpg \
  --output preview.jpg --json-output recognition.json
```

同时保留 `recognize.py`、`weights/best.pt` 和 `runtime-config.json`。
配置与权重 SHA256 绑定，两个类别各用训练侧选择的阈值；不要沿用任务一
阈值，也不要用旧的封口机配准程序只换权重。明确指定 `--confidence`
会覆盖两个类别的阈值，仅用于有记录的调试。

输入 uint8 BGR，内部等比例 letterbox 到 640×640；返回原始图片像素坐标。
640×480 是开发数据尺寸，其他相机/分辨率效果未验证。相机序列号只记录
来源，不作为配准门槛。每个工作线程持有一个 Detector，避免每帧加载权重。

```python
from recognize import Detector, annotate
detector = Detector()
result = detector.predict(frame_bgr, camera_serial)
preview = annotate(frame_bgr, result)
```

`objects.cup`、`objects.target` 分别包含 detected/count/unique/candidates。
仅有唯一该类目标时才提供 `cup_center_uv` 或 `target_center_uv`，它们
是**矩形框中心**，不是精确杯底中心或杯托几何中心。
`both_unique_2d` 只表示两类各有一个框；`cup_target_bbox_iou` 仅为二维
框重叠度，不证明杯子已插入、放稳或完成封口。

`stable_2d` 默认 false，本入口不做时间滤波；实时接入需使用真实帧号/
时间戳另做连续帧验证。`placement_confirmed`、`grasp_pose` 为 null，
`robot_motion_ready=false`。没有输出三维抓取坐标、调用相机或机器人控制。

## 复现与报告

```bash
python prepare.py --source /path/to/task2-cup-sealing-60-20261008 \
  --labels /path/to/labels_my-project-name_2026-10-08-08-16-04.zip
python train.py --pretrained /path/to/generic/yolo11n.pt --epochs 60
# 把 runs/cup-target/weights/best.pt 复制到 weights/best.pt 后：
python calibrate.py
python evaluate.py --images /path/to/task2-cup-sealing-60-20261008/images
python -m unittest test_runtime -v
python -m unittest test_dataset -v
```

prepare.py 拒绝覆盖已有 dataset，请在新的副本中准备复现。
本发布包不包含原始图片或生成的 dataset；先准备数据后才能运行
test_dataset.py。训练参数及审计记录中的本机绝对路径已改为可移植占位路径，
实际复现由 --source、--labels 和 --pretrained 参数提供路径。
初始化为通用 COCO YOLO11n，而非旧螺母、杯筒或任务一权重。
使用本机 CPU；没有更改不可用的 NVIDIA 驱动。
verification.json 分开报告训练回放、分组验证、每类 precision/recall、
合成扰动和合成负例。unlabeled-diagnostics.json 仅记录缺标签四张的预测，
不拿它们计算正确率。人工标注及预测总览始终用完整原图，不混用局部放大。

如需 ONNX，使用 `python export_model.py` 后重新运行 calibrate.py 将
导出文件校验和加入配置，再用 `python verify_export.py --images ...`
检查两种格式。当前导出固定输入 1×3×640×640，输出原始检测张量，
不含 NMS；独立接入务必按类别做 NMS，不能让重叠的 cup/target 互相删除。
本入口使用 ONNX 时需要额外安装 onnxruntime。

本次作为独立模型包发布，保留任务一及历史版本；未部署另一台电脑，
未修改机器人控制代码。
