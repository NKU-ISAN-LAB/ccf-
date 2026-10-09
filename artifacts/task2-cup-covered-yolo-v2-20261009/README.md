# cup_covered 独立检测模型 v2（2026-10-09）

仅一个类别 `cup_covered`，与 `cup`、`target`、`exposed_cup` 不同。
人工框主要覆盖杯口封膜/盖面，不是整杯；输出是这些区域的二维框和中心，不是抓取坐标。
支持多杯同时检测，返回全部候选，不自动选择抓取对象，不控制机器人。

## 来源与改进

使用 `labels_my-project-name_2026-10-09-04-37-50.zip`，配对 `CCF数据集-20261009-154730`。
20 张图片、59 个封膜区域标注。0001–0012 训练，0013–0015 开发验证，0016–0020 保留回归。
第一轮从 COCO YOLO11n 独立训练，未使用已有杯子或杯托的模型权重。
第一轮发现将一个敞口杯误认为封膜杯，因此保留初始候选，另在此目录加入真实来源负样本微调。

**敞口杯与空杯托仅作为“没有 cup_covered”的负样本，不加入 cup 或 target 类别，也不改变原模型。**

- 正样本训练派生 42 张：12 原图、12 环境移除、12 局部放大、6 合成移除封膜图。
- 另增加 10 张真实来源负样本/裁剪，来自旧视频 episode_000000 和 episode_000020。
- 开发验证另加 5 张 episode_000010 的负样本/裁剪。
- episode_000031 的 5 张负样本/裁剪仅用于末尾检查，不参与训练和阈值选择。
- 已有 5 张封膜杯测试图在初版评估过，本轮只称复用回归，不冒充全新的独立测试。

训练来源和空标签见 `negative-source-audit.json`。只有 4 张真实负例来源图，裁剪数量不等于独立场景数量。
数据来自少量场景；不能从这些结果推断现场任意角度或遮挡条件都可靠。

## 最终离线结果

初始训练 31 轮后早停，本轮负样本微调 14 轮后早停，最终打包第 6 轮 best 权重。
默认阈值 **0.40**，只根据训练派生图选择，不人为放大置信度。
目标匹配采用一对一 IoU≥0.5；一个检测框不会重复计为两个杯子，额外框计误检。

| 检查 | 结果 |
|---|---|
| 20 张原图回放 | 58/59 个目标检出，0 个额外框；19/20 张全部正确 |
| 3 张正样本开发验证 | 8/9 个目标检出，0 个额外框 |
| 5 张保留正样本回归 | 15/15 个目标检出，0 个额外框；平均匹配 IoU .903 |
| 保留图的 50 次变换回归 | 150/150 个目标检出，0 个额外框 |
| 23 张合成负样本/空白检查 | 均未检出 |
| 真实来源负样本：训练/验证/保留组 | 10/10、5/5、5/5 均未误检 |
| 初版敞口杯失败的回归检查 | 4/4 均未误检（来源现已进入训练，不是独立测试） |
| PT / ONNX 输出一致性 | 20/20 原图通过 |
| 数据及接口单元测试 | 14 项通过 |

**未解决：0013 中被遮挡、仅露出窄条的封膜区域仍漏检。**
不修改该人工标签，不删除失败图片，也不依据这张图降低阈值。
少量负样本来源图的裁剪检查，不能证明已全面区分所有敞口杯或杯托；正样本回归也不是现场成功率。
初版 `0019_jpeg60` 的额外误检在本轮回归未再出现。

[全部原图结果](full-frame-grid.jpg) · [保留的遮挡失败](failure_0013_occluded.jpg) · [完整报告](verification.json)

## 使用

```bash
cd artifacts/task2-cup-covered-yolo-v2-20261009
python recognize.py \
  /绝对路径/输入.jpg --output /绝对路径/预览.jpg --json-output /绝对路径/结果.json
```

一起使用 `recognize.py`、`runtime-config.json`、`weights/`，不能装进 cup+target 的旧入口。
`detections` / `objects.cup_covered.candidates` 返回所有目标；`covered_cup_centers_uv` 返回二维框中心列表。
仅唯一候选时填写 `single_covered_cup_center_uv`；多个杯子不会擅自选一个来抓。
`seal_quality_verified=null`：只识别外观，不证明封口质量。
`grasp_pose=null`、`robot_motion_ready=false`；本次 GitHub 发布不连接机器人，不自动部署到观测电脑。

## 验证及复现

阈值仅用训练集选择；在开发验证上比较 best/last 的多目标 F1、全部正确帧数和匹配框 IoU。
`checkpoint-selection.json` 记录选择过程；`verification.json` 记录原图/变换回归及失败。
`other-object-checks.json` 分开记录训练、验证和保留负例；`prior-open-cup-checks.json` 是旧误检回归，不是未见测试。
`export-verification.json` 检查 PT/ONNX 一致性。所有原图和用户标签不变。

推理只需本目录代码、配置、权重及 `requirements.txt` 中的依赖；`test_runtime.py` 不需要原始数据。

训练/数据审计脚本保留本机实验路径，需要初始候选目录、原图及旧数据审计文件。
原始数据、初始候选权重与派生训练目录没有随本次发布上传，**并非复制仓库即可完整重训**；
复现时须先准备这些输入并调整脚本路径，`test_dataset.py` 也需要这些输入。

```bash
python prepare.py
python -m unittest test_dataset test_runtime -v
python train.py
python select_checkpoint.py
python export_model.py
python calibrate.py
python evaluate.py
python evaluate_negatives.py
python check_other_objects.py
python verify_export.py
```
