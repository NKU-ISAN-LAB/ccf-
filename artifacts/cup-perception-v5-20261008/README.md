# 2026-10-08 杯筒出口识别包

这是 CPU 参考图匹配算法，不是 YOLO 训练权重。没有 `.pt` 或 `.onnx`
新权重；实际运行需要 `recognize.py` 和 `reference/` 中的图片及清单。

只寻找杯筒出口下方的蓝色杯子下部及其可见下沿。不将杯筒、印刷区、
支架或桌面杯子当作目标。新数据相机序列号是 `CP2L863000RG`，
640×480；旧版相机 `CP2L863000KV` 不能直接套用。

## 运行

在本目录下执行（验证环境为 Python 3.12）：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python recognize.py reference/0001.jpg \
  --camera-serial CP2L863000RG --output preview.jpg
.venv/bin/python -m unittest test_boundaries -v
```

其他画面同样通过位置参数输入，不能把旧相机或其他视角伪装成此相机。
返回 JSON 中 `candidate_valid_2d` 为是否检出，`visible_lower_rim_uv`
为大约 5 像素标注不确定度的可见下沿候选像素点，不是杯底三维中心。

单图调用不会宣称连续帧稳定：`stable_2d` 为 false。序列消费者需调用
`StabilityGate.update(result, frame_id, observed_time)`：3 个递增新帧、
间隔不超过 2 秒、位置偏差不超过 2 像素才稳定。重复、无效或间断帧
会清空历史。消费者仍须核对源图像新鲜度，不能重新发布旧点。

## 数据与验证

原始数据名：`CCF数据集-20261008-120100`。共 50 组彩色图、深度及元数据，
2026-10-08 12:05:28–12:06:53 采集。原数据无目标框；7 张参考图的框由
助手逐图视觉检查标注，沿用用户之前红框指定的语义。新坐标不是用户
再次审核的标注。图片 SHA-256 写入 `reference/manifest.json`。

参考帧：0001、0033、0034、0040、0041、0043、0046。

| 项目 | 开发回放结果 |
|---|---:|
| 旧版视觉算法 | 0/50 |
| 新版全部画面 | 45/50 |
| 新版非参考画面 | 38/43 |
| 前 30 张固定视角 | 30/30 |
| 固定序列稳定输出（扣除前两帧） | 28/28 |
| 负例与输入检查 | 54/54 |
| 连续帧边界单元测试 | 4/4 |

尚未通过 0032、0038、0045、0047、0049，输出无候选点。
不能把这些数字称为泛化准确率：参考帧在开发过程中选择，前 30 张高度
相似，整批不是独立测试集。45 个移除杯子的负例是基于已检查检出框的
合成遮挡；其余检查覆盖错误相机/格式、空白、噪声、倒置和其他视角。
没有真实空杯筒、机械手遮挡或不同杯型的充分验证。

没有上传完整原始数据集。持有原始数据时可复现：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python verify.py --data /path/to/CCF数据集-20261008-120100
# 只有需要从原数据重建参考副本时执行：
.venv/bin/python prepare.py --data /path/to/CCF数据集-20261008-120100
```

`verify.py` 重新生成本目录的回放报告及 `previews/`，不修改原始数据。

## 方法与边界

机架双向特征匹配、RANSAC 单应配准与几何/外观检查后，独立搜索杯子
局部平移、旋转和缩放。杯子灰度相关系数至少 0.82，整体边缘和下沿
边缘相关系数分别至少 0.65。多个参考给出互相矛盾的位置时拒绝输出。
没有“找不到就用固定坐标”的分支。

原始深度标记为 `depth_aligned_to_color=false`，没有可靠的机器人外参。
所有三维及抓取输出为 null，`robot_motion_ready=false`，不调用硬件。
本包尚未替换虚拟机的旧服务；部署前须核对新相机、消息接口和回滚方式。

## 文件

- `recognize.py`：识别与连续帧判定；`reference/`：运行参考资料。
- `reference_specs.json`：框、锚点与标注来源。
- `verification.json` / `replay.json`：统计与逐帧诊断。
- `baseline.json`：旧版回放；`negative-checks.json`：负例记录。
- `review-crops-0.jpg` / `review-crops-1.jpg`：全部画面的检测局部汇总。
- `test_assets/`：两张其他相机视角的负例；`test_boundaries.py`：状态测试。

本包不授予来源数据任何额外许可；使用参考图片时应遵守原数据和赛事的授权要求。
