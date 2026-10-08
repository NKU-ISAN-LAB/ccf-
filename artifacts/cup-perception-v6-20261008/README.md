# v6：工位配准与杯沿识别的鲁棒性修正

目标不变：杯筒出口下方待取杯子的蓝色露出部分及可见下沿。
候选实现，不是新训练的 YOLO 权重；此次发布保留仓库中的 v5。
没有访问已关闭的虚拟机，也没有修改另一台连接相机的电脑。

**尚未解除对封口机配准的依赖：两条配准路径均失败时，仍然不输出
杯子位置。直接识别杯子、工位仅作辅助的架构尚未实现。**

## 对实时日志的判断

用户提供的 v5 结果中，7 个参考都为 `station_not_verified`。这只证明
当帧工位配准未通过，不证明相机断开，也无法区分匹配少、内点比例低、
几何误差大、支持区域出界或外观相关性不足。没有取得该帧原始 RGB，
不能声称本次修改已解决那个实时场景。

## 修改

- 保留同样的 7 张参考图片和标注，没有把 0032/0038/0045/0047/0049
  加入参考库。先尝试原配准；失败后独立使用下部机架平面特征配准，
  避免上方膜卷和机架不共面造成的干扰。原几何和外观通过阈值未降低。
- 原杯子模板严格检查仍保留。它失败时，直接在原始图像像素中验证
  蓝色杯身、白色下沿、弧形边界及杯筒出口邻近关系，而不是把杯子
  当作机架同一平面去校正。
- 验证通过时用实际观察的杯沿细化像素位置，减小不同参考切换导致的
  像素跳动。蓝色分割采用标定亮度和曝光修正亮度两种候选，每个候选
  仍需通过完整几何检查。两个空间分离的候选会拒绝输出。
- 保留连续三新帧、无效即清空、不重复使用旧坐标等边界。
- 新增 `registration_diagnostics`：每个参考、两条配准路径的
  `failed_gate`、匹配数、内点数/比例、空间分布、尺度/角度、误差、
  外观相关性及阈值。通过前即失败的后续指标不会伪造填零。
- `input_diagnostics` 记录输入分辨率、灰度统计、模糊度指标及解码
  像素哈希，用于核对究竟处理了哪个画面；不自动保存实时相机图像。
- `implementation` 记录 OpenCV/NumPy 版本、代码及参考清单校验和，
  便于核对不同电脑上 45/50 与 46/50 等回放差异，不直接推断代码装错。

## 本机结果

- 原始数据开发回放：v5 45/50 → v6 50/50；非参考图 43/43。
- 8 个源画面 × 原图、变暗、变亮、轻微模糊、JPEG 压缩、平移，共
  48 个开发扰动样本：v5 20/48 → v6 48/48。
- 扰动后候选点相对原帧（平移补偿后）的最大位移约 2.51 像素。
  这是重复性比较，不是与真实抓取点的误差。
- 59 项负例/输入检查通过；另外 8 个杯沿遮挡检查通过。
- 连续帧和回归单元测试：10 项通过。

这些是同一批数据上的开发回放及合成扰动，不是独立场景准确率。
没有真实空杯筒/机械手遮挡的全面实测；对不同杯型、不同颜色、严重
遮挡、过暗或严重模糊画面不承诺检出。没有当时实时失败的 RGB，尚未
验证另一台电脑的现场效果。输入仅支持该参考相机 `CP2L863000RG`
的 640×480 BGR 图；不要伪造序列号或随意缩放图像绕过校验。

输出仍是二维候选像素。深度未对齐且没有已验证机器人外参，所有
三维/抓取位姿为 null，`robot_motion_ready=false`。

## 运行与集成

无需 GPU 或新增库，继续使用 OpenCV/NumPy 环境：

```bash
cd artifacts/cup-perception-v6-20261008
# 在已有的兼容环境中运行；或在独立虚拟环境安装 requirements.txt。
OPENBLAS_NUM_THREADS=1 python recognize.py \
  /path/to/current-full-frame.jpg --camera-serial CP2L863000RG --output preview.jpg

OPENBLAS_NUM_THREADS=1 python verify.py --data /path/to/CCF数据集-20261008-120100
OPENBLAS_NUM_THREADS=1 python stress_check.py --data /path/to/CCF数据集-20261008-120100
OPENBLAS_NUM_THREADS=1 CUP_TEST_DATA=/path/to/CCF数据集-20261008-120100 \
  python -m unittest test_boundaries test_robustness -v
python render_report.py --data /path/to/CCF数据集-20261008-120100
```

若另一台电脑的发布器调用 `Detector.predict`，运行接口与 v5 相同。
候选替换文件是 `recognize.py` 和新增的 `observed_rim.py`；现有
`reference/` 校验内容不变。应先备份旧包，再由现场电脑只重启识别
进程使 Python 加载新版，不能因本机生成文件就声称现场已更新。
发布器需保留 `result.registration_diagnostics`，并把外层版本标记
更新到 v6，实际返回 `recognition_revision=observed-rim-v6-20261008`。
每个工作线程单独持有一个 Detector 实例，避免共享诊断状态。

`verification.json`、`stress-report.json` 为验证报告；`replay.json`
有每帧及配准细节；`full-frame-grid-1.jpg` 和 `full-frame-grid-2.jpg`
统一用完整原图显示，成功帧不再单独裁切放大。
