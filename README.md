# PoseAdvisor · 智能拍照姿势顾问

一款**端上实时**的 Android 拍照辅助应用：在相机取景过程中实时检测人体姿态，结合离线生成的姿势库为你推荐适合当前场景的拍照姿势，用"幽灵骨架"叠加在取景画面上引导摆姿，逐肢体打分纠错，并支持**零快门延迟**拍照。

## 核心功能

| 功能 | 说明 |
|---|---|
| 实时取景 | CameraX 预览，720p 流畅帧率 |
| 姿势推荐 | 基于场景条件 VAE 离线生成的精修姿势库，4 大类别（气质站姿 / 坐姿低机位 / 手势特写 / 舒展动态）共 40 组姿势 |
| 幽灵骨架引导 | 半透明骨架叠加在取景画面上，所见即所得 |
| 实时匹配反馈 | MediaPipe PoseLandmarker 端上检测 33 关键点，逐肢体段着色（达标青 / 偏差橙），0-100 匹配度打分 |
| 纠正提示 | 自动定位偏差最大的肢体段，给出方向性提示（如"请把左前臂向上调整"） |
| 场景感知 | 轻量画面统计（亮度/绿色占比/天空占比）毫秒级识别户外自然 / 天空 / 室内等场景，自动切换推荐类别 |
| 零快门延迟拍照 | 基于预览帧环形缓冲的即时成像架构，按下快门直接从内存最新帧编码 JPEG，无相机再捕获延迟 |
| 前后摄切换 | 前置镜头自动镜像适配 |

## 技术架构

```
┌───────────────────────── Android 端（实时）────────────────────────┐
│ CameraX 预览帧流 ─┬─► MediaPipe PoseLandmarker（GPU/CPU）          │
│                   │        └─► 33 关键点 → 25 关节 75 维向量        │
│                   │                 └─► 姿势比对引擎                 │
│                   │                      （归一化 + 肢体段方向角误差） │
│                   │                        └─► 骨架着色 / 打分 / 提示 │
│                   └─► FrameBuffer 环形缓冲 ─► 快门 → 即时 JPEG 成像  │
│ Assets 内置姿势库（pose_library.json + 骨架缩略图）                 │
└────────────────────────────────────────────────────────────────────┘
┌───────────────────────── PC 端（离线，一次生成）───────────────────┐
│ 场景条件 VAE（Scene-Conditioned Pose VAE）                         │
│   CLIP ViT-B/32 场景特征 → 解码器采样 75 维姿势向量                 │
│   └─► 合法性过滤 → 几何规则分类 → 最远点多样性筛选                  │
│        └─► pose_library.json + 骨架缩略图（随 App 打包）            │
└────────────────────────────────────────────────────────────────────┘
```

## 仓库结构

```
├── android/                  # Android 应用（Kotlin + CameraX + MediaPipe）
│   └── app/src/main/
│       ├── java/com/poseadvisor/app/
│       │   ├── MainActivity.kt            # 主界面：相机/推荐/拍照总控
│       │   ├── camera/                    # 相机控制器 + 零延迟帧缓冲
│       │   ├── pose/                      # 姿态检测封装 / 33→25 关节映射 / 比对引擎 / 姿势库
│       │   ├── scene/                     # 轻量场景识别
│       │   └── ui/                        # 骨架叠加渲染 / 姿势卡条
│       ├── assets/                        # 姿势库 JSON + 缩略图 + MediaPipe 模型
│       └── res/                           # 界面与图标资源
├── tools/
│   └── generate_pose_library.py           # 姿势库离线生成管线
├── CNN_model.py                           # 场景条件 VAE 模型定义
├── infer_pose.py                          # 姿势提取/生成推理（研究代码）
├── pose_utils.py                          # 25 关节表示与可视化
├── mediapipe_to_pose_vector.py            # MediaPipe → 75 维向量转换
├── fit_smplx_to_pose75.py                 # SMPL-X 网格拟合（研究代码，需自备 SMPL-X 许可）
└── docs/                                  # 软件著作权申请材料
```

## 构建与运行

### 重新生成姿势库（可选，仓库已内置生成结果）

```bash
pip install torch transformers opencv-python numpy pillow matplotlib
python tools/generate_pose_library.py --checkpoint model/latest.pth
```

> `model/latest.pth` 为训练 checkpoint（约 590MB，未随仓库分发）。姿势库生成结果已提交至 `android/app/src/main/assets/`，不重新生成也可直接构建 App。

### 构建 APK

需要 JDK 17 + Android SDK（platforms;android-34, build-tools;34.0.0）：

```bash
cd android
./gradlew assembleRelease        # 或 assembleDebug
```

产物：`android/app/build/outputs/apk/`

## 许可说明

- 本仓库代码可用于学习与研究
- `fit_smplx_to_pose75.py` 依赖 SMPL-X 模型，SMPL-X 有独立许可协议，需自行从 [SMPL-X 官网](https://smpl-x.is.tue.mpg.de/) 申请，**本仓库不包含亦不分发 SMPL-X 文件**
- MediaPipe PoseLandmarker 模型文件遵循 Google MediaPipe 模型许可
