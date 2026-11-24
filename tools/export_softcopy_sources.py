# -*- coding: utf-8 -*-
"""
软著申请辅助：导出软件源代码鉴别材料

按《计算机软件著作权登记》惯例，将全部源代码合并导出为带页码标记的
文本文档（每页约 50 行），便于直接排版为 PDF 提交前 30 页 + 后 30 页。

用法：
    python tools/export_softcopy_sources.py
输出：
    docs/源代码文档.txt
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 纳入鉴别材料的源文件（相对路径，按软件架构顺序排列）
SOURCE_FILES = [
    "android/app/src/main/AndroidManifest.xml",
    "android/app/src/main/java/com/poseadvisor/app/MainActivity.kt",
    "android/app/src/main/java/com/poseadvisor/app/camera/CameraController.kt",
    "android/app/src/main/java/com/poseadvisor/app/camera/FrameBuffer.kt",
    "android/app/src/main/java/com/poseadvisor/app/pose/PoseModel.kt",
    "android/app/src/main/java/com/poseadvisor/app/pose/PoseLandmarkerHelper.kt",
    "android/app/src/main/java/com/poseadvisor/app/pose/PoseMapper.kt",
    "android/app/src/main/java/com/poseadvisor/app/pose/PoseComparator.kt",
    "android/app/src/main/java/com/poseadvisor/app/pose/PoseLibrary.kt",
    "android/app/src/main/java/com/poseadvisor/app/scene/SceneClassifier.kt",
    "android/app/src/main/java/com/poseadvisor/app/ui/SkeletonOverlayView.kt",
    "android/app/src/main/java/com/poseadvisor/app/ui/PoseStripAdapter.kt",
    "tools/generate_pose_library.py",
    "CNN_model.py",
    "pose_utils.py",
    "mediapipe_to_pose_vector.py",
    "infer_pose.py",
]

LINES_PER_PAGE = 50


def main():
    pages = []
    current = []
    total_lines = 0

    for rel in SOURCE_FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            print(f"[skip] 未找到: {rel}")
            continue
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        header = f"// ===== 文件: {rel} ====="
        block = [header, ""] + lines + [""]
        for line in block:
            current.append(line)
            total_lines += 1
            if len(current) >= LINES_PER_PAGE:
                pages.append(current)
                current = []
    if current:
        pages.append(current)

    out_path = os.path.join(ROOT, "docs", "源代码文档.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("PoseAdvisor 智能拍照姿势顾问 V1.0 源代码\n")
        f.write(f"（共 {len(pages)} 页，{total_lines} 行）\n\n")
        for i, page in enumerate(pages, 1):
            f.write(f"----- 第 {i} 页 -----\n")
            f.write("\n".join(page))
            f.write("\n\n")

    print(f"[softcopy] 已导出 {len(pages)} 页 / {total_lines} 行 → {out_path}")
    print("[softcopy] 提交时通常取前 30 页与后 30 页排版为 PDF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
