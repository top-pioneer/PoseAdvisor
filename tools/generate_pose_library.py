# -*- coding: utf-8 -*-
"""
PoseAdvisor 姿势库离线生成工具

利用已训练的场景条件 VAE（SceneConditionedPoseVAE）离线采样多样化人体姿势，
经过合法性过滤、几何规则分类与多样性筛选后，导出移动端可用的姿势库资源：
  1. pose_library.json  —— 25 关节归一化坐标 + 分类元数据
  2. assets_poses/*.png   —— 姿势骨架缩略图（供 App 姿势卡片使用）

输出坐标系与训练数据一致：MediaPipe 归一化图像坐标（x 右，y 下，z 深度）。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CNN_model import SceneConditionedPoseVAE  # noqa: E402
from pose_utils import NUM_JOINTS, vector_to_joints  # noqa: E402


def load_model(checkpoint_path, device, pose_dim=75, latent_dim=64):
    """加载训练好的 VAE（与 infer_pose.load_model 一致，避免间接依赖 mediapipe）。"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args")
    if args is not None:
        pose_dim = getattr(args, "pose_dim", pose_dim)
        latent_dim = getattr(args, "latent_dim", latent_dim)
    model = SceneConditionedPoseVAE(pose_dim=pose_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, pose_dim, latent_dim


def image_path_to_pil_rgb(path):
    """支持中文路径的图像读取。"""
    import cv2
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 25 关节语义（与 mediapipe_to_pose_vector 训练格式一致）
JOINT_NAMES = [
    "pelvis", "spine_lower", "spine_mid", "chest", "neck", "head",
    "unused_6", "ankle_left", "ankle_right", "unused_9",
    "shoulder_left", "shoulder_right", "hip_left",
    "elbow_left", "elbow_right", "hip_right",
    "wrist_left", "wrist_right", "hand_left", "hand_right",
    "foot", "unused_21", "nose", "eye_left", "eye_right",
]

# App 端渲染使用的解剖学骨架连接（剔除训练格式中的退化腿连接）
RENDER_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],          # 脊柱→头
    [3, 10], [10, 13], [13, 16], [16, 18],            # 左臂
    [3, 11], [11, 14], [14, 17], [17, 19],            # 右臂
    [0, 12], [12, 7],                                  # 左腿（髋→踝）
    [0, 15], [15, 8],                                  # 右腿（髋→踝）
    [5, 22],                                           # 头→鼻
]

CATEGORY_META = {
    "standing": {"label": "气质站姿", "order": 0},
    "sitting":  {"label": "坐姿低机位", "order": 1},
    "gesture":  {"label": "手势特写", "order": 2},
    "dynamic":  {"label": "舒展动态", "order": 3},
}


def is_valid_pose(joints: np.ndarray) -> bool:
    """过滤明显异常的采样结果。"""
    if not np.isfinite(joints).all():
        return False
    xy = joints[:, :2]
    # 关键点应大体落在画面附近（允许少量越界）
    if (xy < -0.6).any() or (xy > 1.6).any():
        return False
    pelvis, neck = joints[0], joints[4]
    torso = np.linalg.norm(neck[:2] - pelvis[:2])
    if torso < 0.05:  # 人体过小/坍缩
        return False
    # 头部应在骨盆上方（y 向下为正）
    if joints[5][1] > pelvis[1]:
        return False
    return True


def normalize_pose(joints: np.ndarray) -> np.ndarray:
    """骨盆对齐原点 + 躯干长度归一，消除画面位置与比例差异（与 App 端比对一致）。"""
    out = joints.copy()
    pelvis = out[0, :2].copy()
    torso = np.linalg.norm(out[4, :2] - pelvis) + 1e-6
    out[:, :2] = (out[:, :2] - pelvis) / torso
    return out


def kmeans_numpy(feats: np.ndarray, k: int, iters: int = 50, seed: int = 0):
    """轻量 KMeans（k-means++ 初始化），避免引入 sklearn 依赖。"""
    rng = np.random.default_rng(seed)
    n = len(feats)
    # k-means++ 初始化
    centers = [feats[rng.integers(n)]]
    for _ in range(k - 1):
        d = np.min(np.linalg.norm(feats[:, None, :] - np.array(centers)[None], axis=-1), axis=1)
        probs = d ** 2
        probs /= probs.sum() + 1e-12
        centers.append(feats[rng.choice(n, p=probs)])
    centers = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = np.linalg.norm(feats[:, None, :] - centers[None], axis=-1)
        new_labels = dist.argmin(axis=1)
        if (new_labels == labels).all():
            break
        labels = new_labels
        for c in range(k):
            if (labels == c).any():
                centers[c] = feats[labels == c].mean(axis=0)
    return labels, centers


def cluster_features(cluster_poses: list):
    """簇均值几何特征：抬手程度、腕距比、腿长（归一化躯干单位）。"""
    mean = np.mean(cluster_poses, axis=0)
    shoulder_y = (mean[10][1] + mean[11][1]) / 2
    hip_y = (mean[12][1] + mean[15][1]) / 2
    ankle_y = (mean[7][1] + mean[8][1]) / 2
    wrist_min_y = min(mean[16][1], mean[17][1])
    wrist_span = abs(mean[16][0] - mean[17][0])
    shoulder_span = abs(mean[10][0] - mean[11][0]) + 1e-6
    return {
        "wrist_raise": shoulder_y - wrist_min_y,      # 越大抬手越高
        "wrist_span_ratio": wrist_span / shoulder_span,
        "leg_drop": ankle_y - hip_y,                   # 越大腿越伸展
    }


def assign_categories(clusters: list) -> list:
    """按簇特征排序分配类别，保证 4 个类别都被覆盖（数据驱动，免阈值）。"""
    feats = [cluster_features(c) for c in clusters]
    cats = [None] * len(clusters)
    remaining = set(range(len(clusters)))

    def take(key, reverse):
        best = max(remaining, key=lambda i: feats[i][key]) if reverse else \
            min(remaining, key=lambda i: feats[i][key])
        remaining.discard(best)
        return best

    # 抬手最高 → 手势；腕距最大 → 动态；腿最短 → 坐姿；其余 → 站姿
    cats[take("wrist_raise", True)] = "gesture"
    cats[take("wrist_span_ratio", True)] = "dynamic"
    cats[take("leg_drop", False)] = "sitting"
    for i in remaining:
        cats[i] = "standing"
    return cats


def pose_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.reshape(-1) - b.reshape(-1)))


def diversify(candidates, max_keep):
    """贪心最远点采样，保证姿势库多样性。"""
    if len(candidates) <= max_keep:
        return candidates
    chosen = [candidates[0]]
    rest = candidates[1:]
    while rest and len(chosen) < max_keep:
        d = [min(pose_distance(c, k) for k in chosen) for c in rest]
        idx = int(np.argmax(d))
        chosen.append(rest.pop(idx))
    return chosen


def render_thumbnail(joints: np.ndarray, out_path: str, size=512):
    """渲染姿势骨架缩略图（深色底 + 品牌蓝线条）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#171A21")
    fig.patch.set_facecolor("#171A21")

    xs, ys = joints[:, 0], joints[:, 1]
    for a, b in RENDER_EDGES:
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color="#5B8CFF",
                linewidth=7, solid_capstyle="round", alpha=0.95)
    key = [0, 3, 4, 5, 10, 11, 13, 14, 16, 17, 12, 15, 7, 8]
    ax.scatter(xs[key], ys[key], s=90, c="#FFFFFF", zorder=5,
               edgecolors="#5B8CFF", linewidths=2)

    # 自适应取景：以主体关节（躯干+四肢）为中心，避免鼻/眼/脚等游离点拉偏构图
    cx, cy = xs[key].mean(), ys[key].mean()
    span = max(xs[key].max() - xs[key].min(),
               ys[key].max() - ys[key].min()) / 2 * 1.25 + 1e-3
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy + span, cy - span)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="model/latest.pth")
    ap.add_argument("--image-dir", default="tools/cond_images")
    ap.add_argument("--samples-per-image", type=int, default=8)
    ap.add_argument("--max-per-category", type=int, default=10)
    ap.add_argument("--out-json", default="android/app/src/main/assets/pose_library.json")
    ap.add_argument("--out-thumbs", default="android/app/src/main/assets/poses")
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pose-lib] device={device}")

    model, _, _ = load_model(args.checkpoint, device)

    images = sorted(
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )
    if not images:
        print("[pose-lib] 未找到条件图片", file=sys.stderr)
        return 2
    print(f"[pose-lib] 条件图片 {len(images)} 张")

    valid = []
    total = 0
    for img_path in images:
        pil = image_path_to_pil_rgb(img_path)
        with torch.no_grad():
            samples = model.generate([pil], num_samples=args.samples_per_image)
        for vec in samples[0].cpu().numpy():
            joints = vector_to_joints(vec)
            total += 1
            if not is_valid_pose(joints):
                continue
            valid.append(joints)
        print(f"[pose-lib] {os.path.basename(img_path)} 完成，累计候选 {total}，有效 {len(valid)}")

    if len(valid) < 4:
        print("[pose-lib] 有效姿势过少，无法聚类", file=sys.stderr)
        return 3

    # 归一化后 KMeans 聚类为 4 簇，再按簇几何特征命名类别
    normed = [normalize_pose(j) for j in valid]
    feats = np.stack([n[:, :2].reshape(-1) for n in normed])
    k = min(4, len(normed))
    labels, _ = kmeans_numpy(feats, k=k, seed=args.seed)

    buckets = {key: [] for key in CATEGORY_META}
    clusters = [[normed[i] for i in np.where(labels == c)[0]] for c in range(k)]
    cat_of = assign_categories(clusters)
    for c in range(k):
        cat = cat_of[c]
        # 库存储归一化坐标（骨盆原点 + 躯干单位长），App 端直接使用
        buckets[cat].extend(clusters[c])
        print(f"[pose-lib] 簇 {c}: {len(clusters[c])} 个 → 类别 {cat}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(args.out_thumbs, exist_ok=True)

    poses_out = []
    counters = {k: 0 for k in CATEGORY_META}
    for cat, meta in sorted(CATEGORY_META.items(), key=lambda kv: kv[1]["order"]):
        cands = diversify(buckets[cat], args.max_per_category)
        print(f"[pose-lib] 类别 {cat}: 候选 {len(buckets[cat])} → 入选 {len(cands)}")
        for joints in cands:
            counters[cat] += 1
            pid = f"{cat}_{counters[cat]:02d}"
            thumb_name = f"{pid}.png"
            render_thumbnail(joints, os.path.join(args.out_thumbs, thumb_name))
            poses_out.append({
                "id": pid,
                "category": cat,
                "label": f"{meta['label']}·{counters[cat]:02d}",
                "thumbnail": f"poses/{thumb_name}",
                "joints": np.round(joints, 4).tolist(),
            })

    library = {
        "version": 1,
        "generator": "SceneConditionedPoseVAE (offline)",
        "jointNames": JOINT_NAMES,
        "renderEdges": RENDER_EDGES,
        "categories": [
            {"id": k, "label": v["label"]}
            for k, v in sorted(CATEGORY_META.items(), key=lambda kv: kv[1]["order"])
        ],
        "poses": poses_out,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(library, f, ensure_ascii=False)
    print(f"[pose-lib] 完成：{len(poses_out)} 个姿势 → {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
