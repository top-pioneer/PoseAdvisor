# -*- coding: utf-8 -*-
"""
真人剪影资产制作 + MLS 网格变形离线验证

1. 从 Wikimedia 公有领域剪影提取人体蒙版，裁剪并生成干净 alpha 资产
2. 定义骨骼锚点（基图坐标 + 骨骼附着规则），导出 anchors.json
3. 用与 App 端一致的 MLS-affine 算法把剪影变形到姿势库姿势，输出预览图

素材许可：Mikael Häggström "Human body silhouette"，Public Domain（Wikimedia Commons）
"""
import json
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "tools", "silhouette_src.png")
OUT_PNG = os.path.join(ROOT, "android", "app", "src", "main", "assets", "silhouette.png")
OUT_ANCHORS = os.path.join(ROOT, "android", "app", "src", "main", "assets", "silhouette_anchors.json")

# ---------------------------------------------------------------- 锚点定义
# name: (base_x, base_y, [jointA, jointB, s, r])
#   base_*：在原始剪影图上的归一化坐标（人工标注）
#   规则：anchor = A + (B-A)*s + perp(norm(B-A))*r*torsoLen；jointA/B 为 25 关节索引；直接关节则 A=B
PELVIS, SPINE_MID, CHEST, NECK, HEAD = 0, 2, 3, 4, 5
ANKLE_L, ANKLE_R = 7, 8
SHOULDER_L, SHOULDER_R, HIP_L, HIP_R = 10, 11, 12, 15
ELBOW_L, ELBOW_R, WRIST_L, WRIST_R = 13, 14, 16, 17

ANCHORS = [
    # 头颈（基坐标经蒙版逐行实测）
    ("head_top",  0.490, 0.010, [NECK, HEAD, 1.40, 0.0]),
    ("head_l",    0.420, 0.060, [NECK, HEAD, 0.85, 0.30]),
    ("head_r",    0.570, 0.060, [NECK, HEAD, 0.85, -0.30]),
    ("jaw_l",     0.440, 0.115, [NECK, HEAD, 0.45, 0.18]),
    ("jaw_r",     0.550, 0.115, [NECK, HEAD, 0.45, -0.18]),
    ("neck",      0.500, 0.155, [NECK, NECK, 0.0, 0.0]),
    # 肩臂
    ("sh_l",      0.280, 0.190, [SHOULDER_L, SHOULDER_L, 0.0, 0.0]),
    ("sh_r",      0.720, 0.190, [SHOULDER_R, SHOULDER_R, 0.0, 0.0]),
    ("armpit_l",  0.240, 0.235, [NECK, SHOULDER_L, 1.05, 0.10]),
    ("armpit_r",  0.760, 0.235, [NECK, SHOULDER_R, 1.05, -0.10]),
    ("elb_l",     0.220, 0.330, [ELBOW_L, ELBOW_L, 0.0, 0.0]),
    ("elb_r",     0.780, 0.330, [ELBOW_R, ELBOW_R, 0.0, 0.0]),
    ("wri_l",     0.160, 0.475, [WRIST_L, WRIST_L, 0.0, 0.0]),
    ("wri_r",     0.830, 0.475, [WRIST_R, WRIST_R, 0.0, 0.0]),
    ("fin_l",     0.080, 0.550, [ELBOW_L, WRIST_L, 1.20, 0.0]),
    ("fin_r",     0.920, 0.550, [ELBOW_R, WRIST_R, 1.20, 0.0]),
    # 躯干
    ("chest",     0.500, 0.270, [PELVIS, NECK, 0.72, 0.0]),
    ("waist_l",   0.310, 0.380, [PELVIS, NECK, 0.45, 0.28]),
    ("waist_r",   0.690, 0.380, [PELVIS, NECK, 0.45, -0.28]),
    ("pelvis",    0.500, 0.495, [PELVIS, PELVIS, 0.0, 0.0]),
    ("hip_l",     0.430, 0.500, [HIP_L, HIP_L, 0.0, 0.0]),
    ("hip_r",     0.570, 0.500, [HIP_R, HIP_R, 0.0, 0.0]),
    ("hipout_l",  0.280, 0.520, [PELVIS, HIP_L, 1.35, 0.0]),
    ("hipout_r",  0.720, 0.520, [PELVIS, HIP_R, 1.35, 0.0]),
    ("crotch",    0.500, 0.575, [NECK, PELVIS, 1.25, 0.0]),
    # 腿脚
    ("knee_l",    0.390, 0.655, [HIP_L, ANKLE_L, 0.50, 0.0]),
    ("knee_r",    0.625, 0.655, [HIP_R, ANKLE_R, 0.50, 0.0]),
    ("ank_l",     0.370, 0.875, [ANKLE_L, ANKLE_L, 0.0, 0.0]),
    ("ank_r",     0.670, 0.875, [ANKLE_R, ANKLE_R, 0.0, 0.0]),
    ("toe_l",     0.330, 0.965, [HIP_L, ANKLE_L, 1.12, 0.0]),
    ("toe_r",     0.720, 0.965, [HIP_R, ANKLE_R, 1.12, 0.0]),
]

BONE_TARGETS = [(SHOULDER_L, ELBOW_L, 0.63), (SHOULDER_R, ELBOW_R, 0.63),
                (ELBOW_L, WRIST_L, 0.53), (ELBOW_R, WRIST_R, 0.53),
                (HIP_L, ANKLE_L, 1.60), (HIP_R, ANKLE_R, 1.60)]


# ---------------------------------------------------------------- 与 App 一致的姿势处理
def normalize_pose(j):
    out = j.copy()
    torso = np.linalg.norm(out[NECK, :2] - out[PELVIS, :2]) + 1e-6
    out[:, :2] = (out[:, :2] - out[PELVIS, :2]) / torso
    # 解剖修正：髋线与骨盆中心齐平（生成数据的髋点偏低，会压折腿根网格）
    out[HIP_L, 1] = out[PELVIS, 1]
    out[HIP_R, 1] = out[PELVIS, 1]
    for a, b, target in BONE_TARGETS:
        d = out[b, :2] - out[a, :2]
        l = np.linalg.norm(d)
        if l < 1e-5:
            continue
        s = np.clip(target / l, 0.6, 4.0)
        out[b, :2] = out[a, :2] + d * s
    # 站姿约束：踝距不低于髋距的 1.8 倍（与引导剪影基型一致，避免腿根网格压折）
    hip_span = abs(out[HIP_L, 0] - out[HIP_R, 0])
    ank_span = abs(out[ANKLE_L, 0] - out[ANKLE_R, 0])
    min_span = max(hip_span * 1.8, 0.30)
    if ank_span < min_span:
        mid = (out[ANKLE_L, 0] + out[ANKLE_R, 0]) / 2
        sign = 1.0 if out[ANKLE_L, 0] >= out[ANKLE_R, 0] else -1.0
        out[ANKLE_L, 0] = mid + sign * min_span / 2
        out[ANKLE_R, 0] = mid - sign * min_span / 2
    # 手臂约束：肘/腕不得进入躯干侧边以内（剪影基型手臂垂于体侧，穿越躯干会压折胸腰网格）
    for sh, hp, el, wr in ((SHOULDER_L, HIP_L, ELBOW_L, WRIST_L),
                           (SHOULDER_R, HIP_R, ELBOW_R, WRIST_R)):
        side = 1.0 if out[sh, 0] >= out[PELVIS, 0] else -1.0
        for jt in (el, wr):
            y = out[jt, 1]
            denom = out[hp, 1] - out[sh, 1]
            t = np.clip((y - out[sh, 1]) / denom, 0.0, 1.0) if abs(denom) > 1e-5 else 0.0
            x_side = out[sh, 0] + (out[hp, 0] - out[sh, 0]) * t
            if side * (out[jt, 0] - x_side) < 0.06:
                out[jt, 0] = x_side + side * 0.06
    return out


def anchor_target(rule, pose):
    a_i, b_i, s, r = rule
    A = pose[a_i, :2]
    B = pose[b_i, :2]
    d = B - A
    n = np.linalg.norm(d)
    if n < 1e-6:
        return A
    perp = np.array([-d[1], d[0]]) / n
    return A + d * s + perp * r


# ---------------------------------------------------------------- MLS affine 变形
def mls_affine(v, base_pts, tgt_pts, alpha=1.0, eps=900.0):
    """v:(M,2) 网格点 base_pts/tgt_pts:(N,2) → 变形后 (M,2)
    eps 为权重平滑项（像素平方），防止靠近锚点时权重爆炸。"""
    N = base_pts.shape[0]
    out = np.empty_like(v)
    for k in range(v.shape[0]):
        p = v[k]
        dist2 = ((base_pts - p) ** 2).sum(axis=1)
        w = 1.0 / (dist2 + eps) ** alpha
        wsum = w.sum()
        p_star = (w[:, None] * base_pts).sum(axis=0) / wsum
        q_star = (w[:, None] * tgt_pts).sum(axis=0) / wsum
        Ph = base_pts - p_star
        Qh = tgt_pts - q_star
        M1 = np.einsum("i,ij,ik->jk", w, Ph, Qh)   # Σ wi Ph^T Qh
        M2 = np.einsum("i,ij,ik->jk", w, Ph, Ph)   # Σ wi Ph^T Ph
        # Schaefer MLS affine: M = (Σw Ph^T Ph)^{-1} · (Σw Ph^T Qh)
        M = np.linalg.inv(M2 + np.eye(2) * 1e-8) @ M1
        out[k] = (p - p_star) @ M + q_star
    return out


# ---------------------------------------------------------------- 剪影蒙版处理
def make_silhouette_asset():
    img = cv2.imread(SRC, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(SRC)
    h, w = img.shape[:2]
    print(f"原图 {w}x{h}, 通道 {img.shape[2] if img.ndim == 3 else 1}")

    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha0 = img[:, :, 3]
    else:
        bgr = img
        alpha0 = np.full((h, w), 255, np.uint8)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # 肤色/非白背景区域：饱和度>12 或 亮度<245，且原 alpha 可见
    mask = (((hsv[:, :, 1] > 12) | (hsv[:, :, 2] < 245)) & (alpha0 > 10)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # 最大连通域
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == idx).astype(np.uint8) * 255

    ys, xs = np.where(mask > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = 4
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w - 1, x1 + pad), min(h - 1, y1 + pad)
    crop = mask[y0:y1 + 1, x0:x1 + 1]
    # 边缘平滑 + 2 倍放大
    crop = cv2.GaussianBlur(crop, (5, 5), 0)
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, crop = cv2.threshold(crop, 100, 255, cv2.THRESH_BINARY)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    # 存为 RGBA：RGB 白 + alpha=蒙版（App 端用 PorterDuff SRC_IN 染成品牌色）
    rgba = np.zeros((crop.shape[0], crop.shape[1], 4), np.uint8)
    rgba[:, :, :3] = 255
    rgba[:, :, 3] = crop
    cv2.imwrite(OUT_PNG, rgba)
    ch, cw = crop.shape
    print(f"剪影资产 {cw}x{ch} → {OUT_PNG}")

    # 锚点基坐标 → 裁剪框归一化
    anchors_out = []
    for name, bx, by, rule in ANCHORS:
        cx = (bx * w - x0) / (x1 - x0 + 1)
        cy = (by * h - y0) / (y1 - y0 + 1)
        anchors_out.append({"name": name, "base": [round(cx, 4), round(cy, 4)], "rule": rule})
    with open(OUT_ANCHORS, "w", encoding="utf-8") as f:
        json.dump({"anchors": anchors_out}, f, ensure_ascii=False)
    print(f"锚点 {len(anchors_out)} 个 → {OUT_ANCHORS}")
    return crop, (x0, y0, x1, y1), (w, h)


# ---------------------------------------------------------------- 预览
def preview(crop, crop_box, src_wh):
    x0, y0, x1, y1 = crop_box
    w, h = src_wh
    ch, cw = crop.shape
    lib = json.load(open(os.path.join(ROOT, "android/app/src/main/assets/pose_library.json"), encoding="utf-8"))
    picks = ["standing_01", "standing_02", "sitting_01", "gesture_01", "dynamic_01", "dynamic_03"]
    samples = [p for p in lib["poses"] if p["id"] in picks]

    base_pts = []
    for name, bx, by, rule in ANCHORS:
        cx = (bx * w - x0) / (x1 - x0 + 1) * cw
        cy = (by * h - y0) / (y1 - y0 + 1) * ch
        base_pts.append([cx, cy])
    base_pts = np.array(base_pts)

    GH, GW = 12, 8
    gx, gy = np.meshgrid(np.linspace(0, cw - 1, GW), np.linspace(0, ch - 1, GH))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(12, 9), facecolor="#171A21")
    for ax, p in zip(axes.flat, samples):
        pose = normalize_pose(np.array(p["joints"])[:, :2])
        # MediaPipe 左/右语义相对画面是镜像的（面对镜头时其左手在画面右侧），
        # 与剪影标注方向相反，需镜像 x 后再配对锚点
        pose[:, 0] = -pose[:, 0]
        tgt = np.array([anchor_target(rule, pose) for _, _, _, rule in ANCHORS])
        # 目标锚点（躯干单位）→ 像素：身高对齐剪影高度
        span_y = tgt[:, 1].max() - tgt[:, 1].min()
        scale = (ch * 0.92) / max(span_y, 1e-3)
        ctr = tgt.mean(axis=0)
        tgt_px = (tgt - ctr) * scale + np.array([cw / 2, ch / 2])
        warped_grid = mls_affine(grid, base_pts, tgt_px)

        # 逐网格三角形仿射贴图（与 Android drawBitmapMesh 同一渲染语义）
        warped = np.zeros_like(crop)
        G_src = grid.reshape(GH, GW, 2)
        G_dst = warped_grid.reshape(GH, GW, 2)
        for r in range(GH - 1):
            for c in range(GW - 1):
                for tri in ([(r, c), (r, c + 1), (r + 1, c)],
                            [(r, c + 1), (r + 1, c + 1), (r + 1, c)]):
                    sTri = np.float32([G_src[t[0], t[1]] for t in tri])
                    dTri = np.float32([G_dst[t[0], t[1]] for t in tri])
                    M = cv2.getAffineTransform(sTri, dTri)
                    piece = cv2.warpAffine(crop, M, (cw, ch), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    mask = np.zeros_like(crop)
                    cv2.fillConvexPoly(mask, np.int32(dTri), 255)
                    np.copyto(warped, piece, where=mask > 0)
        rgba = np.zeros((ch, cw, 4), np.uint8)
        rgba[:, :, 0] = rgba[:, :, 1] = rgba[:, :, 2] = 255
        rgba[:, :, 3] = warped
        ax.imshow(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))
        ax.axis("off")
        ax.set_title(p["label"], color="white", fontsize=11)
        ax.set_facecolor("#171A21")
    plt.tight_layout()
    out = os.path.join(ROOT, "tools", "mls_preview.png")
    plt.savefig(out, dpi=110, facecolor="#171A21")
    print(f"saved {out}")


if __name__ == "__main__":
    crop, crop_box, src_wh = make_silhouette_asset()
    preview(crop, crop_box, src_wh)
