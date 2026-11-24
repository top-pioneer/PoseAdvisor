# -*- coding: utf-8 -*-
"""离线预览：与 App 端 BodyOutlineRenderer 相同算法的人形轮廓渲染验证"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

HEAD_RX, HEAD_RY, NECK_W = 0.34, 0.44, 0.13
UARM_W1, UARM_W2, FARM_W2, HAND_LEN = 0.17, 0.12, 0.08, 0.16
TORSO_PAD, WAIST_RATIO, HIP_PAD = 0.05, 0.80, 0.09
THIGH_W1, CALF_W1, CALF_W2, FOOT_LEN = 0.20, 0.13, 0.075, 0.20

PELVIS, SPINE_MID, CHEST, NECK, HEAD = 0, 2, 3, 4, 5
ANKLE_L, ANKLE_R = 7, 8
SHOULDER_L, SHOULDER_R, HIP_L, HIP_R = 10, 11, 12, 15
ELBOW_L, ELBOW_R, WRIST_L, WRIST_R = 13, 14, 16, 17


def norm(v):
    l = np.linalg.norm(v)
    return v / l if l > 1e-9 else np.array([0.0, -1.0])


def perp(d):
    return np.array([-d[1], d[0]])


def outline(j, t):
    pts = []
    body_up = norm(j[NECK] - j[PELVIS])
    side_l = perp(body_up)
    head, neck, pelvis = j[HEAD], j[NECK], j[PELVIS]
    half_shoulder = np.linalg.norm(j[SHOULDER_L] - j[SHOULDER_R]) / 2
    half_hip = np.linalg.norm(j[HIP_L] - j[HIP_R]) / 2

    def head_pt(u, l):
        return head + body_up * u + side_l * l

    neck_r = neck - side_l * NECK_W * t
    neck_l = neck + side_l * NECK_W * t

    def arm_chain(s_i, e_i, w_i, sign):
        s, e, w = j[s_i], j[e_i], j[w_i]
        u1 = norm(e - s); n1 = perp(u1)
        if np.dot(n1, side_l * sign) < 0:
            n1 = -n1
        u2 = norm(w - e); n2 = perp(u2)
        if np.dot(n2, n1) < 0:
            n2 = -n2
        tip = w + u2 * HAND_LEN * t
        return [s + n1 * UARM_W1 * t, e + n1 * UARM_W2 * t, w + n2 * FARM_W2 * t, tip,
                w - n2 * FARM_W2 * t, e - n1 * UARM_W2 * t, s - n1 * UARM_W1 * t * 0.75]

    def leg_chain(h_i, a_i, sign):
        h, a = j[h_i], j[a_i]
        k = (h + a) / 2
        u3 = norm(k - h); n3 = perp(u3)
        if np.dot(n3, side_l * sign) < 0:
            n3 = -n3
        u4 = norm(a - k); n4 = perp(u4)
        if np.dot(n4, n3) < 0:
            n4 = -n4
        toe_dir = norm(u4 + n3 * 0.5)
        toe = a + toe_dir * FOOT_LEN * t
        return [h + n3 * THIGH_W1 * t, k + n3 * CALF_W1 * t, a + n4 * CALF_W2 * t, toe,
                a - n4 * CALF_W2 * t, k - n3 * CALF_W1 * t]

    def torso_side(sign):
        s_l = side_l * sign
        chest = j[CHEST] + s_l * (half_shoulder + TORSO_PAD * t)
        waist = j[SPINE_MID] + s_l * (half_shoulder * WAIST_RATIO)
        return [chest, waist]

    arm_l = arm_chain(SHOULDER_L, ELBOW_L, WRIST_L, +1)
    arm_r = arm_chain(SHOULDER_R, ELBOW_R, WRIST_R, -1)
    leg_l = leg_chain(HIP_L, ANKLE_L, +1)
    leg_r = leg_chain(HIP_R, ANKLE_R, -1)
    torso_l, torso_r = torso_side(+1), torso_side(-1)
    crotch_l = pelvis + side_l * half_hip * 0.25 + np.array([0, 0.06 * t])
    crotch_r = pelvis - side_l * half_hip * 0.25 + np.array([0, 0.06 * t])

    pts.append(neck_r)
    pts += arm_r[0:4]
    pts += arm_r[4:7]
    pts += torso_r
    pts.append(j[HIP_R] - side_l * HIP_PAD * t)
    pts += leg_r
    pts += [crotch_r, crotch_l]
    pts += leg_l[::-1]
    pts.append(j[HIP_L] + side_l * HIP_PAD * t)
    pts += torso_l[::-1]
    pts += arm_l[6:7]
    pts += arm_l[4:6][::-1]
    pts += arm_l[3:4]
    pts += arm_l[0:3][::-1]
    pts.append(neck_l)
    pts += [head_pt(0, HEAD_RX * t),
            head_pt(HEAD_RY * t * 0.72, HEAD_RX * t * 0.72),
            head_pt(HEAD_RY * t, 0),
            head_pt(HEAD_RY * t * 0.72, -HEAD_RX * t * 0.72),
            head_pt(0, -HEAD_RX * t)]
    return np.array(pts)


def smooth_path(pts):
    n = len(pts)
    def mid(a, b):
        return (a + b) / 2
    verts = [mid(pts[0], pts[1])]
    codes = [Path.MOVETO]
    for i in range(1, n):
        m = mid(pts[i], pts[(i + 1) % n])
        verts += [pts[i], m]
        codes += [Path.CURVE3, Path.CURVE3]
    m_last = mid(pts[-1], pts[0])
    verts += [pts[0], m_last]
    codes += [Path.CURVE3, Path.CURVE3]
    verts.append(verts[0]); codes.append(Path.CLOSEPOLY)
    return Path(verts, codes)


lib = json.load(open("android/app/src/main/assets/pose_library.json", encoding="utf-8"))
poses = lib["poses"]
# 每类取 2 个 + 站姿全部首页
samples = [p for p in poses if p["id"] in (
    "standing_01", "standing_02", "sitting_01", "gesture_01", "dynamic_01", "dynamic_03")]

fig, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor="#171A21")
for ax, p in zip(axes.flat, samples):
    j = np.array(p["joints"])[:, :2]
    t = np.linalg.norm(j[NECK] - j[PELVIS])
    pts = outline(j, t)
    path = smooth_path(pts)
    ax.add_patch(PathPatch(path, facecolor="#5B8CFF", edgecolor="none", alpha=0.9))
    allpts = np.vstack(pts)
    cx, cy = allpts.mean(axis=0)
    span = max(np.ptp(allpts[:, 0]), np.ptp(allpts[:, 1])) * 0.65
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy + span, cy - span)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(p["label"], color="white", fontsize=11)
    ax.set_facecolor("#171A21")
plt.tight_layout()
plt.savefig("tools/outline_preview.png", dpi=110, facecolor="#171A21")
print("saved tools/outline_preview.png")
