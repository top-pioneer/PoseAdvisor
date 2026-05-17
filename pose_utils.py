# pose_utils.py — 与训练一致的 25×3 关节表示与骨架可视化
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — 注册 3d projection

# 与 mediapipe_to_pose_vector.visualize_pose 中骨架定义一致（与现有 poses.npy 兼容）
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (4, 10), (10, 13), (13, 16), (16, 18),
    (4, 11), (11, 14), (14, 17), (17, 19),
    (0, 12), (12, 4), (4, 7), (7, 20),
    (0, 15), (15, 5), (5, 8), (8, 20),
    (22, 23), (22, 24),
]

NUM_JOINTS = 25
POSE_DIM = NUM_JOINTS * 3


def vector_to_joints(pose_vector):
    """(75,) 或 (B,75) -> (25,3) 或 (B,25,3) numpy"""
    v = np.asarray(pose_vector, dtype=np.float32)
    if v.ndim == 1:
        assert v.shape[0] == POSE_DIM, f"期望 {POSE_DIM} 维，得到 {v.shape[0]}"
        return v.reshape(NUM_JOINTS, 3)
    assert v.ndim == 2 and v.shape[1] == POSE_DIM
    return v.reshape(v.shape[0], NUM_JOINTS, 3)


def joints_to_vector(joints):
    """(25,3) 或 (B,25,3) -> (75,) 或 (B,75)"""
    j = np.asarray(joints, dtype=np.float32)
    if j.ndim == 2:
        assert j.shape == (NUM_JOINTS, 3)
        return j.reshape(-1)
    assert j.ndim == 3 and j.shape[1:] == (NUM_JOINTS, 3)
    return j.reshape(j.shape[0], -1)


def _draw_skeleton_3d(ax, joints, line_color="C0", point_color="C0", linewidth=2, alpha=0.85):
    j = np.asarray(joints, dtype=np.float64)
    for a, b in SKELETON_CONNECTIONS:
        if a < len(j) and b < len(j):
            seg = np.stack([j[a], j[b]], axis=0)
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=line_color, linewidth=linewidth, alpha=alpha)
    ax.scatter(j[:, 0], j[:, 1], j[:, 2], c=point_color, s=28, depthshade=True, alpha=0.9)


def plot_pose_3d(
    joints,
    title="3D pose",
    elev=20,
    azim=-60,
    figsize=(6, 6),
    equal_aspect=True,
):
    """单幅 3D 骨架图"""
    j = vector_to_joints(joints) if np.asarray(joints).size == POSE_DIM else np.asarray(joints)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    _draw_skeleton_3d(ax, j)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if equal_aspect:
        max_range = np.ptp(j, axis=0).max() / 2.0 + 1e-6
        ctr = j.mean(axis=0)
        ax.set_xlim(ctr[0] - max_range, ctr[0] + max_range)
        ax.set_ylim(ctr[1] - max_range, ctr[1] + max_range)
        ax.set_zlim(ctr[2] - max_range, ctr[2] + max_range)
    ax.view_init(elev=elev, azim=azim)
    return fig, ax


def plot_pose_panel(
    pose_vectors,
    titles=None,
    suptitle=None,
    save_path=None,
    ncols=None,
):
    """
    在同一图中绘制多个 75 维姿态（例如：GT、重构、多次采样）。
    pose_vectors: list of (75,) numpy 或 list of (25,3)
    """
    n = len(pose_vectors)
    if ncols is None:
        ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    colors = plt.cm.tab10(np.linspace(0, 0.9, n))

    for i, pv in enumerate(pose_vectors):
        j = vector_to_joints(pv) if np.asarray(pv).size == POSE_DIM else np.asarray(pv)
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        c = colors[i % len(colors)]
        _draw_skeleton_3d(ax, j, line_color=c, point_color=c)
        ttl = titles[i] if titles and i < len(titles) else f"Pose {i + 1}"
        ax.set_title(ttl)
        max_range = np.ptp(j, axis=0).max() / 2.0 + 1e-6
        ctr = j.mean(axis=0)
        ax.set_xlim(ctr[0] - max_range, ctr[0] + max_range)
        ax.set_ylim(ctr[1] - max_range, ctr[1] + max_range)
        ax.set_zlim(ctr[2] - max_range, ctr[2] + max_range)
        ax.view_init(elev=20, azim=-60)

    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_pose_2d_image_plane(joints, image_shape_hw, title="2D projection (x,y in image norm)"):
    """将 MediaPipe 归一化 x,y 投影到像素平面用于叠图参考。image_shape: (H, W)"""
    j = vector_to_joints(joints) if np.asarray(joints).size == POSE_DIM else np.asarray(joints)
    H, W = image_shape_hw
    fig, ax = plt.subplots(figsize=(6, 6))
    xs = j[:, 0] * W
    ys = j[:, 1] * H
    for a, b in SKELETON_CONNECTIONS:
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], "b-", linewidth=2, alpha=0.7)
    ax.scatter(xs, ys, c="red", s=40, zorder=5)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_title(title)
    return fig, ax
