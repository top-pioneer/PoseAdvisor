# pose_body_mesh.py — 由 25×3 关节生成简易三维人体网格（OBJ），供 Blender / MeshLab 等打开
# 说明：数据为 MediaPipe 归一化关节，非 SMPL；此为骨架包络网格。真实肌肉表面需 SMPL/SMPL-X 与官方模型文件。
from __future__ import annotations

import argparse
import os

import numpy as np

from pose_utils import POSE_DIM, SKELETON_CONNECTIONS, vector_to_joints

_PHI = (1.0 + np.sqrt(5.0)) / 2.0
# 单位球内接二十面体（再按半径缩放）
_ICO_VERTS = np.array(
    [
        [-1, _PHI, 0],
        [1, _PHI, 0],
        [-1, -_PHI, 0],
        [1, -_PHI, 0],
        [0, -1, _PHI],
        [0, 1, _PHI],
        [0, -1, -_PHI],
        [0, 1, -_PHI],
        [_PHI, 0, -1],
        [_PHI, 0, 1],
        [-_PHI, 0, -1],
        [-_PHI, 0, 1],
    ],
    dtype=np.float64,
)
_ICO_VERTS /= np.linalg.norm(_ICO_VERTS, axis=1, keepdims=True)
_ICO_FACES = np.array(
    [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ],
    dtype=np.int64,
)


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    z = d / n
    if abs(z[2]) < 0.9:
        a = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z


def mesh_joint_sphere(center: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    v = _ICO_VERTS * radius + center.reshape(1, 3)
    f = _ICO_FACES.copy()
    return v, f


def mesh_bone_cylinder(p0: np.ndarray, p1: np.ndarray, radius: float, n_sides: int = 10) -> tuple[np.ndarray, np.ndarray]:
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-10:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    x, y, z = _orthonormal_basis(d)
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    ring0 = []
    ring1 = []
    for th in angles:
        off = radius * (np.cos(th) * x + np.sin(th) * y)
        ring0.append(p0 + off)
        ring1.append(p1 + off)
    verts = np.array(ring0 + ring1, dtype=np.float64)
    faces = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        bi, bj, ti, tj = i, j, i + n_sides, j + n_sides
        faces.append([bi, bj, tj])
        faces.append([bi, tj, ti])
    return verts, np.array(faces, dtype=np.int64)


def skeleton_mesh_from_joints(
    joints: np.ndarray,
    bone_radius_frac: float = 0.045,
    joint_radius_frac: float = 0.065,
    n_sides: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    joints: (25, 3)
    半径相对骨架包围盒对角线比例，适配 MediaPipe 归一化坐标尺度。
    """
    j = np.asarray(joints, dtype=np.float64).reshape(25, 3)
    diag = float(np.linalg.norm(np.ptp(j, axis=0)))
    diag = max(diag, 1e-4)
    r_bone = diag * bone_radius_frac
    r_joint = diag * joint_radius_frac

    all_v = []
    all_f = []
    offset = 0

    for idx in range(25):
        v, f = mesh_joint_sphere(j[idx], r_joint)
        all_v.append(v)
        all_f.append(f + offset)
        offset += len(v)

    seen_bones: set[tuple[int, int]] = set()
    for a, b in SKELETON_CONNECTIONS:
        if a >= len(j) or b >= len(j):
            continue
        key = (min(a, b), max(a, b))
        if key in seen_bones:
            continue
        seen_bones.add(key)
        v, f = mesh_bone_cylinder(j[a], j[b], r_bone, n_sides=n_sides)
        if len(v) == 0:
            continue
        all_v.append(v)
        all_f.append(f + offset)
        offset += len(v)

    if not all_v:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    vertices = np.vstack(all_v)
    faces = np.vstack(all_f)
    return vertices, faces


def write_obj(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# pose_body_mesh: skeleton envelope\n")
        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for i, j, k in faces:
            # OBJ 1-based
            f.write(f"f {i + 1} {j + 1} {k + 1}\n")


def load_pose_vector(path: str, index: int = 0) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 1:
        if arr.shape[0] != POSE_DIM:
            raise ValueError(f"期望一维长度 {POSE_DIM}，得到 {arr.shape[0]}")
        return arr.astype(np.float32)
    if arr.ndim == 2:
        if arr.shape[1] != POSE_DIM:
            raise ValueError(f"期望末维 {POSE_DIM}，得到 {arr.shape[1]}")
        if index < 0 or index >= arr.shape[0]:
            raise IndexError(f"index={index} 超出 [0, {arr.shape[0]})")
        return arr[index].astype(np.float32)
    raise ValueError(f"不支持的数组形状: {arr.shape}")


def export_pose_to_obj(
    pose_vector: np.ndarray,
    out_obj: str,
    bone_radius_frac: float = 0.045,
    joint_radius_frac: float = 0.065,
    n_sides: int = 10,
) -> None:
    joints = vector_to_joints(pose_vector)
    v, f = skeleton_mesh_from_joints(
        joints,
        bone_radius_frac=bone_radius_frac,
        joint_radius_frac=joint_radius_frac,
        n_sides=n_sides,
    )
    write_obj(out_obj, v, f)


def main():
    p = argparse.ArgumentParser(description="将 75 维姿态导出为三维骨架网格 OBJ")
    p.add_argument("--npy", type=str, required=True, help="poses.npy 单行、或 (N,75) 如 1_generated.npy")
    p.add_argument("--index", type=int, default=0, help="批量数组时选第几条")
    p.add_argument("-o", "--out", type=str, default="body_mesh.obj", help="输出 .obj 路径")
    p.add_argument("--bone_frac", type=float, default=0.045, help="圆柱半径相对骨架对角线")
    p.add_argument("--joint_frac", type=float, default=0.065, help="关节球半径相对骨架对角线")
    p.add_argument("--sides", type=int, default=10, help="圆柱圆周分段数")
    args = p.parse_args()

    vec = load_pose_vector(args.npy, args.index)
    export_pose_to_obj(
        vec,
        args.out,
        bone_radius_frac=args.bone_frac,
        joint_radius_frac=args.joint_frac,
        n_sides=args.sides,
    )
    print(f"已写入: {args.out}  (顶点 {vector_to_joints(vec).shape[0]} 关节)")


if __name__ == "__main__":
    main()
