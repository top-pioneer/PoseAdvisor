from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch

import smplx

from pose_utils import POSE_DIM, vector_to_joints

USER_TO_SMPLX = [
    (0, 0, 1.0),
    (1, 3, 1.0),
    (2, 6, 1.0),
    (3, 9, 1.0),
    (4, 12, 1.0),
    (5, 15, 1.0),
    (7, 7, 1.0),
    (8, 8, 1.0),
    (10, 16, 1.0),
    (11, 17, 1.0),
    (12, 1, 1.0),
    (13, 18, 1.0),
    (14, 19, 1.0),
    (15, 2, 1.0),
    (16, 20, 1.0),
    (17, 21, 1.0),
    (18, 20, 0.35),
    (19, 21, 0.35),
    (20, 10, 1.0),
    (22, 15, 0.2),
    (23, 23, 0.5),
    (24, 24, 0.5),
]


def _default_model_root() -> str:
    return os.path.dirname(os.path.abspath(__file__)) # smplx模型路径


def mediapipe_points_to_smpl_upright(P: np.ndarray, flip_y: bool, flip_z: bool, swap_yz: bool) -> np.ndarray:
    Q = np.asarray(P, dtype=np.float64).copy()
    if swap_yz:
        Q = Q[:, [0, 2, 1]]
    if flip_y:
        Q[:, 1] *= -1.0
    if flip_z:
        Q[:, 2] *= -1.0
    return Q


def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    h = Pc.T @ Qc
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    return r


def prepare_target_points(
    joints25: np.ndarray,
    flip_y: bool = True,
    flip_z: bool = False,
    swap_yz: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    j = np.asarray(joints25, dtype=np.float64).reshape(25, 3)
    u_idx = []
    s_idx = []
    w = []
    for ui, si, wi in USER_TO_SMPLX:
        if np.linalg.norm(j[ui]) < 1e-8:
            continue
        u_idx.append(ui)
        s_idx.append(si)
        w.append(wi)
    if len(u_idx) < 4:
        raise ValueError("有效关节过少，无法拟合（请确认姿态向量非零）。")
    u_idx = np.array(u_idx, dtype=np.int64)
    s_idx = np.array(s_idx, dtype=np.int64)
    w = np.array(w, dtype=np.float64)
    P = j[u_idx].copy()
    pelvis_rows = np.where(u_idx == 0)[0]
    if len(pelvis_rows):
        P -= P[pelvis_rows[0] : pelvis_rows[0] + 1]
    else:
        P -= P.mean(axis=0, keepdims=True)
    hip_l = j[12]
    hip_r = j[15]
    hip_w = float(np.linalg.norm(hip_l - hip_r))
    if hip_w < 1e-6:
        hip_w = 0.35
    scale = 0.34 / hip_w
    P *= scale
    P = mediapipe_points_to_smpl_upright(P, flip_y=flip_y, flip_z=flip_z, swap_yz=swap_yz)
    return P, u_idx, s_idx, w


def fit_smplx(
    model_path: str,
    target_xyz: np.ndarray,
    smplx_joint_indices: np.ndarray,
    joint_weights: np.ndarray,
    gender: str = "neutral",
    device: str | None = None,
    steps: int = 400,
    lr: float = 0.02,
    optimize_betas: bool = False,
    stage1_ratio: float = 0.35,
    pose_reg: float = 0.008,
    aa_limit_reg: float = 0.02,
) -> tuple[dict, torch.Tensor, torch.Tensor, float]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = smplx.create(
        model_path,
        model_type="smplx",
        gender=gender,
        use_pca=False,
        batch_size=1,
        ext="npz",
    ).to(dev)

    tgt = torch.tensor(target_xyz, dtype=torch.float32, device=dev)
    w = torch.tensor(joint_weights, dtype=torch.float32, device=dev)
    sm_idx = torch.tensor(smplx_joint_indices, dtype=torch.long, device=dev)

    global_orient = torch.zeros(1, 3, device=dev, requires_grad=True)
    body_pose = torch.zeros(1, 63, device=dev, requires_grad=True)
    transl = torch.zeros(1, 3, device=dev, requires_grad=True)
    model.betas.requires_grad_(bool(optimize_betas))

    z3 = torch.zeros(1, 3, device=dev)
    z63 = torch.zeros(1, 63, device=dev)
    lh = torch.zeros(1, 45, device=dev)
    rh = torch.zeros(1, 45, device=dev)

    with torch.no_grad():
        out0 = model(
            global_orient=z3,
            body_pose=z63,
            transl=z3,
            jaw_pose=z3,
            leye_pose=z3,
            reye_pose=z3,
            left_hand_pose=lh,
            right_hand_pose=rh,
            return_verts=False,
        )
        j0 = out0.joints[0, smplx_joint_indices, :].detach().cpu().numpy()
        tgt_np = target_xyz.astype(np.float64)
        r_align = kabsch_rotation(j0, tgt_np)
        rvec, _ = cv2.Rodrigues(r_align.astype(np.float64))
        global_orient.data = torch.tensor(rvec.flatten(), dtype=torch.float32, device=dev).unsqueeze(0)

        out1 = model(
            global_orient=global_orient,
            body_pose=z63,
            transl=z3,
            jaw_pose=z3,
            leye_pose=z3,
            reye_pose=z3,
            left_hand_pose=lh,
            right_hand_pose=rh,
            return_verts=False,
        )
        j1 = out1.joints[0, smplx_joint_indices, :]
        pelvis_rows = np.where(smplx_joint_indices == 0)[0]
        if len(pelvis_rows) > 0:
            transl.data = (tgt[pelvis_rows[0]] - j1[pelvis_rows[0]]).unsqueeze(0)
        else:
            transl.data = (tgt[0] - j1[0]).unsqueeze(0)

    n_stage1 = int(steps * stage1_ratio)
    if steps > 1:
        n_stage1 = min(max(1, n_stage1), steps - 1)
    else:
        n_stage1 = 0

    def make_opt(orient_body: bool):
        ps = [global_orient, transl]
        if orient_body:
            ps.append(body_pose)
        if optimize_betas:
            ps.append(model.betas)
        return torch.optim.Adam(ps, lr=lr)

    opt = make_opt(orient_body=False)

    fit_mse = 0.0
    pose_penalty = 0.0
    for it in range(steps):
        if it == n_stage1:
            opt = make_opt(orient_body=True)
        freeze_body = it < n_stage1
        if freeze_body:
            body_pose.data.zero_()
            body_pose.requires_grad_(False)
        else:
            body_pose.requires_grad_(True)

        opt.zero_grad()
        bp = z63 if freeze_body else body_pose
        out = model(
            global_orient=global_orient,
            body_pose=bp,
            transl=transl,
            jaw_pose=z3,
            leye_pose=z3,
            reye_pose=z3,
            left_hand_pose=lh,
            right_hand_pose=rh,
            return_verts=False,
        )
        pred = out.joints[0, sm_idx, :]
        diff = pred - tgt
        fit_loss = (w.unsqueeze(1) * (diff ** 2)).sum() / (w.sum() * 3 + 1e-8)
        loss = fit_loss
        if not freeze_body:
            bp_ = body_pose.reshape(-1, 3)
            aa_norm = torch.linalg.norm(bp_, dim=-1)
            over = torch.relu(aa_norm - 1.4)
            pose_pen = (
                pose_reg * (body_pose ** 2).mean()
                + aa_limit_reg * (over ** 2).mean()
            )
            loss = (
                loss
                + pose_pen
                + 0.002 * (global_orient ** 2).mean()
            )
            pose_penalty = float(pose_pen.detach().cpu().item())
        else:
            loss = loss + 0.002 * (global_orient ** 2).mean()
            pose_penalty = 0.0
        fit_mse = float(fit_loss.detach().cpu().item())
        loss.backward()
        opt.step()

    with torch.no_grad():
        out_f = model(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            jaw_pose=z3,
            leye_pose=z3,
            reye_pose=z3,
            left_hand_pose=lh,
            right_hand_pose=rh,
            return_verts=True,
        )

    state = {
        "global_orient": global_orient.detach().cpu().numpy(),
        "body_pose": body_pose.detach().cpu().numpy(),
        "transl": transl.detach().cpu().numpy(),
        "betas": model.betas.detach().cpu().numpy(),
    }
    score = fit_mse + pose_penalty
    return state, out_f.vertices[0].detach().cpu(), model.faces, score


def write_obj(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# SMPL-X fitted mesh\n")
        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for tri in faces:
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            f.write(f"f {i + 1} {j + 1} {k + 1}\n")


def load_pose_npy(path: str, index: int) -> np.ndarray:
    a = np.load(path)
    if a.ndim == 1:
        if a.shape[0] != POSE_DIM:
            raise ValueError(f"需要长度 {POSE_DIM}，得到 {a.shape[0]}")
        return a
    if a.shape[1] != POSE_DIM:
        raise ValueError(f"需要末维 {POSE_DIM}")
    return a[index]


def main():
    p = argparse.ArgumentParser(description="75维姿态 -> SMPL-X 拟合 + 导出 OBJ")
    p.add_argument("--model_root", type=str, default="", help="smplx 模型根目录")
    p.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"])
    p.add_argument("--npy", type=str, required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("-o", "--out", type=str, default="", help="输出 OBJ 路径")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--betas", action="store_true", help="优化体型参数")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no_flip_y", action="store_true", help="关闭 Y 轴翻转")
    p.add_argument("--flip_z", action="store_true", help="翻转 Z 轴")
    p.add_argument("--swap_yz", action="store_true", help="交换 Y/Z 轴")
    p.add_argument("--stage1_ratio", type=float, default=0.35)
    p.add_argument("--pose_reg", type=float, default=0.008)
    p.add_argument("--auto_axes", action="store_true", help="自动尝试多组坐标轴组合")
    args = p.parse_args()

    root = args.model_root.strip() or _default_model_root()
    smplx_dir = os.path.join(root, "smplx")
    if not os.path.isdir(smplx_dir):
        print(f"未找到 SMPL-X 目录: {smplx_dir}", file=sys.stderr)
        print("请用 --model_root 指定", file=sys.stderr)
        return 2

    pose = load_pose_npy(args.npy, args.index)
    j25 = vector_to_joints(pose)
    out_obj = args.out.strip()
    if not out_obj:
        out_obj = os.path.join(os.path.dirname(os.path.abspath(args.npy)), f"smplx_fit_idx{args.index}.obj")

    print(f"模型目录: {root}")
    candidates = []
    if args.auto_axes:
        candidates = [
            ("auto_default", True, False, False),
            ("auto_flipz", True, True, False),
            ("auto_swapyz", True, False, True),
            ("auto_noflipy_flipz", False, True, False),
        ]
    else:
        candidates = [("single", not args.no_flip_y, args.flip_z, args.swap_yz)]

    best = None
    best_pack = None
    best_axes = None
    for name, flip_y, flip_z, swap_yz in candidates:
        try:
            P, _, s_idx, w = prepare_target_points(j25, flip_y=flip_y, flip_z=flip_z, swap_yz=swap_yz)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 1
        print(f"[{name}] 有效约束关节数: {len(P)}，优化步数: {args.steps}")
        state, verts, faces, score = fit_smplx(
            root, P, s_idx, w,
            gender=args.gender,
            device=args.device,
            steps=args.steps,
            lr=args.lr,
            optimize_betas=args.betas,
            stage1_ratio=args.stage1_ratio,
            pose_reg=args.pose_reg,
        )
        if best is None or score < best:
            best = score
            best_pack = (state, verts, faces)
            best_axes = (flip_y, flip_z, swap_yz)

    state, verts, faces = best_pack
    v_np = verts.numpy()
    f_np = np.asarray(faces, dtype=np.int64)
    write_obj(out_obj, v_np, f_np)
    npz_path = out_obj.replace(".obj", "_params.npz")
    state["axes"] = np.array(best_axes, dtype=np.int64)
    np.savez(npz_path, **{k: v for k, v in state.items()})
    print(f"best_score={best:.6f}, axes(flip_y,flip_z,swap_yz)={best_axes}")
    print(f"已写入网格: {out_obj}")
    print(f"已写入参数: {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
