import argparse
import os
import sys
import urllib.request

import cv2
import numpy as np
import torch
from PIL import Image

from CNN_model import SceneConditionedPoseVAE
from mediapipe_to_pose_vector import mediapipe_to_75d_vector
from pose_utils import POSE_DIM, plot_pose_panel, plot_pose_3d, vector_to_joints

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_POSE_LITE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def _ensure_pose_landmarker_model():
    override = os.environ.get("POSE_LANDMARKER_MODEL")
    if override and os.path.isfile(override):
        return override
    root = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.join(root, ".cache", "mediapipe")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "pose_landmarker_lite.task")
    min_bytes = 5_000_000
    if not os.path.isfile(path) or os.path.getsize(path) < min_bytes:
        print(f"正在下载姿态模型到: {path}", flush=True)
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(_POSE_LITE_URL, path)
                if os.path.isfile(path) and os.path.getsize(path) >= min_bytes:
                    break
            except (urllib.error.ContentTooShortError, OSError):
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                if attempt == 2:
                    raise
    return path


def _landmarks_to_simple_list(landmarks):
    class _LM:
        __slots__ = ("x", "y", "z")

        def __init__(self, x, y, z):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z) if z is not None else 0.0

    return [_LM(lm.x, lm.y, getattr(lm, "z", None)) for lm in landmarks]


def _imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def image_path_to_pil_rgb(path):
    img_bgr = _imread_unicode(path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def extract_pose_vector_from_image(image_path, static_image_mode=True):
    import mediapipe as mp

    if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
        from mediapipe.tasks.python.vision.core.image import Image as MPImage
        from mediapipe.tasks.python.vision.core.image import ImageFormat

        img_bgr = _imread_unicode(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        model_path = _ensure_pose_landmarker_model()
        landmarker = mp.tasks.vision.PoseLandmarker.create_from_model_path(model_path)
        try:
            mp_img = MPImage(ImageFormat.SRGB, image_rgb)
            result = landmarker.detect(mp_img)
        finally:
            landmarker.close()
        if not result.pose_landmarks:
            return None, None
        simple = _landmarks_to_simple_list(result.pose_landmarks[0])
        vec, joints25 = mediapipe_to_75d_vector(simple)
        return np.asarray(vec, dtype=np.float32), joints25

    if hasattr(mp, "solutions"):
        img_bgr = _imread_unicode(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=2,
            min_detection_confidence=0.5,
        ) as pose:
            results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return None, None
        vec, joints25 = mediapipe_to_75d_vector(results.pose_landmarks.landmark)
        return np.asarray(vec, dtype=np.float32), joints25

    raise RuntimeError("当前 mediapipe 安装既不包含 tasks 也不包含 solutions，无法提取姿态。")


def load_model(checkpoint_path, device, pose_dim=75, latent_dim=64):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args")
    if args is not None:
        pose_dim = getattr(args, "pose_dim", pose_dim)
        latent_dim = getattr(args, "latent_dim", latent_dim)
    model = SceneConditionedPoseVAE(pose_dim=pose_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, pose_dim, latent_dim


def random_demo_pose(batch_size=1, device="cpu", seed=0):
    torch.manual_seed(seed)
    return torch.rand(batch_size, POSE_DIM, device=device, dtype=torch.float32) * 0.6 + 0.2


def run_extract(args):
    vec, joints = extract_pose_vector_from_image(args.image)
    if vec is None:
        print("未检测到人体姿态，请换图或调低 min_detection_confidence。", file=sys.stderr)
        return 1
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    npy_path = os.path.join(args.out_dir, f"{base}_pose75.npy")
    np.save(npy_path, vec)
    print(f"已保存训练同款 75 维向量: {npy_path} shape={vec.shape}")
    fig = plot_pose_3d(vec, title="MediaPipe -> 25x3 (训练格式)")
    out_png = os.path.join(args.out_dir, f"{base}_skeleton_3d.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"3D 骨架图: {out_png}")
    return 0


def run_generate(args, device):
    model, pose_dim, _ = load_model(args.checkpoint, device)
    pil = image_path_to_pil_rgb(args.image)
    with torch.no_grad():
        samples = model.generate([pil], num_samples=args.num_samples)
    samples_np = samples[0].cpu().numpy()
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    np.save(os.path.join(args.out_dir, f"{base}_generated.npy"), samples_np)
    titles = [f"sample {i + 1}" for i in range(len(samples_np))]
    out_png = os.path.join(args.out_dir, f"{base}_generate_skeleton.png")
    plot_pose_panel(list(samples_np), titles=titles, suptitle="条件生成 (VAE decode)", save_path=out_png)
    print(f"已保存 {args.num_samples} 组姿态与可视化: {out_png}")
    return 0


def run_reconstruct(args, device):
    model, pose_dim, _ = load_model(args.checkpoint, device)
    pil = image_path_to_pil_rgb(args.image)
    if args.pose_npy:
        vec = np.load(args.pose_npy).astype(np.float32).reshape(-1)
    else:
        vec, _ = extract_pose_vector_from_image(args.image)
        if vec is None:
            print("未检测到姿态且未提供 --pose_npy。", file=sys.stderr)
            return 1
    if vec.shape[0] != pose_dim:
        print(f"姿态维度 {vec.shape[0]} 与模型 pose_dim={pose_dim} 不一致。", file=sys.stderr)
        return 1
    pose_gt = torch.from_numpy(vec).unsqueeze(0).to(device)
    with torch.no_grad():
        pred, mu, logvar = model([pil], pose_gt)
    pred_np = pred[0].cpu().numpy()
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    np.save(os.path.join(args.out_dir, f"{base}_recon.npy"), pred_np)
    err = np.linalg.norm(vector_to_joints(pred_np) - vector_to_joints(vec))
    print(f"重构 L2 (关节空间近似): {err:.6f}")
    out_png = os.path.join(args.out_dir, f"{base}_reconstruct_compare.png")
    plot_pose_panel(
        [vec, pred_np],
        titles=["GT (75d 输入)", "重构输出"],
        suptitle="场景条件 VAE 重构",
        save_path=out_png,
    )
    print(f"对比图: {out_png}")
    return 0


def run_demo_model(args, device):
    model, pose_dim, _ = load_model(args.checkpoint, device)
    pil = image_path_to_pil_rgb(args.image)
    pose_gt = random_demo_pose(1, device=device, seed=args.seed)
    with torch.no_grad():
        pred, _, _ = model([pil], pose_gt)
    gt = pose_gt[0].cpu().numpy()
    pr = pred[0].cpu().numpy()
    os.makedirs(args.out_dir, exist_ok=True)
    out_png = os.path.join(args.out_dir, "demo_random_pose_recon.png")
    plot_pose_panel([gt, pr], titles=["随机 GT(演示)", "重构"], save_path=out_png)
    print(f"演示对比图: {out_png}")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="场景条件姿态 VAE：提取 75d 关节、生成或重构并可视化")
    p.add_argument("--mode", type=str, choices=["extract", "generate", "reconstruct", "demo"], required=True)
    p.add_argument("--image", type=str, help="输入 RGB 场景图")
    p.add_argument("--checkpoint", type=str, help="训练保存的 .pth 文件")
    p.add_argument("--out_dir", type=str, default="infer_out", help="输出目录")
    p.add_argument("--num_samples", type=int, default=5, help="generate 模式采样数")
    p.add_argument("--pose_npy", type=str, default=None, help="reconstruct: 外部 75 维 poses.npy")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu")
    p.add_argument("--seed", type=int, default=0, help="demo 模式随机种子")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode in ("extract", "generate", "reconstruct", "demo") and not args.image:
        print("--image 必填", file=sys.stderr)
        return 2
    if args.mode in ("generate", "reconstruct", "demo") and not args.checkpoint:
        print("该模式需要 --checkpoint", file=sys.stderr)
        return 2

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device={device}")

    if args.mode == "extract":
        return run_extract(args)
    if args.mode == "generate":
        return run_generate(args, device)
    if args.mode == "reconstruct":
        return run_reconstruct(args, device)
    if args.mode == "demo":
        return run_demo_model(args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
