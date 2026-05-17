# mediapipe_to_pose_vector.py
import cv2
import mediapipe as mp
import numpy as np
import torch
import os

def mediapipe_to_75d_vector(landmarks):
    """
    将MediaPipe的33个姿态关键点转换为25个关键点的75维向量

    MediaPipe有33个关键点，我们选择其中25个主要的：
    索引对应关系（MediaPipe索引 -> 自定义索引）：
    0: 鼻子 -> 22
    1: 左眼内侧 -> 23?
    2: 左眼 -> 23
    3: 左眼外侧 -> 23?
    4: 右眼内侧 -> 24
    5: 右眼 -> 24
    6: 右眼外侧 -> 24?
    7: 左耳 -> 不使用
    8: 右耳 -> 不使用
    9: 嘴左 -> 不使用
    10: 嘴右 -> 不使用
    11: 左肩 -> 10
    12: 右肩 -> 11
    13: 左肘 -> 13
    14: 右肘 -> 14
    15: 左腕 -> 16
    16: 右腕 -> 17
    17: 左手掌 -> 18?
    18: 右手掌 -> 19?
    19: 左小指 -> 不使用
    20: 右小指 -> 不使用
    21: 左食指 -> 不使用
    22: 右食指 -> 不使用
    23: 左拇指 -> 不使用
    24: 右拇指 -> 不使用
    25: 左髋 -> 12
    26: 右髋 -> 15
    27: 左膝 -> 4
    28: 右膝 -> 5
    29: 左踝 -> 7
    30: 右踝 -> 8
    31: 左脚跟 -> 20?
    32: 左脚趾 -> 20?
    """

    # 定义我们需要的25个关键点的MediaPipe索引映射
    # 格式: [mediapipe_index, 我们的索引(0-24)]
    selected_indices = [
        # 躯干和头部 (0-5)
        [0, 22],  # 鼻子
        [2, 23],  # 左眼 (使用左眼中心)
        [5, 24],  # 右眼 (使用右眼中心)
        [11, 10],  # 左肩
        [12, 11],  # 右肩
        [25, 12],  # 左髋
        [26, 15],  # 右髋

        # 脊柱 (我们通过平均来估计)
        # 我们需要估计脊柱点：骨盆、脊柱下、脊柱中、脊柱上、颈部

        # 手臂 (6-11)
        [13, 13],  # 左肘
        [14, 14],  # 右肘
        [15, 16],  # 左腕
        [16, 17],  # 右腕
        [17, 18],  # 左手 (使用左手掌)
        [18, 19],  # 右手 (使用右手掌)

        # 腿 (12-17)
        [27, 4],  # 左膝
        [28, 5],  # 右膝
        [29, 7],  # 左踝
        [30, 8],  # 右踝
        [31, 20],  # 左脚 (使用左脚跟)
        [32, 20],  # 左脚也可以使用脚趾，这里我们合并
    ]

    # 创建25个关键点的数组，初始化为0
    selected_joints = np.zeros((25, 3), dtype=np.float32)

    # 填充选中的关键点
    for mp_idx, our_idx in selected_indices:
        if mp_idx < len(landmarks):
            lm = landmarks[mp_idx]
            selected_joints[our_idx] = [lm.x, lm.y, lm.z]

    # 估计脊柱点
    # 使用左肩、右肩、左髋、右髋的平均来估计
    left_shoulder = selected_joints[10]  # 我们的索引10对应左肩
    right_shoulder = selected_joints[11]  # 我们的索引11对应右肩
    left_hip = selected_joints[12]  # 我们的索引12对应左髋
    right_hip = selected_joints[15]  # 我们的索引15对应右髋

    # 检查是否所有需要的点都存在（非零）
    if np.any(left_shoulder) and np.any(right_shoulder) and np.any(left_hip) and np.any(right_hip):
        # 骨盆中心 (索引0)
        selected_joints[0] = (left_hip + right_hip) / 2

        # 脊柱下 (索引1)
        selected_joints[1] = (left_hip + right_hip + left_shoulder + right_shoulder) / 4 * 0.3 + selected_joints[
            0] * 0.7

        # 脊柱中 (索引2)
        selected_joints[2] = (left_shoulder + right_shoulder + left_hip + right_hip) / 4

        # 脊柱上 (索引3)
        selected_joints[3] = (left_shoulder + right_shoulder) / 2

        # 颈部 (索引4)
        nose = selected_joints[22]
        if np.any(nose):
            selected_joints[4] = (left_shoulder + right_shoulder + nose) / 3
        else:
            selected_joints[4] = (left_shoulder + right_shoulder) / 2

        # 头部 (索引5)
        left_eye = selected_joints[23]
        right_eye = selected_joints[24]
        if np.any(left_eye) and np.any(right_eye):
            selected_joints[5] = (left_eye + right_eye) / 2 + np.array([0, 0.1, 0])  # 在眼睛上方一点
        elif np.any(nose):
            selected_joints[5] = nose + np.array([0, 0.15, 0])

    # 展平为75维向量
    pose_vector = selected_joints.flatten()

    return pose_vector, selected_joints


def save_as_pt(pose_vector, output_path):
    """
    保存为PyTorch的.pt格式
    """
    # 转换为torch tensor
    pose_tensor = torch.tensor(pose_vector, dtype=torch.float32)

    # 保存
    torch.save(pose_tensor, output_path)
    print(f"已保存为.pt格式: {output_path}")
    print(f"张量形状: {pose_tensor.shape}")

    return pose_tensor


def process_single_image(image_path, output_dir, pose):
    """
    处理单张图片，输出75维向量并保存为.pt格式
    """

    # 读取图片（支持中文路径）
    def imread_unicode(path):
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img

    image = imread_unicode(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return None

    # 转换为RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 处理
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # 转换为75维向量
        pose_vector, joints_25 = mediapipe_to_75d_vector(results.pose_landmarks.landmark)

        # 打印信息
        print(f"成功从图片提取姿态: {image_path}")
        print(f"75维向量前10个值: {pose_vector[:10]}")

        # 确定输出路径
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        os.makedirs(output_dir, exist_ok=True)

        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        pt_path = os.path.join(output_dir, f"{base_name}_pose.pt")
        npy_path = os.path.join(output_dir, f"{base_name}_pose.npy")
        joints_path = os.path.join(output_dir, f"{base_name}_joints.npy")

        # 保存为.pt格式
        save_as_pt(pose_vector, pt_path)

        # 同时也保存为npy格式（方便查看）
        np.save(npy_path, pose_vector)
        np.save(joints_path, joints_25)

        print(f"同时保存了npy格式: {npy_path}")
        print(f"25个关节点保存到: {joints_path}")

        # 可视化
        visualize_pose(image, results.pose_landmarks, joints_25, output_dir, base_name)

        return pose_vector, joints_25
    else:
        print(f"未检测到姿态: {image_path}")
        return None


def visualize_pose(original_image, landmarks, joints_25, output_dir, base_name):
    """
    可视化检测结果
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 创建可视化图
    fig = plt.figure(figsize=(20, 5))

    # 1. 原始图像
    ax1 = fig.add_subplot(141)
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    ax1.imshow(image_rgb)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # 2. MediaPipe检测结果
    ax2 = fig.add_subplot(142)
    annotated_image = original_image.copy()
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_drawing.draw_landmarks(
        annotated_image,
        landmarks,
        mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    ax2.imshow(annotated_image_rgb)
    ax2.set_title('MediaPipe Detection')
    ax2.axis('off')

    # 3. 25个关节点在2D上的投影
    ax3 = fig.add_subplot(143)
    ax3.scatter(joints_25[:, 0] * image_rgb.shape[1],
                joints_25[:, 1] * image_rgb.shape[0],
                c='red', s=50)
    ax3.set_xlim(0, image_rgb.shape[1])
    ax3.set_ylim(image_rgb.shape[0], 0)
    ax3.set_title('25 Joints (2D Projection)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')

    # 4. 3D姿态
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.scatter(joints_25[:, 0], joints_25[:, 1], joints_25[:, 2],
                c=joints_25[:, 1], cmap='viridis', s=50)

    # 绘制连接线（定义骨架连接）
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),  # 脊柱
        (4, 10), (10, 13), (13, 16), (16, 18),  # 左臂
        (4, 11), (11, 14), (14, 17), (17, 19),  # 右臂
        (0, 12), (12, 4), (4, 7), (7, 20),  # 左腿
        (0, 15), (15, 5), (5, 8), (8, 20),  # 右腿
        (22, 23), (22, 24)  # 眼睛
    ]

    for conn in connections:
        if conn[0] < len(joints_25) and conn[1] < len(joints_25):
            points = np.array([joints_25[conn[0]], joints_25[conn[1]]])
            ax4.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', linewidth=2, alpha=0.5)

    ax4.set_title('3D Pose')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')

    plt.tight_layout()

    # 保存可视化结果
    vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存: {vis_path}")


def batch_process(image_dir, output_dir=None):
    """
    批量处理目录中的所有图片
    """
    if output_dir is None:
        output_dir = os.path.join(image_dir, 'pose_outputs')
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图片
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for f in os.listdir(image_dir):
        if any(f.lower().endswith(ext) for ext in image_extensions):
            image_files.append(f)

    image_files.sort()
    print(f"找到 {len(image_files)} 张图片")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    )

    all_poses = []
    successful = 0

    for i, img_file in enumerate(image_files):
        print(f"\n处理 [{i + 1}/{len(image_files)}]: {img_file}")
        img_path = os.path.join(image_dir, img_file)

        result = process_single_image(img_path, output_dir, pose)

        if result is not None:
            pose_vector, _ = result
            all_poses.append(pose_vector)
            successful += 1

    # 保存所有姿态到一个文件
    if all_poses:
        all_poses = np.array(all_poses, dtype=np.float32)

        # 保存为npy
        npy_path = os.path.join(output_dir, 'poses.npy')
        np.save(npy_path, all_poses)
        print(f"\n所有姿态已保存到: {npy_path}")

        # 保存为pt
        pt_path = os.path.join(output_dir, 'all_poses.pt')
        torch.save(torch.from_numpy(all_poses), pt_path)
        print(f"所有姿态已保存到: {pt_path}")

        print(f"\n处理完成! 成功: {successful}/{len(image_files)}")
        print(f"最终姿态数组形状: {all_poses.shape}")

        return all_poses
    else:
        print("没有成功提取到任何姿态")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description='使用MediaPipe提取人体姿态并转换为75维向量')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图片路径或目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（可选）')
    parser.add_argument('--batch', action='store_true',
                        help='批量处理模式（输入为目录时使用）')

    args = parser.parse_args()

    if args.batch or os.path.isdir(args.input):
        # 批量处理模式
        batch_process(args.input, args.output_dir)
    else:
        # 单张图片处理
        process_single_image(args.input, args.output_dir)


if __name__ == "__main__":
    # # 使用示例
    # # image_path = r"E:\大创\photo_structure\images\image_10.jpg"
    # # output_dir = r"E:\大创\photo_structure\pose_outputs"
    # # process_single_image(image_path, output_dir)
    #
    # # 或者使用命令行参数
    # # python mediapipe_to_pose_vector.py --input "你的图片路径.jpg" --output_dir "输出目录"
    #
    # # 批量处理
    # # python mediapipe_to_pose_vector.py --input "图片目录" --batch
    #
    # # 直接在这里设置参数运行
    # image_path = r"E:\大创\photo_structure\images\image_10.jpg"
    # output_dir = r"E:\大创\photo_structure\pose_outputs"
    #
    # print("开始处理单张图片...")
    # result = process_single_image(image_path, output_dir)
    #
    # if result is not None:
    #     pose_vector, joints_25 = result
    #     print("\n处理成功！")
    #     print(f"75维姿态向量形状: {pose_vector.shape}")
    #     print(f"25个关节点形状: {joints_25.shape}")
    # else:
    #     print("处理失败")
    pass