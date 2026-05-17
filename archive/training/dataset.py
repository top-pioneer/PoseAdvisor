import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np


class PoseImageDataset(Dataset):
    def __init__(self, data_root="data", split='train', transform=None,
                 preload_features=False, clip_processor=None, random_seed=42,
                 feature_path=None):  # 添加feature_path参数
        """
        Args:
            data_root: 数据根目录
            split: 'train', 'val', 或 'test'
            transform: 图像变换（可选）
            preload_features: 是否预加载CLIP特征（加速训练）
            clip_processor: CLIP处理器（如果preload_features=False，需要传入）
            feature_path: 预计算特征的路径（如果preload_features=True，需要传入）
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.preload_features = preload_features
        self.clip_processor = clip_processor
        self.random_seed = random_seed
        self.features = None

        # 加载图像路径列表
        self.image_dir = os.path.join(data_root, 'images')
        self.image_files = sorted([f for f in os.listdir(self.image_dir)
                                   if f.endswith(('.jpg', '.png', '.jpeg'))])

        total = len(self.image_files)
        rng = np.random.RandomState(random_seed)
        indices = np.arange(total)
        rng.shuffle(indices)

        n_train = int(0.8 * total)
        n_val = int(0.1 * total)

        if split == 'train':
            selected_indices = indices[:n_train]
        elif split == 'val':
            selected_indices = indices[n_train:n_train + n_val]
        else:  # test
            selected_indices = indices[n_train + n_val:]

        # 根据选中的索引筛选文件和姿态
        self.image_files = [self.image_files[i] for i in selected_indices]
        all_poses = np.load(os.path.join(data_root, 'poses.npy'))
        self.poses = all_poses[selected_indices]

        # 如果预加载特征，加载特征文件
        if self.preload_features:
            self.load_precomputed_features(feature_path)

    def load_precomputed_features(self, feature_path):
        """加载预计算的CLIP特征"""
        if feature_path is None:
            # 默认特征路径
            feature_path = os.path.join(self.data_root, 'scene_features.pt')

        try:
            # 加载所有特征
            all_features = torch.load(feature_path)
            print(f"加载特征文件: {feature_path}")
            print(f"特征类型: {type(all_features)}")

            # 根据当前split选择对应的特征
            if isinstance(all_features, dict):
                # 如果特征是字典格式，按split选择
                if self.split in all_features:
                    self.features = all_features[self.split]
                else:
                    # 如果不是按split存储的，使用索引选择
                    total_features = len(self.image_files)
                    if self.split == 'train':
                        self.features = all_features[:int(0.8 * total_features)]
                    elif self.split == 'val':
                        self.features = all_features[int(0.8 * total_features):int(0.9 * total_features)]
                    else:
                        self.features = all_features[int(0.9 * total_features):]
            else:
                # 如果特征是列表或张量，直接使用
                self.features = all_features

            print(f"成功加载特征: {feature_path}")
            print(f"特征数量: {len(self.features)}")
            print(f"特征维度: {self.features[0].shape if isinstance(self.features, list) else self.features.shape}")

        except Exception as e:
            print(f"加载特征失败: {e}")
            self.features = None
            self.preload_features = False

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载姿态
        pose = torch.tensor(self.poses[idx], dtype=torch.float32)

        # 如果预加载特征并且特征存在
        if self.preload_features and self.features is not None:
            # 直接返回预计算的特征
            feature = self.features[idx]
            # 确保特征是张量
            if not isinstance(feature, torch.Tensor):
                feature = torch.tensor(feature)
            return feature, pose
        else:
            # 否则返回原始图像
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            return image, pose


def create_dataloader(data_root="data", batch_size=32, split='train',
                      preload_features=False, clip_processor=None,
                      num_workers=4, shuffle=True, feature_path=None):
    """创建数据加载器的辅助函数"""
    dataset = PoseImageDataset(
        data_root=data_root,
        split=split,
        preload_features=preload_features,
        clip_processor=clip_processor,
        feature_path=feature_path
    )

    # 在Windows系统上设置num_workers=0
    if os.name == 'nt':  # Windows系统
        num_workers = 0
        print("检测到Windows系统，设置num_workers=0")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return dataloader