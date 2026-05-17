# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    """
    VAE损失函数：重构损失 + KL散度
    """

    def __init__(self, beta=0.001, recon_loss_type='mse'):
        """
        Args:
            beta: KL散度的权重
            recon_loss_type: 'mse' 或 'l1' 或 'smooth_l1'
        """
        super().__init__()
        self.beta = beta
        self.recon_loss_type = recon_loss_type

        if recon_loss_type == 'mse':
            self.recon_loss_fn = nn.MSELoss(reduction='mean')
        elif recon_loss_type == 'l1':
            self.recon_loss_fn = nn.L1Loss(reduction='mean')
        elif recon_loss_type == 'smooth_l1':
            self.recon_loss_fn = nn.SmoothL1Loss(reduction='mean')
        else:
            raise ValueError(f"Unknown recon_loss_type: {recon_loss_type}")

    def forward(self, pose_pred, pose_gt, mu, logvar):
        """
        Args:
            pose_pred: 预测的姿态 (batch_size, pose_dim)
            pose_gt: 真实的姿态 (batch_size, pose_dim)
            mu: 均值 (batch_size, latent_dim)
            logvar: 对数方差 (batch_size, latent_dim)

        Returns:
            total_loss: 总损失
            recon_loss: 重构损失
            kl_loss: KL散度
        """
        # 1. 重构损失
        recon_loss = self.recon_loss_fn(pose_pred, pose_gt)

        # 2. KL散度
        # 公式: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / pose_gt.shape[0]  # 平均每个样本

        # 3. 总损失
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss


class PoseMetrics:
    """姿态评估指标"""

    @staticmethod
    def mpjpe(pred_poses, gt_poses):
        """
        平均关节位置误差 (Mean Per Joint Position Error)
        假设姿态是 (batch_size, 75) 其中每3个值是一个关节点的坐标
        """
        batch_size = pred_poses.shape[0]
        pred_poses = pred_poses.reshape(batch_size, -1, 3)
        gt_poses = gt_poses.reshape(batch_size, -1, 3)

        # 计算每个关节的欧氏距离
        distances = torch.norm(pred_poses - gt_poses, dim=2)  # (batch_size, num_joints)

        # 平均
        return distances.mean().item()

    @staticmethod
    def diversity(poses):
        """
        计算生成姿态的多样性（平均成对距离）
        poses: (batch_size, num_samples, pose_dim)
        """
        batch_size, num_samples, pose_dim = poses.shape

        # 重塑为 (batch_size * num_samples, pose_dim)
        poses_flat = poses.reshape(-1, pose_dim)

        # 计算所有样本间的成对距离
        total_diversity = 0
        for i in range(batch_size):
            batch_poses = poses[i]  # (num_samples, pose_dim)
            # 计算两两之间的距离
            distances = torch.cdist(batch_poses, batch_poses, p=2)  # (num_samples, num_samples)
            # 取上三角的平均（排除对角线）
            mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
            avg_distance = distances[mask].mean()
            total_diversity += avg_distance

        return (total_diversity / batch_size).item()