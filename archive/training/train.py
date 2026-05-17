# train.py
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from CNN_model import SceneConditionedPoseVAE
from dataset import create_dataloader
from losses import VAELoss, PoseMetrics


def parse_args():
    parser = argparse.ArgumentParser(description='训练场景条件姿态VAE')

    # 数据参数
    parser.add_argument('--data_root', type=str, default='data',
                        help='数据根目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')

    # 模型参数
    parser.add_argument('--pose_dim', type=int, default=75,
                        help='姿态维度')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='潜在空间维度')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--beta', type=float, default=0.001,
                        help='KL散度权重')
    parser.add_argument('--beta_final', type=float, default=0.1,
                        help='KL散度最终权重（用于annealing）')
    parser.add_argument('--beta_warmup', type=int, default=10,
                        help='beta warmup的轮数')
    parser.add_argument('--recon_loss', type=str, default='mse',
                        choices=['mse', 'l1', 'smooth_l1'],
                        help='重构损失类型')

    # 其他参数
    parser.add_argument('--use_precomputed_features', default='True',
                        help='使用预计算的特征')
    parser.add_argument('--feature_path', type=str, default='data\\scene_features.pt',
                        help='预计算特征文件路径')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的checkpoint路径')
    parser.add_argument('--device', type=str, default='cpu',
                        help='训练设备')

    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device, beta_current):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (data, poses) in enumerate(pbar):
        poses = poses.to(device)

        if isinstance(data, torch.Tensor) and data.shape[1] == 512:
            # 如果是预计算的特征
            scene_feat = data.to(device)
            # 手动进行前向传播（因为模型期望rgb）
            # 这里需要修改模型或手动处理
            # 为了简化，我们假设模型有一个接受特征的方法
            # 这里我们先实现一个临时方案
            x = torch.cat([poses, scene_feat], dim=-1)
            h = model.encoder(x)
            mu = model.mu(h)
            logvar = model.logvar(h)
            z = model.reparameterize(mu, logvar)
            dec_input = torch.cat([z, scene_feat], dim=-1)
            pose_pred = model.decoder(dec_input)
        else:
            # 如果是原始图像
            data = data.to(device)
            pose_pred, mu, logvar = model(data, poses)

        # 计算损失
        loss, recon, kl = criterion(pose_pred, poses, mu, logvar)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 记录
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Recon': f'{recon.item():.4f}',
            'KL': f'{kl.item():.4f}',
            'Beta': f'{beta_current:.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_recon = total_recon / len(dataloader)
    avg_kl = total_kl / len(dataloader)

    return avg_loss, avg_recon, avg_kl

def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_mpjpe = 0

    with torch.no_grad():
        for data, poses in tqdm(dataloader, desc='Validating'):
            poses = poses.to(device)

            if isinstance(data, torch.Tensor) and data.shape[1] == 512:
                scene_feat = data.to(device)
                x = torch.cat([poses, scene_feat], dim=-1)
                h = model.encoder(x)
                mu = model.mu(h)
                logvar = model.logvar(h)
                z = model.reparameterize(mu, logvar)
                dec_input = torch.cat([z, scene_feat], dim=-1)
                pose_pred = model.decoder(dec_input)
            else:
                data = data.to(device)
                pose_pred, mu, logvar = model(data, poses)

            loss, recon, kl = criterion(pose_pred, poses, mu, logvar)

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()

            # 计算MPJPE
            mpjpe = PoseMetrics.mpjpe(pose_pred, poses)
            total_mpjpe += mpjpe

    avg_loss = total_loss / len(dataloader)
    avg_recon = total_recon / len(dataloader)
    avg_kl = total_kl / len(dataloader)
    avg_mpjpe = total_mpjpe / len(dataloader)

    return avg_loss, avg_recon, avg_kl, avg_mpjpe

def save_checkpoint(model, optimizer, epoch, loss, args, is_best=False):
    """保存checkpoint"""
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': args
    }

    # 常规保存
    torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

    # 最佳模型
    if is_best:
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_model.pth'))

    # 最新模型
    torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'latest.pth'))

def plot_losses(train_losses, val_losses, save_path='loss_plot.png'):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(save_path)
    plt.close()

def main():
    args = parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    model = SceneConditionedPoseVAE(
        pose_dim=args.pose_dim,
        latent_dim=args.latent_dim
    ).to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 创建数据加载器
    print("创建数据加载器...")
    if args.use_precomputed_features:
        # 加载预计算特征
        print(f"加载预计算特征: {args.feature_path}")
        all_features = torch.load(args.feature_path)
        # 这里需要将特征按分割分配给train/val
        # 简化：假设我们已经有了分割好的特征文件
        train_loader = create_dataloader(
            args.data_root, args.batch_size, 'train',
            preload_features=True, num_workers=args.num_workers
        )
        val_loader = create_dataloader(
            args.data_root, args.batch_size, 'val',
            preload_features=True, num_workers=args.num_workers
        )
    else:
        train_loader = create_dataloader(
            args.data_root, args.batch_size, 'train',
            preload_features=False, num_workers=args.num_workers
        )
        val_loader = create_dataloader(
            args.data_root, args.batch_size, 'val',
            preload_features=False, num_workers=args.num_workers
        )

    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")

    # 创建损失函数
    criterion = VAELoss(beta=args.beta, recon_loss_type=args.recon_loss)

    # 创建优化器
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # 训练记录
    train_losses = []
    val_losses = []
    val_mpjpes = []

    # 训练循环
    print("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Beta annealing
        if epoch < args.beta_warmup:
            beta_current = args.beta + (args.beta_final - args.beta) * (epoch / args.beta_warmup)
        else:
            beta_current = args.beta_final
        criterion.beta = beta_current

        # 训练
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, criterion, optimizer, device, beta_current
        )

        # 验证
        val_loss, val_recon, val_kl, val_mpjpe = validate(
            model, val_loader, criterion, device
        )

        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_mpjpes.append(val_mpjpe)

        # 打印结果
        print(f"Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}, MPJPE: {val_mpjpe:.4f}")

        # 学习率调整
        scheduler.step(val_loss)

        # 保存最佳模型
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"新的最佳模型! Loss: {val_loss:.4f}")

        # 定期保存
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, args, is_best)

    # 训练结束，绘制损失曲线
    plot_losses(train_losses, val_losses, os.path.join(args.checkpoint_dir, 'loss_plot.png'))

    print("训练完成!")

if __name__ == "__main__":
    main()