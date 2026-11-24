import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class SceneConditionedPoseVAE(nn.Module):

    def __init__(self, pose_dim=75, latent_dim=64):
        super().__init__()

        # ========= Scene Encoder (CLIP) =========
        # 提取特征信息
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        for p in self.clip.parameters():
            p.requires_grad = False

        scene_dim = 512

        # ========= Encoder =========
        # 将姿势与场景结合
        self.encoder = nn.Sequential(
            nn.Linear(pose_dim + scene_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

        # ========= Decoder =========
        # 还原姿势
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + scene_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, pose_dim)
        )

    # # 对数字归一化
    # def encode_scene(self, rgb):
    #     with torch.no_grad():
    #         inputs = self.clip_processor(images=rgb, return_tensors="pt").to(rgb.device)
    #         feat = self.clip.get_image_features(**inputs)
    #         feat = feat / feat.norm(dim=-1, keepdim=True)
    #     return feat

    def encode_scene(self, rgb):
        with torch.no_grad():
            inputs = self.clip_processor(images=rgb, return_tensors="pt").to(self.clip.device)
            feat = self.clip.get_image_features(**inputs)
            # transformers v5 起返回 BaseModelOutputWithPooling，取投影后特征
            if not torch.is_tensor(feat):
                feat = feat.pooler_output
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    # 随机采样（核心要点，变化范围）
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_with_scene_features(self, pose_gt, scene_feat):
        """与训练时一致：拼接 (pose, scene) 后编码，再与 scene 解码。"""
        x = torch.cat([pose_gt, scene_feat], dim=-1)
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        dec_input = torch.cat([z, scene_feat], dim=-1)
        pose_pred = self.decoder(dec_input)
        return pose_pred, mu, logvar

    def generate_with_scene_features(self, scene_feat, num_samples=5):
        """仅依赖场景特征，从先验采样 z 生成姿态 (B, num_samples, pose_dim)。"""
        B = scene_feat.shape[0]
        z = torch.randn(B * num_samples, self.mu.out_features, device=scene_feat.device, dtype=scene_feat.dtype)
        sf = scene_feat.unsqueeze(1).expand(-1, num_samples, -1).reshape(B * num_samples, -1)
        dec_input = torch.cat([z, sf], dim=-1)
        poses = self.decoder(dec_input)
        return poses.reshape(B, num_samples, -1)

    def forward(self, rgb, pose_gt):
        scene_feat = self.encode_scene(rgb)
        return self.forward_with_scene_features(pose_gt, scene_feat)

    def generate(self, rgb, num_samples=5):
        scene_feat = self.encode_scene(rgb)
        return self.generate_with_scene_features(scene_feat, num_samples=num_samples)
