"""
=============================================================================
Lab 04: StyleGAN Latent Studio – Đại Số Vector & Biến Hình (Morphing)
=============================================================================
Dành cho học sinh: Tìm hiểu cách làm phép toán đại số trong Không gian Tiềm ẩn (Latent Space):
Vector(Mặt cười có kính) = Vector(Mặt gốc) + alpha * Vector(Nụ cười) + beta * Vector(Kính râm)
và nội suy chuyển động mượt mà giữa 2 người (Latent Interpolation / Morphing).
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 1. Deep Convolutional Generator (DCGAN Generator tạo ảnh mặt người / Emoji 32x32)
class FaceGenerator(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 4, 4)
        return self.conv(x)

def demo_latent_space_arithmetic():
    latent_dim = 32
    G = FaceGenerator(latent_dim=latent_dim)
    G.eval()

    # 1. Định nghĩa các điểm mốc trong không gian tiềm ẩn
    torch.manual_seed(42)
    z_neutral = torch.randn(1, latent_dim) # Mặt bình thường

    # Tìm vector thuộc tính bằng phương pháp sai phân (Difference Vector)
    # Trong thực tế với StyleGAN: v_smile = Mean(z_smiling_faces) - Mean(z_neutral_faces)
    v_smile = torch.randn(1, latent_dim) * 0.4
    v_glasses = torch.randn(1, latent_dim) * 0.5

    # Phép toán 1: Thêm nụ cười
    z_smiling = z_neutral + 1.2 * v_smile
    # Phép toán 2: Thêm kính râm
    z_glasses = z_neutral + 1.5 * v_glasses
    # Phép toán 3: Vừa cười vừa đeo kính!
    z_combo = z_neutral + 1.2 * v_smile + 1.5 * v_glasses

    with torch.no_grad():
        img_neutral = (G(z_neutral)[0].permute(1, 2, 0) + 1) / 2
        img_smiling = (G(z_smiling)[0].permute(1, 2, 0) + 1) / 2
        img_glasses = (G(z_glasses)[0].permute(1, 2, 0) + 1) / 2
        img_combo = (G(z_combo)[0].permute(1, 2, 0) + 1) / 2

    # 2. Phép biến hình (Latent Morphing Interpolation giữa Người A và Người B)
    z_person_A = torch.randn(1, latent_dim)
    z_person_B = torch.randn(1, latent_dim)

    morph_steps = 7
    alphas = np.linspace(0, 1, morph_steps)
    morph_imgs = []

    with torch.no_grad():
        for a in alphas:
            z_interp = (1 - a) * z_person_A + a * z_person_B
            img_step = (G(z_interp)[0].permute(1, 2, 0) + 1) / 2
            morph_imgs.append(img_step.clamp(0, 1).numpy())

    # Lưu biểu đồ minh họa
    os.makedirs("results", exist_ok=True)
    fig = plt.figure(figsize=(12, 6))

    # Hàng 1: Đại số Vector
    ax1 = fig.add_subplot(2, 4, 1); ax1.imshow(img_neutral.clamp(0, 1).numpy()); ax1.set_title("1. Gốc (z_base)"); ax1.axis('off')
    ax2 = fig.add_subplot(2, 4, 2); ax2.imshow(img_smiling.clamp(0, 1).numpy()); ax2.set_title("2. + Vector Nụ cười"); ax2.axis('off')
    ax3 = fig.add_subplot(2, 4, 3); ax3.imshow(img_glasses.clamp(0, 1).numpy()); ax3.set_title("3. + Vector Kính"); ax3.axis('off')
    ax4 = fig.add_subplot(2, 4, 4); ax4.imshow(img_combo.clamp(0, 1).numpy()); ax4.set_title("4. Cười + Đeo Kính"); ax4.axis('off')

    # Hàng 2: Morphing từ Người A sang Người B
    for idx, img in enumerate(morph_imgs):
        ax = fig.add_subplot(2, morph_steps, morph_steps + idx + 1)
        ax.imshow(img)
        ax.set_title(f"{int(alphas[idx]*100)}%")
        ax.axis('off')

    plt.suptitle("Đại Số Vector Không Gian Tiềm Ẩn & Biến Hình (Latent Morphing)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/04_latent_arithmetic_result.png", dpi=150)
    plt.close()
    print("✓ Đã lưu kết quả tại results/04_latent_arithmetic_result.png")

if __name__ == "__main__":
    demo_latent_space_arithmetic()
