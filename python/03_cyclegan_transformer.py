"""
=============================================================================
Lab 03: CycleGAN – Máy Biến Đổi Thế Giới (Unpaired Image Translation)
=============================================================================
Dành cho học sinh: Tìm hiểu nguyên lý biến đổi qua lại giữa 2 miền không ghép cặp 
(Ngựa <-> Ngựa vằn) bằng vòng lặp khép kín: Cycle Consistency Loss:
L_cycle = ||F(G(A)) - A|| + ||G(F(B)) - B||
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1. Kiến trúc Generator ResNet / Conv mini cho CycleGAN
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

class CycleGenerator(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_res=3):
        super().__init__()
        # Initial Conv
        model = [
            nn.Conv2d(in_c, 32, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            # Downsampling
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        # Residual Blocks
        for _ in range(n_res):
            model.append(ResidualBlock(64))
        # Upsampling
        model += [
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_c, 7, 1, 3),
            nn.Tanh()
        ]
        self.net = nn.Sequential(*model)

    def forward(self, x):
        return self.net(x)

# 2. Discriminator (PatchGAN)
class CycleDiscriminator(nn.Module):
    def __init__(self, in_c=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 4, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# 3. Tạo tập dữ liệu mô phỏng 2 miền: Miền A (Đơn sắc tròn) & Miền B (Vằn sọc)
def get_unpaired_batch(batch_size=8, img_size=32):
    # Miền A: Khối tròn màu trơn
    domain_A = torch.zeros(batch_size, 3, img_size, img_size)
    # Miền B: Khối có các đường kẻ sọc (Zebra stripes pattern)
    domain_B = torch.zeros(batch_size, 3, img_size, img_size)

    for i in range(batch_size):
        # Domain A
        cy, cx = np.random.randint(10, 22, 2)
        y, x = np.ogrid[:img_size, :img_size]
        mask_a = (x - cx)**2 + (y - cy)**2 <= 8**2
        domain_A[i, 0, mask_a] = 0.8  # Đỏ nâu
        domain_A[i, 1, mask_a] = 0.4
        domain_A[i, 2, mask_a] = 0.1

        # Domain B
        mask_b = (x - cx)**2 + (y - cy)**2 <= 8**2
        domain_B[i, :, mask_b] = 0.9  # Nền trắng
        stripes = (x % 4 == 0) & mask_b
        domain_B[i, :, stripes] = -0.9 # Sọc đen [-1, 1]

    return domain_A * 2 - 1, domain_B

def train_cyclegan(epochs=40, batch_size=8, lambda_cycle=10.0, lambda_id=5.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ Huấn luyện CycleGAN trên thiết bị: {device}")

    # 2 Generators: G (A -> B) và F (B -> A)
    G_AB = CycleGenerator().to(device)
    F_BA = CycleGenerator().to(device)

    # 2 Discriminators: D_A (soi ảnh A) và D_B (soi ảnh B)
    D_A = CycleDiscriminator().to(device)
    D_B = CycleDiscriminator().to(device)

    criterion_gan = nn.BCELoss()
    criterion_cycle = nn.L1Loss()
    criterion_id = nn.L1Loss()

    opt_G = optim.Adam(list(G_AB.parameters()) + list(F_BA.parameters()), lr=0.001, betas=(0.5, 0.999))
    opt_D_A = optim.Adam(D_A.parameters(), lr=0.001, betas=(0.5, 0.999))
    opt_D_B = optim.Adam(D_B.parameters(), lr=0.001, betas=(0.5, 0.999))

    for epoch in range(1, epochs + 1):
        real_A, real_B = get_unpaired_batch(batch_size)
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # --------------------------------
        # 1. Huấn luyện 2 Generators G & F
        # --------------------------------
        # 1.1 Identity Loss
        loss_id_A = criterion_id(F_BA(real_A), real_A) * lambda_id
        loss_id_B = criterion_id(G_AB(real_B), real_B) * lambda_id

        # 1.2 Adversarial Loss
        fake_B = G_AB(real_A)
        pred_fake_B = D_B(fake_B)
        loss_gan_AB = criterion_gan(pred_fake_B, torch.ones_like(pred_fake_B))

        fake_A = F_BA(real_B)
        pred_fake_A = D_A(fake_A)
        loss_gan_BA = criterion_gan(pred_fake_A, torch.ones_like(pred_fake_A))

        # 1.3 Cycle Consistency Loss: A -> B -> A' và B -> A -> B'
        rec_A = F_BA(fake_B)
        loss_cycle_A = criterion_cycle(rec_A, real_A) * lambda_cycle

        rec_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(rec_B, real_B) * lambda_cycle

        total_loss_G = (
            loss_gan_AB + loss_gan_BA +
            loss_cycle_A + loss_cycle_B +
            loss_id_A + loss_id_B
        )

        opt_G.zero_grad()
        total_loss_G.backward()
        opt_G.step()

        # --------------------------------
        # 2. Huấn luyện 2 Discriminators
        # --------------------------------
        # Train D_A
        pred_real_A = D_A(real_A)
        pred_fake_A_det = D_A(fake_A.detach())
        loss_D_A = (criterion_gan(pred_real_A, torch.ones_like(pred_real_A)) + 
                    criterion_gan(pred_fake_A_det, torch.zeros_like(pred_fake_A_det))) / 2

        opt_D_A.zero_grad()
        loss_D_A.backward()
        opt_D_A.step()

        # Train D_B
        pred_real_B = D_B(real_B)
        pred_fake_B_det = D_B(fake_B.detach())
        loss_D_B = (criterion_gan(pred_real_B, torch.ones_like(pred_real_B)) + 
                    criterion_gan(pred_fake_B_det, torch.zeros_like(pred_fake_B_det))) / 2

        opt_D_B.zero_grad()
        loss_D_B.backward()
        opt_D_B.step()

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs}] | Cycle Loss: {(loss_cycle_A + loss_cycle_B).item():.4f} | D_A: {loss_D_A.item():.3f} | D_B: {loss_D_B.item():.3f}")

    # Xuất ảnh kết quả vòng lặp Cycle A -> B -> A'
    os.makedirs("results", exist_ok=True)
    with torch.no_grad():
        sample_A, sample_B = get_unpaired_batch(2)
        sample_A = sample_A.to(device)
        trans_B = G_AB(sample_A).cpu()
        recon_A = F_BA(G_AB(sample_A)).cpu()

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for i in range(2):
        axes[i, 0].imshow(((sample_A[i].cpu().permute(1, 2, 0) + 1) / 2).clip(0, 1))
        axes[i, 0].set_title("1. Gốc A (Ngựa)")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(((trans_B[i].permute(1, 2, 0) + 1) / 2).clip(0, 1))
        axes[i, 1].set_title("2. G(A) Biến Đổi (Ngựa vằn)")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(((recon_A[i].permute(1, 2, 0) + 1) / 2).clip(0, 1))
        axes[i, 2].set_title("3. F(G(A)) Tái tạo khép kín")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("results/03_cyclegan_result.png", dpi=150)
    plt.close()
    print("✓ Đã lưu kết quả tại results/03_cyclegan_result.png")

if __name__ == "__main__":
    train_cyclegan(epochs=30)
