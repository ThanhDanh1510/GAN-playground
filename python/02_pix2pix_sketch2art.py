"""
=============================================================================
Lab 02: Pix2Pix – Bút Vẽ Phù Thủy (Sketch-to-Image với PyTorch)
=============================================================================
Dành cho học sinh: Tìm hiểu cấu trúc U-Net Generator (với Skip Connections) 
và PatchGAN Discriminator để biến nét vẽ đơn giản thành bức tranh hoàn thiện.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1. U-Net Generator (Kiến trúc Encoder-Decoder có Skip Connections)
class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True, use_dropout=False):
        super().__init__()
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            layers = [
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ]
            if use_dropout:
                layers.append(nn.Dropout(0.3))
            self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class MiniUNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        # Encoder (Thu nhỏ ảnh, trích xuất đặc trưng hình dáng nét vẽ)
        self.e1 = nn.Conv2d(in_channels, 32, 4, 2, 1) # 64 -> 32
        self.e2 = UNetBlock(32, 64, down=True)         # 32 -> 16
        self.e3 = UNetBlock(64, 128, down=True)        # 16 -> 8

        # Decoder (Phóng to ảnh & tô màu, kết hợp Skip Connections)
        self.d1 = UNetBlock(128, 64, down=False)       # 8 -> 16
        self.d2 = UNetBlock(64 + 64, 32, down=False)   # 16 -> 32 (Ghép nối Skip Connection e2)
        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(32 + 32, out_channels, 4, 2, 1), # 32 -> 64 (Ghép nối e1)
            nn.Tanh() # Ảnh chuẩn hóa về [-1, 1]
        )

    def forward(self, x):
        e1 = nn.LeakyReLU(0.2)(self.e1(x))
        e2 = self.e2(e1)
        e3 = self.e3(e2)

        d1 = self.d1(e3)
        d2 = self.d2(torch.cat([d1, e2], dim=1)) # Skip Connection 1
        out = self.d3(torch.cat([d2, e1], dim=1)) # Skip Connection 2
        return out

# 2. PatchGAN Discriminator (Chấm điểm từng ô vuông nhỏ trên ảnh)
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=4): # 1 kênh Sketch + 3 kênh Color = 4 kênh
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, sketch, color):
        x = torch.cat([sketch, color], dim=1)
        return self.net(x)

# 3. Hàm tạo tập dữ liệu hình học nhân tạo nhanh (Geometric Sketches to Colored Art)
def create_synthetic_paired_batch(batch_size=16, img_size=64):
    sketches = torch.zeros(batch_size, 1, img_size, img_size)
    colors = torch.zeros(batch_size, 3, img_size, img_size)

    for i in range(batch_size):
        # Tạo hình tròn, hình vuông hoặc tam giác ngẫu nhiên
        cx, cy = np.random.randint(20, 44, 2)
        r = np.random.randint(10, 18)
        color_choice = np.random.rand(3) * 2 - 1 # [-1, 1]

        # Vẽ nét viền cho Sketch
        y, x = np.ogrid[:img_size, :img_size]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        ring_mask = (dist >= r - 2) & (dist <= r + 2)
        sketches[i, 0, ring_mask] = 1.0

        # Tô màu đầy đủ cho Color
        fill_mask = dist <= r
        for c in range(3):
            colors[i, c, fill_mask] = color_choice[c]

    return sketches, colors

def train_pix2pix(epochs=60, batch_size=16, lambda_l1=100.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ Huấn luyện Pix2Pix trên thiết bị: {device}")

    G = MiniUNetGenerator().to(device)
    D = PatchGANDiscriminator().to(device)

    bce_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()

    opt_G = optim.Adam(G.parameters(), lr=0.002, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=0.002, betas=(0.5, 0.999))

    for epoch in range(1, epochs + 1):
        sketches, real_colors = create_synthetic_paired_batch(batch_size)
        sketches = sketches.to(device)
        real_colors = real_colors.to(device)

        # ---------------------
        #  Huấn luyện Cảnh sát D
        # ---------------------
        fake_colors = G(sketches)
        pred_real = D(sketches, real_colors)
        pred_fake = D(sketches, fake_colors.detach())

        loss_d_real = bce_loss(pred_real, torch.ones_like(pred_real))
        loss_d_fake = bce_loss(pred_fake, torch.zeros_like(pred_fake))
        loss_D = (loss_d_real + loss_d_fake) / 2

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ---------------------
        #  Huấn luyện Máy tạo G
        # ---------------------
        pred_fake_g = D(sketches, fake_colors)
        loss_g_gan = bce_loss(pred_fake_g, torch.ones_like(pred_fake_g))
        loss_g_l1 = l1_loss(fake_colors, real_colors) * lambda_l1
        loss_G = loss_g_gan + loss_g_l1

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs}] | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f} (L1: {loss_g_l1.item():.2f})")

    # Lưu kết quả trực quan Before / After
    os.makedirs("results", exist_ok=True)
    with torch.no_grad():
        test_sketches, test_reals = create_synthetic_paired_batch(4)
        test_sketches = test_sketches.to(device)
        test_fakes = G(test_sketches).cpu()

    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    for i in range(4):
        axes[0, i].imshow(test_sketches[i, 0].cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f"Nét vẽ {i+1}")
        axes[0, i].axis('off')

        axes[1, i].imshow(((test_fakes[i].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1))
        axes[1, i].set_title(f"AI Tô màu")
        axes[1, i].axis('off')

        axes[2, i].imshow(((test_reals[i].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1))
        axes[2, i].set_title(f"Ảnh Gốc (Mẫu)")
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig("results/02_pix2pix_sketch2art_result.png", dpi=150)
    plt.close()
    print("✓ Đã lưu kết quả tại results/02_pix2pix_sketch2art_result.png")

if __name__ == "__main__":
    train_pix2pix(epochs=50)
