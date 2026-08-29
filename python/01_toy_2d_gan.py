"""
=============================================================================
Lab 01: 2D Point GAN trong 35 dòng PyTorch (Cơ chế Đối Kháng Cốt Lõi)
=============================================================================
Dành cho học sinh: Quan sát Kẻ làm giả (Generator) biến đổi nhiễu ngẫu nhiên
thành hình dạng Trái tim (Heart) hoặc Đường xoắn ốc để đánh lừa Cảnh sát (Discriminator).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1. Định nghĩa Generator (Kẻ làm tranh giả) & Discriminator (Cảnh sát)
class Generator(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# 2. Hàm sinh dữ liệu mẫu: Hình trái tim 💖
def get_heart_distribution(batch_size=256):
    t = torch.rand(batch_size) * 2 * np.pi - np.pi
    x = 16 * (torch.sin(t) ** 3) / 18
    y = (13 * torch.cos(t) - 5 * torch.cos(2*t) - 2 * torch.cos(3*t) - torch.cos(4*t)) / 18
    noise = torch.randn(batch_size, 2) * 0.04
    return torch.stack([x, y], dim=1) + noise

def train_2d_gan(epochs=300, batch_size=256, lr=0.005, save_fig=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ Đang chạy trên thiết bị: {device}")

    G = Generator().to(device)
    D = Discriminator().to(device)
    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(1, epochs + 1):
        # --- BƯỚC 1: CẬP NHẬT DISCRIMINATOR ---
        real_data = get_heart_distribution(batch_size).to(device)
        noise = torch.randn(batch_size, 2).to(device)
        fake_data = G(noise)

        # Cảnh sát chấm điểm thật (1) và giả (0)
        d_real_loss = criterion(D(real_data), torch.ones(batch_size, 1, device=device))
        d_fake_loss = criterion(D(fake_data.detach()), torch.zeros(batch_size, 1, device=device))
        d_loss = (d_real_loss + d_fake_loss) / 2

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # --- BƯỚC 2: CẬP NHẬT GENERATOR ---
        # Generator muốn Cảnh sát chấm điểm 1 cho ảnh giả của mình
        g_loss = criterion(D(fake_data), torch.ones(batch_size, 1, device=device))

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        if epoch % 50 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:03d}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    if save_fig:
        os.makedirs("results", exist_ok=True)
        with torch.no_grad():
            real_samples = get_heart_distribution(500).cpu().numpy()
            fake_samples = G(torch.randn(500, 2, device=device)).cpu().numpy()

        plt.figure(figsize=(6, 6))
        plt.scatter(real_samples[:, 0], real_samples[:, 1], c='#f97316', s=12, label='Dữ liệu Thật (Trái tim)', alpha=0.7)
        plt.scatter(fake_samples[:, 0], fake_samples[:, 1], c='#06b6d4', s=12, label='Dữ liệu do Generator tạo', alpha=0.7)
        plt.title(f"Kết Quả 2D GAN Sau {epochs} Vòng Huấn Luyện")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("results/01_2d_heart_result.png", dpi=150)
        plt.close()
        print("✓ Đã lưu kết quả tại results/01_2d_heart_result.png")

if __name__ == "__main__":
    train_2d_gan(epochs=200)
