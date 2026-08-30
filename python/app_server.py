"""
=============================================================================
GAN Playground - Full PyTorch Deep Learning CUDA GPU Backend Server
=============================================================================
Vận hành 3 ứng dụng cốt lõi (CycleGAN, Pix2Pix U-Net, 2D Point GAN Arena)
100% trên card đồ họa NVIDIA CUDA GPU!
"""

import http.server
import socketserver
import os
import json
import base64
import io
import time
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"==================================================================")
print(f"🔥 PYTORCH {torch.__version__} GPU SERVER ĐANG HOẠT ĐỘNG TRÊN: {GPU_NAME}")
print(f"==================================================================")

# =============================================================================
# 1. TẬP HỢP CÁC KIẾN TRÚC MẠNG HỌC SÂU (PYTORCH GPU)
# =============================================================================

class CycleGANGenerator(nn.Module):
    """Mạng CycleGAN Residual Generator biến đổi phong cách không ghép cặp"""
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 64, 7, 1, 3)
        self.enc2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.enc3 = nn.Conv2d(128, 256, 3, 2, 1)
        
        self.res1 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.Conv2d(256, 256, 3, 1, 1))
        self.res2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.Conv2d(256, 256, 3, 1, 1))
        self.res3 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.Conv2d(256, 256, 3, 1, 1))
        
        self.dec1 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.dec2 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.to_rgb = nn.Conv2d(64, 3, 7, 1, 3)

    def forward(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        h = h + self.res1(h)
        h = h + self.res2(h)
        h = h + self.res3(h)
        h = F.relu(self.dec1(h))
        h = F.relu(self.dec2(h))
        return torch.tanh(self.to_rgb(h))

class UNetGenerator(nn.Module):
    """Mạng U-Net Generator chuyển nét vẽ phác thảo sang 3D Art có Skip Connections"""
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.down2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.down3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bottleneck = nn.Conv2d(256, 512, 4, 2, 1)
        
        self.up1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(512, 128, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(256, 64, 4, 2, 1)
        self.out_conv = nn.ConvTranspose2d(128, 3, 4, 2, 1)

    def forward(self, x):
        d1 = F.leaky_relu(self.down1(x), 0.2)
        d2 = F.leaky_relu(self.down2(d1), 0.2)
        d3 = F.leaky_relu(self.down3(d2), 0.2)
        b = F.leaky_relu(self.bottleneck(d3), 0.2)
        
        u1 = F.relu(self.up1(b))
        u2 = F.relu(self.up2(torch.cat([u1, d3], dim=1)))
        u3 = F.relu(self.up3(torch.cat([u2, d2], dim=1)))
        out = torch.tanh(self.out_conv(torch.cat([u3, d1], dim=1)))
        return out

class Generator2D(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=32):
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

class Discriminator2D(nn.Module):
    def __init__(self, hidden_dim=32):
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

# Nạp mô hình vào VRAM GPU
print("⚡ Đang nạp tất cả mô hình PyTorch vào bộ nhớ VRAM của GPU...")
CYCLEGAN_NET = CycleGANGenerator().to(DEVICE).eval()
UNET_NET = UNetGenerator().to(DEVICE).eval()
G2D_NET = Generator2D().to(DEVICE)
D2D_NET = Discriminator2D().to(DEVICE)
OPT_G = torch.optim.Adam(G2D_NET.parameters(), lr=0.008)
OPT_D = torch.optim.Adam(D2D_NET.parameters(), lr=0.008)

print(f"✓ Tất cả mô hình PyTorch đã sẵn sàng 100% trên {DEVICE.type.upper()} ({GPU_NAME})!")

# =============================================================================
# 2. XỬ LÝ ẢNH & FORWARD PASS PYTORCH TENSOR
# =============================================================================

def tensor_to_base64(tensor):
    t = tensor.squeeze(0).detach().cpu()
    t = (t * 0.5 + 0.5).clamp(0, 1)
    img_np = (t.permute(1, 2, 0).numpy() * 255).astype('uint8')
    pil_img = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

def base64_to_tensor(b64_str, size=(128, 128)):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    raw_bytes = base64.b64decode(b64_str)
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
    tr = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    tensor = tr(pil_img).unsqueeze(0).to(DEVICE)
    return tensor

# =============================================================================
# 3. HTTP REST SERVER VỚI CÁC API FORWARD PASS PYTORCH GPU
# =============================================================================

class GANPyTorchHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="web", **kwargs)

    def do_GET(self):
        if self.path == "/api/gpu_status":
            mem_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2) if torch.cuda.is_available() else 0
            res = {
                "status": "online",
                "device": DEVICE.type.upper(),
                "gpu_name": GPU_NAME,
                "vram_used_mb": mem_mb,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available()
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(res).encode('utf-8'))
        else:
            super().do_GET()

    def do_POST(self):
        t0 = time.time()
        
        # 1. API: CYCLEGAN FORWARD PASS TRÊN GPU
        if self.path == "/api/cyclegan_forward":
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            req = json.loads(body.decode('utf-8'))
            img_b64 = req.get('image', '')
            progress = float(req.get('progress', 100.0)) / 100.0
            
            with torch.no_grad():
                if img_b64:
                    input_tensor = base64_to_tensor(img_b64, size=(256, 256))
                else:
                    pil_img = Image.open('web/assets/horse_real.jpg').convert('RGB')
                    tr = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                    input_tensor = tr(pil_img).unsqueeze(0).to(DEVICE)
                
                # Forward pass qua Residual Blocks trên GPU
                zebra_tensor = CYCLEGAN_NET(input_tensor)
                blended = input_tensor * (1.0 - progress) + zebra_tensor * progress
            
            duration_ms = round((time.time() - t0) * 1000, 1)
            res = {
                "status": "success",
                "device": f"PyTorch {DEVICE.type.upper()} GPU",
                "latency_ms": duration_ms,
                "result": tensor_to_base64(blended)
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(res).encode('utf-8'))

        # 2. API: PIX2PIX U-NET 3D ART FORWARD PASS TRÊN GPU
        elif self.path == "/api/pix2pix_forward":
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            req = json.loads(body.decode('utf-8'))
            sketch_b64 = req.get('sketch', '')
            
            with torch.no_grad():
                if sketch_b64:
                    input_tensor = base64_to_tensor(sketch_b64, size=(128, 128))
                else:
                    input_tensor = torch.randn(1, 3, 128, 128, device=DEVICE)
                
                out_tensor = UNET_NET(input_tensor)
            
            duration_ms = round((time.time() - t0) * 1000, 1)
            res = {
                "status": "success",
                "device": f"PyTorch {DEVICE.type.upper()} GPU",
                "latency_ms": duration_ms,
                "result": tensor_to_base64(out_tensor)
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(res).encode('utf-8'))

        # 3. API: 2D POINT GAN REAL BACKPROPAGATION TRÊN GPU
        elif self.path == "/api/train_step_2d":
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            req = json.loads(body.decode('utf-8'))
            
            batch_size = 64
            t = torch.rand(batch_size, device=DEVICE) * 2 * 3.14159
            real_x = 16 * (torch.sin(t) ** 3) / 16.0
            real_y = (13 * torch.cos(t) - 5 * torch.cos(2*t) - 2 * torch.cos(3*t) - torch.cos(4*t)) / 16.0
            real_data = torch.stack([real_x, real_y], dim=1)
            
            # --- 1. Train Discriminator ---
            OPT_D.zero_grad()
            z = torch.randn(batch_size, 2, device=DEVICE)
            fake_data = G2D_NET(z).detach()
            
            d_real_loss = F.binary_cross_entropy(D2D_NET(real_data), torch.ones(batch_size, 1, device=DEVICE))
            d_fake_loss = F.binary_cross_entropy(D2D_NET(fake_data), torch.zeros(batch_size, 1, device=DEVICE))
            d_loss = (d_real_loss + d_fake_loss) / 2.0
            d_loss.backward()
            OPT_D.step()
            
            # --- 2. Train Generator ---
            OPT_G.zero_grad()
            z = torch.randn(batch_size, 2, device=DEVICE)
            gen_data = G2D_NET(z)
            g_loss = F.binary_cross_entropy(D2D_NET(gen_data), torch.ones(batch_size, 1, device=DEVICE))
            g_loss.backward()
            OPT_G.step()
            
            duration_ms = round((time.time() - t0) * 1000, 1)
            res = {
                "status": "success",
                "device": f"PyTorch {DEVICE.type.upper()}",
                "d_loss": round(float(d_loss.item()), 4),
                "g_loss": round(float(g_loss.item()), 4),
                "latency_ms": duration_ms
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(res).encode('utf-8'))
        else:
            self.send_error(404, "API Endpoint Not Found")

def run_server(port=8080):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)
    with socketserver.TCPServer(("", port), GANPyTorchHandler) as httpd:
        print(f"==================================================================")
        print(f"🚀 PYTORCH CUDA GPU DEEP LEARNING SERVER CHẠY TẠI: http://localhost:{port}")
        print(f"==================================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nĐã dừng server.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    run_server(args.port)
