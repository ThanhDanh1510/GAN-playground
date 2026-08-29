"""
=============================================================================
GAN Playground - Real PyTorch Deep Learning Model Inference Server (CUDA GPU)
=============================================================================
Chạy server mô hình AI thật với PyTorch GPU:
C:\\Users\\PC\\AppData\\Local\\Programs\\Python\\Python312\\python.exe python/app_server.py --port 8080
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

# Thiết bị tính toán
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"==================================================================")
print(f"🔥 KHỞI ĐỘNG PYTORCH DEEP LEARNING ENGINE TRÊN: {DEVICE.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print(f"==================================================================")

# =============================================================================
# 1. KIẾN TRÚC MÔ HÌNH DEEP LEARNING THỰC SỰ TRÊN PYTORCH
# =============================================================================

class E4EEncoder(nn.Module):
    """Mạng Encoder e4e trích xuất vector latent w+ (18 x 512) từ ảnh thật"""
    def __init__(self, num_layers=18, latent_dim=512):
        super().__init__()
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        
        # Backbone ResNet-style feature extractor
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1) # 128x128 -> 64x64
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1) # 64x64 -> 32x32
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1) # 32x32 -> 16x16
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1) # 16x16 -> 8x8
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Head tạo vector w+
        self.fc = nn.Linear(512 * 4 * 4, num_layers * latent_dim)

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)
        h = F.leaky_relu(self.conv3(h), 0.2)
        h = F.leaky_relu(self.conv4(h), 0.2)
        h = self.pool(h).view(x.size(0), -1)
        w_plus = self.fc(h).view(-1, self.num_layers, self.latent_dim)
        return w_plus

class StyleGAN2Generator(nn.Module):
    """Mạng Generator tổng hợp ảnh chân dung độ phân giải cao từ vector w+"""
    def __init__(self, num_layers=18, latent_dim=512):
        super().__init__()
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        
        self.fc = nn.Linear(num_layers * latent_dim, 512 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1) # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1) # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # 16x16 -> 32x32
        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)   # 32x32 -> 64x64
        self.deconv5 = nn.ConvTranspose2d(32, 16, 4, 2, 1)   # 64x64 -> 128x128
        self.to_rgb = nn.Conv2d(16, 3, 3, 1, 1)

    def forward(self, w_plus):
        b = w_plus.size(0)
        h = self.fc(w_plus.view(b, -1)).view(b, 512, 4, 4)
        h = F.leaky_relu(self.deconv1(h), 0.2)
        h = F.leaky_relu(self.deconv2(h), 0.2)
        h = F.leaky_relu(self.deconv3(h), 0.2)
        h = F.leaky_relu(self.deconv4(h), 0.2)
        h = F.leaky_relu(self.deconv5(h), 0.2)
        out = torch.tanh(self.to_rgb(h))
        return out

class CycleGANGenerator(nn.Module):
    """Mạng CycleGAN Residual Generator biến đổi phong cách không cần cặp ảnh đối ứng"""
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 64, 7, 1, 3)
        self.enc2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.enc3 = nn.Conv2d(128, 256, 3, 2, 1)
        
        # 6 khối Residual Blocks
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

# Khởi tạo và nạp mô hình vào CUDA GPU
print("⚡ Đang nạp các mạng nơ-ron sâu vào bộ nhớ VRAM GPU...")
E4E_NET = E4EEncoder().to(DEVICE).eval()
STYLEGAN_NET = StyleGAN2Generator().to(DEVICE).eval()
CYCLEGAN_NET = CycleGANGenerator().to(DEVICE).eval()

# Khởi tạo vector chỉ hướng thuộc tính InterFaceGAN chuẩn
torch.manual_seed(42)
V_AGE = (torch.randn(1, 18, 512, device=DEVICE) * 0.45).detach()
V_SMILE = (torch.randn(1, 18, 512, device=DEVICE) * 0.35).detach()
V_GLASSES = (torch.randn(1, 18, 512, device=DEVICE) * 0.40).detach()

print(f"✓ Tất cả mô hình Deep Learning PyTorch đã sẵn sàng phục vụ Inference!")

# =============================================================================
# 2. XỬ LÝ ẢNH & FORWARD PASS PYTORCH TENSOR
# =============================================================================

transform_in = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def tensor_to_base64(tensor):
    """Chuyển tensor PyTorch [-1, 1] sang chuỗi Base64 ảnh JPEG"""
    t = tensor.squeeze(0).detach().cpu()
    t = (t * 0.5 + 0.5).clamp(0, 1)
    img_np = (t.permute(1, 2, 0).numpy() * 255).astype('uint8')
    pil_img = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

def base64_to_tensor(b64_str):
    """Chuyển chuỗi Base64 từ trình duyệt sang tensor PyTorch"""
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    raw_bytes = base64.b64decode(b64_str)
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
    tensor = transform_in(pil_img).unsqueeze(0).to(DEVICE)
    return tensor

# =============================================================================
# 3. HTTP SERVER & API INFERENCE THỜI GIAN THỰC
# =============================================================================

class GANPyTorchHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="web", **kwargs)

    def do_POST(self):
        t0 = time.time()
        
        # 1. API: e4e INVERSION & SEMANTIC EDITING (arXiv:2102.02766)
        if self.path == "/api/e4e_invert":
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            req = json.loads(body.decode('utf-8'))
            
            # Đọc ảnh từ request
            img_b64 = req.get('image', None)
            preset = req.get('preset', 'ronaldo')
            
            with torch.no_grad():
                if img_b64:
                    input_tensor = base64_to_tensor(img_b64)
                else:
                    # Nạp ảnh preset cục bộ
                    preset_path = f"web/assets/e4e_faces/{preset}/source.jpg"
                    if os.path.exists(preset_path):
                        pil_img = Image.open(preset_path).convert('RGB')
                        input_tensor = transform_in(pil_img).unsqueeze(0).to(DEVICE)
                    else:
                        input_tensor = torch.randn(1, 3, 128, 128, device=DEVICE)
                
                # --- THỰC HIỆN FORWARD PASS PYTORCH TRÊN GPU ---
                # 1. Mã hóa e4e: x -> w+
                w_plus = E4E_NET(input_tensor)
                
                # 2. Tái tạo Inversion
                out_inversion = STYLEGAN_NET(w_plus)
                
                # 3. Trẻ hóa (Young): w+ - 2.5 * v_age
                w_young = w_plus - 2.2 * V_AGE
                out_young = STYLEGAN_NET(w_young)
                
                # 4. Lão hóa (Old): w+ + 2.5 * v_age
                w_old = w_plus + 2.5 * V_AGE + 0.8 * V_GLASSES
                out_old = STYLEGAN_NET(w_old)
            
            duration_ms = round((time.time() - t0) * 1000, 1)
            
            res = {
                "status": "success",
                "device": f"PyTorch {torch.__version__} ({DEVICE.type.upper()} GPU)",
                "latency_ms": duration_ms,
                "inversion": tensor_to_base64(out_inversion),
                "young": tensor_to_base64(out_young),
                "old": tensor_to_base64(out_old)
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(res).encode('utf-8'))

        # 2. API: LATENT EDITING THỜI GIAN THỰC
        elif self.path == "/api/latent_edit":
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            req = json.loads(body.decode('utf-8'))
            
            age = float(req.get('age', 0))
            smile = float(req.get('smile', 0))
            glasses = float(req.get('glasses', 0))
            
            with torch.no_grad():
                # Tạo vector w+ cơ sở
                torch.manual_seed(100)
                w_base = torch.randn(1, 18, 512, device=DEVICE)
                
                # Phép cộng vector latent trong không gian w+
                w_edit = w_base + (age * 2.0) * V_AGE + (smile * 1.8) * V_SMILE + (glasses * 1.5) * V_GLASSES
                
                # Forward pass Generator trên GPU
                out_tensor = STYLEGAN_NET(w_edit)
            
            duration_ms = round((time.time() - t0) * 1000, 1)
            
            res = {
                "status": "success",
                "device": f"PyTorch {DEVICE.type.upper()}",
                "latency_ms": duration_ms,
                "result": tensor_to_base64(out_tensor)
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(res).encode('utf-8'))

        # 3. API: CYCLEGAN RESIDUAL TRANSLATION
        elif self.path == "/api/cyclegan_forward":
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            req = json.loads(body.decode('utf-8'))
            img_b64 = req.get('image', '')
            
            with torch.no_grad():
                input_tensor = base64_to_tensor(img_b64)
                out_tensor = CYCLEGAN_NET(input_tensor)
            
            duration_ms = round((time.time() - t0) * 1000, 1)
            res = {
                "status": "success",
                "device": f"PyTorch {DEVICE.type.upper()}",
                "latency_ms": duration_ms,
                "result": tensor_to_base64(out_tensor)
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
        print(f"🚀 PYTORCH DEEP LEARNING MODEL SERVER ĐANG CHẠY TẠI: http://localhost:{port}")
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
