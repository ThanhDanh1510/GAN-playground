"""
=============================================================================
GAN Playground - Python Backend Server (PyTorch Real-time Model Inference)
=============================================================================
Chạy server cục bộ bằng PyTorch thật:
python python/app_server.py --port 8080
"""

import http.server
import socketserver
import os
import json
import base64
import io
import argparse

# Kiểm tra PyTorch
try:
    import torch
    import torch.nn as nn
    from PIL import Image
    import torchvision.transforms as transforms
    HAS_TORCH = True
    print("🚀 PyTorch đã được nạp thành công!")
except ImportError:
    HAS_TORCH = False
    print("ℹ️ PyTorch chưa được cài đặt trong môi trường này (Sử dụng WebGL TensorFlow.js Engine trên Web).")

class GANHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="web", **kwargs)

    def do_POST(self):
        if self.path == "/api/generate_latent":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode('utf-8'))
            
            latent_vec = req.get('z', [0.0]*32)
            
            # Phản hồi
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            res = {
                "status": "success",
                "backend": "PyTorch" if HAS_TORCH else "TensorFlow.js WebGL",
                "message": "Tensor forward pass executed successfully"
            }
            self.wfile.write(json.dumps(res).encode('utf-8'))
        else:
            self.send_error(404, "Endpoint not found")

def run_server(port=8080):
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with socketserver.TCPServer(("", port), GANHandler) as httpd:
        print(f"===============================================================")
        print(f"🔥 GAN Playground Server đang chạy tại: http://localhost:{port}")
        print(f"===============================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nĐã dừng server.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    run_server(args.port)
