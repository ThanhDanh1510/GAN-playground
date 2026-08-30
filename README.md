# ⚡ GAN Playground Dành Cho Học Sinh (Student AI Lab)

> **Sân chơi trực quan & Bộ giáo trình thực hành về Mạng Đối Nghịch Tạo Sinh (GAN - Generative Adversarial Networks)** dành riêng cho học sinh, sinh viên và người mới bắt đầu học Trí Tuệ Nhân Tạo. Vận hành 100% bằng mô hình PyTorch Deep Learning trên GPU!

![License](https://img.shields.io/badge/License-MIT-emerald.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![CUDA GPU](https://img.shields.io/badge/CUDA_GPU-Accelerated-green.svg)
![Web Playground](https://img.shields.io/badge/Web_Playground-Real--Time-cyan.svg)
![Google Colab](https://img.shields.io/badge/Google_Colab-1--Click_Ready-yellow.svg)

---

## 🌟 3 Sân Chơi Cốt Lõi (Sổ Tay Google Colab 1-Click Ready)

Mỗi chủ đề là một tệp Jupyter Notebook độc lập, có sẵn nút **Open in Colab (GPU miễn phí)**:

| STT | Sân Chơi (Playground) | Chủ Đề & Điểm Nhấn Khoa Học | Sổ Tay Google Colab |
| :---: | :--- | :--- | :---: |
| **01** | **🦓 CycleGAN Transformer** | **Máy biến đổi phong cách không cần cặp ảnh**: Nhận diện thân chú ngựa trắng và vẽ các dải sọc vằn trực tiếp lên cơ bắp của chính chú ngựa đó với nguyên lý *Cycle Consistency Loss* ($A \to B \to A'$). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDanh1510/GAN-playground/blob/main/notebooks/03_cyclegan_world_transformer.ipynb) |
| **02** | **🎨 Pix2Pix Sketch-to-Art** | **Bút vẽ phù thủy**: Vẽ nét phác thảo (Doodle) $\to$ Mạng nơ-ron U-Net tự động phân tích đường viền và tạo bóng 3D, nguồn sáng và kết cấu vật liệu sống động! | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDanh1510/GAN-playground/blob/main/notebooks/02_pix2pix_sketch2art_playground.ipynb) |
| **03** | **⚡ 2D Point GAN Arena** | **Đấu trường Minimax thời gian thực**: Cuộc chiến giữa Kẻ làm giả ($G$) vs Cảnh sát ($D$); học phân phối **Hình Trái Tim 💖, Vòng Tròn ⭕, Xoắn Ốc 🌀** trong 35 dòng PyTorch! | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDanh1510/GAN-playground/blob/main/notebooks/01_2d_point_gan_playground.ipynb) |
| **Tổng Hợp** | **📚 Top 3 GAN Playground** | **Giáo trình trọn gói**: Toàn bộ lý thuyết trò chơi Minimax, cân bằng Nash, kiến trúc U-Net, Cycle Consistency + **10 Mẹo thực chiến (GAN Hacks)**. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDanh1510/GAN-playground/blob/main/notebooks/top3_gan_playground.ipynb) |

---

## 🚀 Cách Sử Dụng Nhanh

### 1. Chạy Web Playground Cục Bộ (Tăng Tốc PyTorch GPU)
Khởi động máy chủ backend PyTorch Inference Server:
```bash
python python/app_server.py --port 8080
```
Truy cập trình duyệt tại địa chỉ: **[http://localhost:8080](http://localhost:8080)** để trải nghiệm giao diện tương tác thời gian thực với card đồ họa GPU!

### 2. Chạy Thực Hành Trên Google Colab (Có GPU Miễn Phí)
Bấm trực tiếp vào các nút **Open in Colab** ở bảng trên.
*(Trên giao diện Colab, chọn menu **Runtime > Change runtime type > T4 GPU** để kích hoạt tăng tốc phần cứng!)*

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
GAN playground/
│
├── web/                                         # 🌐 Ứng dụng Web tương tác thời gian thực
│   ├── index.html                               # Giao diện chính chứa 3 Studio cốt lõi
│   ├── js/
│   │   ├── app_engine.js                        # Điều khiển CycleGAN, Pix2Pix & kết nối GPU API
│   │   ├── gan_engine.js                        # Mini Neural Net engine thuần JavaScript
│   │   ├── tfjs_gan_engine.js                   # TensorFlow.js GPU WebGL backend
│   │   └── visualizer.js                        # Trực quan hóa 2D Points, Heatmap & Loss Battle
│   └── assets/                                  # Tài nguyên ảnh chụp & dữ liệu mẫu
│
├── notebooks/                                   # 📓 Sổ tay Colab độc lập cho từng bài học
│   ├── 01_2d_point_gan_playground.ipynb         # Bài 1: 2D Point GAN Arena
│   ├── 02_pix2pix_sketch2art_playground.ipynb   # Bài 2: Pix2Pix Sketch-to-Art (U-Net)
│   ├── 03_cyclegan_world_transformer.ipynb      # Bài 3: CycleGAN Unpaired Translation
│   └── top3_gan_playground.ipynb               # Giáo trình tổng hợp trọn bộ + 10 GAN Hacks
│
├── python/                                      # 🐍 Máy chủ Deep Learning PyTorch GPU
│   └── app_server.py                            # REST API server thực hiện Forward Pass & Backprop
│
└── docs/                                        # 📚 Tài liệu học tập chuyên sâu
    ├── top3_gan_adventures.md                   # Hướng dẫn khám phá 3 sân chơi bằng hình vẽ & câu chuyện
    ├── notebook_code_deep_dive.md               # Giải thích chi tiết TẠI SAO từng cell code như vậy
    └── gan_hacks_for_students.md                # 10 mẹo thực chiến của Soumith Chintala
```

---

## 💡 Nguồn Mở & Tài Liệu Tham Khảo Quốc Tế
- **CycleGAN & Pix2Pix**: [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) (UC Berkeley)
- **GAN Lab**: [poloclub.github.io/ganlab](https://poloclub.github.io/ganlab/) (Google PAIR & Georgia Tech)
- **PyTorch-GAN**: [eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)
- **GAN Hacks**: [soumith/ganhacks](https://github.com/soumith/ganhacks) (Soumith Chintala)
