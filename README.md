# ⚡ GAN Playground Dành Cho Học Sinh (Student AI Lab)

> **Sân chơi trực quan & Bộ giáo trình thực hành về Mạng Đối Nghịch Tạo Sinh (GAN - Generative Adversarial Networks)** dành riêng cho học sinh và người mới bắt đầu học Trí tuệ Nhân tạo.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDanh1510/GAN-playground/blob/main/notebooks/top3_gan_playground.ipynb)
![License](https://img.shields.io/badge/License-MIT-emerald.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Web Playground](https://img.shields.io/badge/Web_Playground-Zero--Install-cyan.svg)

---

## 🌟 Top 3 Ứng Dụng GAN Kỳ Diệu Được Tích Hợp

| Ứng dụng | Điểm thú vị & Trải nghiệm thực tế | Nguyên lý cốt lõi |
| :--- | :--- | :--- |
| **🦓 1. CycleGAN** | Biến đổi phong cảnh: **Ngựa thường ↔ Ngựa vằn**, **Mùa hè ↔ Mùa đông tuyết phủ**, **Ảnh chụp ↔ Tranh sơn dầu Van Gogh** mà không cần 2 bức ảnh ghép cặp trước! | *Cycle Consistency Loss* ($A \to B \to A' \approx A$) |
| **🎨 2. Pix2Pix** | **Bút vẽ phù thủy**: Học sinh vẽ vài nét phác thảo (giày sneaker, mèo con, ngôi nhà) $\to$ AI tự động tô màu, tạo khối da/lông/kính chân thực! | *U-Net Skip Connections* & *PatchGAN Discriminator* |
| **😎 3. StyleGAN Latent Studio** | **Đại số vector mặt người**: $\text{Mặt cười có kính} = \text{Mặt thường} + \vec{v}_{cười} + \vec{v}_{kính}$; Kéo thanh trượt để biến hình (Morphing) mượt mà giữa 2 người! | *Latent Space Arithmetic* & *Linear Interpolation* |

---

## 🚀 Cách Sử Dụng Nhanh

### 1. Chạy Web Playground Tương Tác (Không cần cài đặt!)
Chỉ cần mở tệp `web/index.html` bằng bất kỳ trình duyệt web nào (Chrome, Edge, Firefox, Safari):
- **Trải nghiệm cả 5 Studio**: CycleGAN World Transformer, Pix2Pix Sketchpad, StyleGAN Face Studio, 2D Point GAN Arena (chạy neural net thời gian thực), và Mini-game Thám tử AI.

### 2. Chạy Thực Hành Code Trên Google Colab (Có GPU Miễn Phí)
Bấm trực tiếp vào nút bên dưới để mở sổ tay Notebook trên Google Colab:

👉 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDanh1510/GAN-playground/blob/main/notebooks/top3_gan_playground.ipynb)

*(Trên Colab, chọn menu **Runtime > Change runtime type > T4 GPU** để tăng tốc độ huấn luyện!)*

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
GAN playground/
│
├── web/                                 # 🌐 Ứng dụng Web tương tác (Zero-install)
│   ├── index.html                       # Giao diện chính chứa 5 Studio
│   ├── js/
│   │   ├── app_engine.js                # Xử lý CycleGAN, Pix2Pix, StyleGAN, Turing Game
│   │   ├── gan_engine.js                # Mini Neural Net engine thuần JavaScript
│   │   └── visualizer.js                # Trực quan hóa 2D Points, Heatmap & Loss Battle
│   └── assets/                          # Bộ ảnh vector SVG mẫu đẹp mắt
│
├── python/                              # 🐍 Bộ mã nguồn PyTorch mẫu chuẩn mực
│   ├── 01_toy_2d_gan.py                 # 35 dòng PyTorch học phân phối hình trái tim
│   ├── 02_pix2pix_sketch2art.py         # Pix2Pix U-Net biến nét vẽ thành tranh
│   ├── 03_cyclegan_transformer.py       # CycleGAN chuyển đổi phong cách không cần cặp ảnh
│   └── 04_latent_face_editor.py         # Đại số vector và biến hình không gian tiềm ẩn
│
├── notebooks/                           # 📓 Sổ tay tương tác Jupyter
│   └── top3_gan_playground.ipynb        # Giáo trình đầy đủ có nút bấm 1-Click Colab
│
└── docs/                                # 📚 Sổ tay kiến thức dành cho học sinh
    ├── top3_gan_adventures.md           # Giải thích 3 ứng dụng bằng câu chuyện & hình vẽ
    └── gan_hacks_for_students.md        # 10 mẹo thực chiến của Soumith Chintala
```

---

## 💡 Nguồn Mở & Tài Liệu Tham Khảo Quốc Tế
- **GAN Lab**: [poloclub.github.io/ganlab](https://poloclub.github.io/ganlab/) (Google PAIR & Georgia Tech)
- **PyTorch-GAN**: [eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)
- **CycleGAN & Pix2Pix**: [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) (UC Berkeley)
- **GAN Hacks**: [soumith/ganhacks](https://github.com/soumith/ganhacks) (Soumith Chintala)
