# ⚡ GAN Playground Dành Cho Học Sinh (Student AI Lab)

> **Sân chơi trực quan & Bộ giáo trình thực hành về Mạng Đối Nghịch Tạo Sinh (GAN - Generative Adversarial Networks)** dành riêng cho học sinh và người mới bắt đầu học Trí tuệ Nhân tạo.

![License](https://img.shields.io/badge/License-MIT-emerald.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Web Playground](https://img.shields.io/badge/Web_Playground-Zero--Install-cyan.svg)

---

## 🌟 4 Sân Chơi Notebook Độc Lập Chạy 1-Click Trên Google Colab

Mỗi chủ đề là một tệp Jupyter Notebook riêng biệt, có sẵn nút **Open in Colab (GPU miễn phí)** và chạy độc lập:

| Playground | Chủ đề & Điểm thú vị | Sổ tay Google Colab |
| :--- | :--- | :--- |
| **⚡ 1. 2D Point GAN Arena** | Cuộc đấu trực tiếp giữa Kẻ làm giả vs Cảnh sát; học phân phối **Hình Trái Tim 💖, Vòng Tròn ⭕, Xoắn Ốc 🌀** trong 35 dòng PyTorch! | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDanh1510/GAN-playground/blob/main/notebooks/01_2d_point_gan_playground.ipynb) |
| **🎨 2. Pix2Pix Sketch-to-Art** | **Bút vẽ phù thủy**: Vẽ nét phác thảo (Doodle) $\to$ U-Net AI tự động tô màu và đổ bóng chân thực! | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDanh1510/GAN-playground/blob/main/notebooks/02_pix2pix_sketch2art_playground.ipynb) |
| **🦓 3. CycleGAN Transformer** | **Máy biến đổi thế giới không cần ghép đôi**: *Ngựa $\leftrightarrow$ Ngựa vằn*, *Mùa hè $\leftrightarrow$ Mùa đông*, với nguyên lý *Cycle Consistency Loss* ($A \to B \to A'$). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDanh1510/GAN-playground/blob/main/notebooks/03_cyclegan_world_transformer.ipynb) |
| **😎 4. StyleGAN Latent Studio** | **Đại số vector mặt người**: $\text{Mặt cười có kính} = \text{Mặt gốc} + \vec{v}_{cười} + \vec{v}_{kính}$; Kéo slider biến hình (Morphing) mượt mà giữa 2 người! | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDanh1510/GAN-playground/blob/main/notebooks/04_stylegan_latent_studio.ipynb) |

---

## 🚀 Cách Sử Dụng Nhanh

### 1. Chạy Web Playground Tương Tác (Không cần cài đặt!)
Chỉ cần mở tệp [web/index.html](file:///c:/Users/PC/Downloads/GAN%20playground/web/index.html) bằng bất kỳ trình duyệt web nào (Chrome, Edge, Firefox, Safari):
- **Trải nghiệm cả 5 Studio**: CycleGAN World Transformer, Pix2Pix Sketchpad, StyleGAN Face Studio, 2D Point GAN Arena (chạy neural net thời gian thực), và Mini-game Thám tử AI.

### 2. Chạy Thực Hành Code Trên Google Colab (Có GPU Miễn Phí)
Bấm trực tiếp vào các nút **Open in Colab** ở bảng trên để mở bất kỳ bài học nào bạn muốn!

*(Trên Colab, chọn menu **Runtime > Change runtime type > T4 GPU** để tăng tốc độ huấn luyện!)*

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
GAN playground/
│
├── web/                                         # 🌐 Ứng dụng Web tương tác (Zero-install)
│   ├── index.html                               # Giao diện chính chứa 5 Studio
│   ├── js/
│   │   ├── app_engine.js                        # Xử lý CycleGAN, Pix2Pix, StyleGAN, Turing Game
│   │   ├── gan_engine.js                        # Mini Neural Net engine thuần JavaScript
│   │   └── visualizer.js                        # Trực quan hóa 2D Points, Heatmap & Loss Battle
│   └── assets/                                  # Bộ ảnh vector SVG mẫu đẹp mắt
│
├── notebooks/                                   # 📓 4 Sổ tay Colab độc lập cho từng Playground
│   ├── 01_2d_point_gan_playground.ipynb         # Playground 1: 2D Point GAN
│   ├── 02_pix2pix_sketch2art_playground.ipynb   # Playground 2: Pix2Pix Sketch-to-Art
│   ├── 03_cyclegan_world_transformer.ipynb      # Playground 3: CycleGAN
│   └── 04_stylegan_latent_studio.ipynb          # Playground 4: StyleGAN Latent Studio
│
├── python/                                      # 🐍 Bộ mã nguồn PyTorch mẫu chuẩn mực
│   ├── 01_toy_2d_gan.py                         # 35 dòng PyTorch học phân phối hình trái tim
│   ├── 02_pix2pix_sketch2art.py                 # Pix2Pix U-Net biến nét vẽ thành tranh
│   ├── 03_cyclegan_transformer.py               # CycleGAN chuyển đổi phong cách không cần cặp ảnh
│   └── 04_latent_face_editor.py                 # Đại số vector và biến hình không gian tiềm ẩn
│
└── docs/                                        # 📚 Sổ tay kiến thức dành cho học sinh
    ├── top3_gan_adventures.md                   # Giải thích 3 ứng dụng bằng câu chuyện & hình vẽ
    └── gan_hacks_for_students.md                # 10 mẹo thực chiến của Soumith Chintala
```

---

## 💡 Nguồn Mở & Tài Liệu Tham Khảo Quốc Tế
- **GAN Lab**: [poloclub.github.io/ganlab](https://poloclub.github.io/ganlab/) (Google PAIR & Georgia Tech)
- **PyTorch-GAN**: [eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)
- **CycleGAN & Pix2Pix**: [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) (UC Berkeley)
- **GAN Hacks**: [soumith/ganhacks](https://github.com/soumith/ganhacks) (Soumith Chintala)
