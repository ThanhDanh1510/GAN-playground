# 🔬 Giải Thích Chi Tiết Từng Dòng Code Trong Các Notebook Playground (Code Deep Dive)

Tài liệu này được biên soạn dành cho học sinh, sinh viên và giảng viên nhằm giải thích chi tiết **TẠI SAO từng ô code (cell) trong các Jupyter Notebook lại được thiết kế và lập trình chính xác như vậy**.

---

## 📑 Mục Lục
1. [Notebook 01: 2D Point GAN Arena (`01_2d_point_gan_playground.ipynb`)](#1-notebook-01-2d-point-gan-arena)
2. [Notebook 02: Pix2Pix Sketch-to-Art (`02_pix2pix_sketch2art_playground.ipynb`)](#2-notebook-02-pix2pix-sketch-to-art)
3. [Notebook 03: CycleGAN World Transformer (`03_cyclegan_world_transformer.ipynb`)](#3-notebook-03-cyclegan-world-transformer)
4. [Notebook Tổng Hợp: Top 3 GAN Playground (`top3_gan_playground.ipynb`)](#4-notebook-tổng-hợp-top-3-gan-playground)

---

# 1. Notebook 01: 2D Point GAN Arena

File: [`notebooks/01_2d_point_gan_playground.ipynb`](file:///c:/Users/PC/Downloads/GAN%20playground/notebooks/01_2d_point_gan_playground.ipynb)

### 🔹 Cell 1: Thiết Lập Môi Trường & Thiết Bị (Device Setup)
```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
* **Tại sao cần `torch.device`?**
  Mã nguồn tự động nhận diện nếu máy tính có card đồ họa NVIDIA (CUDA GPU) để đẩy các phép tính ma trận vào GPU VRAM (tốc độ nhanh hơn 15-30 lần), đồng thời vẫn chạy mượt mà trên CPU nếu không có GPU rời.

---

### 🔹 Cell 2: Định Nghĩa Kiến Trúc Mạng Nơ-ron (`Generator2D` & `Discriminator2D`)
```python
class Generator2D(nn.Module):
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
```
* **Tại sao `latent_dim = 2`?**
  Vì chúng ta đang trực quan hóa bài toán trên mặt phẳng 2D $(x, y)$. Vector nhiễu $z \in \mathbb{R}^2$ cho phép quan sát sự biến dạng không gian topo một cách trực quan nhất.
* **Tại sao dùng `LeakyReLU(0.2)` thay vì `ReLU` thông thường?**
  Hàm `ReLU` thông thường sẽ biến tất cả giá trị âm $x < 0$ thành $0$, khiến đạo hàm bị triệt tiêu ($\frac{\partial y}{\partial x} = 0$, hiện tượng *"Dying ReLU"*). Với `LeakyReLU(0.2)`, khi $x < 0$, giá trị trả về là $0.2x$, giúp gradient luôn chảy qua toàn bộ mạng để cập nhật tọa độ.
* **Tại sao lớp cuối của Generator KHÔNG dùng hàm kích hoạt nào (Linear)?**
  Vì tọa độ của các điểm trên mặt phẳng có thể mang cả giá trị âm và dương trong khoảng $[-1.5, 1.5]$. Không dùng `Sigmoid` hay `Tanh` ở đây để Generator tự do vẽ điểm ở bất kỳ tọa độ nào.

```python
class Discriminator2D(nn.Module):
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
```
* **Tại sao lớp cuối của Discriminator dùng `nn.Sigmoid()`?**
  Discriminator đóng vai trò là một bộ phân loại nhị phân (*Binary Classifier*). `Sigmoid` nén mọi giá trị về khoảng xác suất $[0, 1]$: $1.0$ là 100% tin rằng đây là điểm thật, $0.0$ là 100% tin rằng đây là điểm giả do Generator vẽ.

---

### 🔹 Cell 3: Hàm Sinh Phân Phối Dữ Liệu Mục Tiêu (Hình Trái Tim)
```python
def get_heart_distribution(batch_size=256):
    t = torch.rand(batch_size) * 2 * np.pi - np.pi
    x = 16 * (torch.sin(t) ** 3) / 18
    y = (13 * torch.cos(t) - 5 * torch.cos(2*t) - 2 * torch.cos(3*t) - torch.cos(4*t)) / 18
    noise = torch.randn(batch_size, 2) * 0.04
    return torch.stack([x, y], dim=1) + noise
```
* **Tại sao dùng phương trình tham số này?**
  Đây là phương trình toán học Cardioid chuẩn của đường cong hình trái tim.
* **Tại sao phải cộng thêm `noise = torch.randn(...) * 0.04`?**
  Nếu không có nhiễu Gaussian, dữ liệu thật sẽ là một đường cong vô cùng mỏng (độ đo Lebesgue bằng 0 trong $\mathbb{R}^2$). Điều này sẽ khiến Discriminator quá dễ dàng tạo ra một đường phân cách hoàn hảo, làm triệt tiêu gradient của Generator. Việc thêm một lượng nhiễu nhỏ $0.04$ tạo ra một dải phân phối liên tục (*Continuous Support*) giúp việc huấn luyện ổn định theo lý thuyết toán học của WGAN.

---

### 🔹 Cell 4: Vòng Lặp Huấn Luyện Minimax & Các Kỹ Thuật Tối Ưu
```python
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
```
* **Tại sao đặt `betas=(0.5, 0.999)` thay vì mặc định `(0.9, 0.999)`?**
  Hệ số $\beta_1 = 0.5$ là một trong những mẹo kinh điển từ bài báo *DCGAN (Radford et al.)*. Trong GAN, 2 mạng đối đầu nhau khiến mục tiêu di chuyển liên tục (*Non-stationary Objective*). Giảm $\beta_1$ từ $0.9$ xuống $0.5$ làm giảm quán tính (momentum) của bước trước, ngăn hiện tượng dao động quá mức (*Overshooting*) quanh điểm cân bằng Nash.

```python
# --- BƯỚC 1: HUẤN LUYỆN DISCRIMINATOR ---
loss_real = criterion(D(real_pts), torch.ones(batch_size, 1, device=device) * 0.9) # Label smoothing
loss_fake = criterion(D(fake_pts.detach()), torch.zeros(batch_size, 1, device=device))
d_loss = (loss_real + loss_fake) / 2
opt_D.zero_grad(); d_loss.backward(); opt_D.step()
```
* **Tại sao lại dùng `torch.ones(...) * 0.9` (One-sided Label Smoothing)?**
  Thay vì gán nhãn ảnh thật là $1.0$, gán nhãn $0.9$ ngăn hàm mất mát BCE tiến tới cực trị vô cùng, giữ cho gradient của Discriminator luôn mềm mại và có ích cho Generator học tập.
* **Tại sao BẮT BUỘC phải dùng `fake_pts.detach()` khi cập nhật $D$?**
  `fake_pts` được sinh ra bởi $G$. Khi huấn luyện $D$, ta chỉ muốn tính đạo hàm theo các trọng số của $D$. Dùng `.detach()` để cắt đứt đồ thị tính toán (*Computational Graph*), ngăn PyTorch lãng phí bộ nhớ và thời gian tính toán đạo hàm ngược vào mạng $G$.

```python
# --- BƯỚC 2: HUẤN LUYỆN GENERATOR ---
g_loss = criterion(D(fake_pts), torch.ones(batch_size, 1, device=device))
opt_G.zero_grad(); g_loss.backward(); opt_G.step()
```
* **Tại sao gán nhãn của `fake_pts` là `1.0` (torch.ones) khi train $G$?**
  Mục tiêu của Generator là muốn Cảnh sát tin rằng ảnh của nó là thật ($D(G(z)) \to 1$). Vì vậy, hàm mất mát của $G$ là độ lệch giữa phán quyết của Cảnh sát và nhãn $1.0$.

---

# 2. Notebook 02: Pix2Pix Sketch-to-Art

File: [`notebooks/02_pix2pix_sketch2art_playground.ipynb`](file:///c:/Users/PC/Downloads/GAN%20playground/notebooks/02_pix2pix_sketch2art_playground.ipynb)

### 🔹 Cell 2: Kiến Trúc `MiniUNetGenerator` với Skip Connections
```python
class MiniUNetGenerator(nn.Module):
    def __init__(self, in_c=1, out_c=3):
        super().__init__()
        # Encoder
        self.e1 = nn.Conv2d(in_c, 32, 4, 2, 1) # 64x64 -> 32x32
        self.e2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.e3 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        
        # Decoder
        self.d1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.d2 = nn.Sequential(nn.ConvTranspose2d(64 + 64, 32, 4, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.d3 = nn.Sequential(nn.ConvTranspose2d(32 + 32, out_c, 4, 2, 1), nn.Tanh())
        
    def forward(self, x):
        e1 = nn.LeakyReLU(0.2)(self.e1(x))
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        d1 = self.d1(e3)
        d2 = self.d2(torch.cat([d1, e2], dim=1)) # Skip Connection: 64 + 64 = 128 kênh
        out = self.d3(torch.cat([d2, e1], dim=1)) # Skip Connection: 32 + 32 = 64 kênh
        return out
```
* **Tại sao dùng `Conv2d(stride=2)` thay vì `MaxPool2d`?**
  Phép tích chập bước nhảy (*Strided Convolution*) cho phép mạng **tự học** cách giảm kích thước ảnh một cách tối ưu thay vì dùng phép lấy giá trị lớn nhất cố định (*Max Pooling*) làm mất thông tin vị trí không gian.
* **Bản chất của `torch.cat([d1, e2], dim=1)` (Skip Connection) là gì?**
  Trong quá trình nén ảnh xuống $e_3$, thông tin ngữ nghĩa trừu tượng được giữ lại nhưng tọa độ pixel của nét vẽ bị co cụm lại. Bằng cách nối tắt $e_2$ sang $d_2$ và $e_1$ sang $d_3$, các kênh đặc trưng đường viền độ phân giải cao được sao chép trực tiếp sang Decoder. Kết quả là ảnh sinh ra không bao giờ bị nhòe nét viền!
* **Tại sao lớp cuối dùng `nn.Tanh()`?**
  Hàm `Tanh` giới hạn giá trị đầu ra chính xác trong khoảng $[-1, 1]$. Toàn bộ dữ liệu ảnh màu RGB đều được chuẩn hóa về $[-1, 1]$ trước khi đưa vào huấn luyện.

---

### 🔹 Cell 2 (tiếp): Kiến Trúc `PatchGANDiscriminator`
```python
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
```
* **Tại sao đầu vào của Discriminator là `4 kênh`?**
  Pix2Pix là mạng GAN có điều kiện (*Conditional GAN - cGAN*). Discriminator không chỉ nhìn vào bức ảnh màu (3 kênh) mà phải ghép nối (*concatenate*) với nét vẽ phác thảo đầu vào (1 kênh) để đánh giá xem: *"Bức tranh này có thực sự được vẽ đúng theo nét phác thảo hay không?"*.
* **Tại sao gọi là PatchGAN?**
  Lớp cuối cùng không dùng `Linear` để trả về 1 số thực duy nhất, mà trả về một ma trận $M \times M$ (mỗi ô đại diện cho một vùng $N \times N$ pixel trên ảnh gốc). Điều này ép Discriminator tập trung soi xét độ chân thực ở từng góc nhỏ của bức tranh.

---

### 🔹 Cell 4: Hàm Mất Mát Kết Hợp ($\mathcal{L}_{GAN} + 100 \times \mathcal{L}_{L1}$)
```python
pred_fake_g = D_pix(sketches, fake_colors)
loss_gan = bce_loss(pred_fake_g, torch.ones_like(pred_fake_g))
loss_l1 = l1_loss(fake_colors, real_colors) * lambda_l1
loss_G = loss_gan + loss_l1
```
* **Tại sao phải có $L1\ Loss$ với trọng số lớn $\lambda = 100$?**
  - Nếu chỉ dùng $\mathcal{L}_{GAN}$, Generator có thể vẽ ra một bức tranh rất đẹp nhưng hoàn toàn sai màu sắc so với ảnh mẫu.
  - Nếu chỉ dùng $\mathcal{L}_{L1}$, ảnh sinh ra sẽ bị mờ nhòe (do L1 tính trung bình các khả năng màu sắc có thể xảy ra).
  - Kết hợp cả hai: $\mathcal{L}_{L1}$ đảm bảo bố cục và màu sắc chính xác, trong khi $\mathcal{L}_{GAN}$ đảm bảo độ tương phản 3D sắc nét và không bị mờ!

---

# 3. Notebook 03: CycleGAN World Transformer

File: [`notebooks/03_cyclegan_world_transformer.ipynb`](file:///c:/Users/PC/Downloads/GAN%20playground/notebooks/03_cyclegan_world_transformer.ipynb)

### 🔹 Cell 2: Khối `ResidualBlock` & `InstanceNorm2d`
```python
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
```
* **Tại sao CycleGAN dùng `InstanceNorm2d` thay vì `BatchNorm2d`?**
  `BatchNorm` tính giá trị trung bình trên toàn bộ mini-batch, vô tình trộn lẫn đặc trưng phong cách của các ảnh khác nhau. Trong bài toán chuyển đổi phong cách (*Style Transfer*), mỗi bức ảnh có ánh sáng và màu sắc riêng biệt. `InstanceNorm` chuẩn hóa độc lập trên từng bức ảnh, giúp giữ nguyên phong cách riêng của từng mẫu dữ liệu.
* **Tại sao kiến trúc Residual $x + \text{block}(x)$ là chìa khóa của CycleGAN?**
  Phép cộng $x + F(x)$ cho phép mạng nơ-ron thiết lập trạng thái mặc định là giữ nguyên ảnh gốc ($F(x) = 0 \implies \text{output} = x$). Khi học biến đổi, mạng chỉ cần học thêm phần thay đổi hoa văn (như vẽ thêm sọc vằn hoặc phủ thêm tuyết) mà không phá hủy cấu trúc hình thể ban đầu của chú ngựa.

---

### 🔹 Cell 4: Vòng Lặp Chu Trình Khép Kín (Cycle Consistency & Identity Loss)
```python
# 1. Cycle Consistency Loss: A -> B -> A' và B -> A -> B'
rec_A = F_BA(fake_B); loss_cycle_A = criterion_cycle(rec_A, real_A) * lambda_cycle
rec_B = G_AB(fake_A); loss_cycle_B = criterion_cycle(rec_B, real_B) * lambda_cycle

# 2. Identity Loss: F_BA(A) -> A và G_AB(B) -> B
loss_id_A = criterion_id(F_BA(real_A), real_A) * lambda_id
loss_id_B = criterion_id(G_AB(real_B), real_B) * lambda_id
```
* **Tại sao cần Cycle Consistency Loss?**
  Nếu không có hàm này, máy tạo $G_{AB}$ có thể biến mọi chú ngựa thành cùng một bức ảnh ngựa vằn ngẫu nhiên (hiện tượng *Mode Collapse*). Bằng cách bắt buộc $F_{BA}(G_{AB}(A)) \approx A$, thông tin hình dáng của chú ngựa gốc $A$ bắt buộc phải được bảo toàn trong ảnh trung gian $B'$.
* **Tại sao cần Identity Loss ($L_{id}$)?**
  Nếu đưa một bức ảnh đã là ngựa vằn vào máy tạo $G_{AB}$ (vốn có nhiệm vụ tạo ngựa vằn), nó không được phép thay đổi màu sắc của bức ảnh đó. Identity Loss ngăn hiện tượng đảo ngược hệ màu (như biến màu trời xanh thành màu vàng) trong quá trình huấn luyện.

---

# 4. Notebook Tổng Hợp: Top 3 GAN Playground

File: [`notebooks/top3_gan_playground.ipynb`](file:///c:/Users/PC/Downloads/GAN%20playground/notebooks/top3_gan_playground.ipynb)

* Đây là giáo trình thực hành toàn diện tích hợp 3 bài học trên vào một file duy nhất, kèm theo 10 Mẹo thực chiến (GAN Hacks) của các nhà nghiên cứu hàng đầu thế giới.
* Mọi cell đều được tối ưu hóa để chạy trong dưới 2 phút trên Google Colab GPU miễn phí, giúp học sinh có thể vừa học lý thuyết vừa tận mắt nhìn thấy kết quả biến đổi ngay lập tức!
