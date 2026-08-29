# 🛠️ 10 Mẹo Thực Chiến Huấn Luyện GAN Thành Công (GAN Hacks for Students)

*Được tổng hợp và đơn giản hóa từ cẩm nang "How to Train a GAN?" của Soumith Chintala (Đồng sáng lập PyTorch).*

---

### 1. Chuẩn hóa ảnh về khoảng $[-1, 1]$
- Lớp cuối cùng của Generator luôn dùng hàm kích hoạt `nn.Tanh()`.
- Dữ liệu ảnh đầu vào chia cho $127.5$ rồi trừ đi $1.0$.

### 2. Dùng `LeakyReLU` thay vì `ReLU`
- `ReLU` làm triệt tiêu các giá trị âm (trả về 0), khiến gradient bị "chết" (Dying ReLU).
- `LeakyReLU(0.2)` cho phép một phần gradient âm đi qua, giúp mạng tiếp tục học.

### 3. Lấy mẫu nhiễu từ Phân phối Chuẩn (Gaussian / Normal Distribution)
- Dùng `torch.randn()` thay vì `torch.rand()`. Phân phối hình chuông Gauss giúp các đặc trưng phân bổ tự nhiên hơn.

### 4. Kỹ thuật Làm Mềm Nhãn (Label Smoothing)
- Thay vì gán nhãn ảnh thật tuyệt đối bằng $1.0$, hãy đặt là $0.9$. Điều này ngăn Cảnh sát (Discriminator) trở nên quá tự tin và áp đảo Kẻ làm giả.

### 5. Dùng Optimizer Adam với $\beta_1 = 0.5$
- Thông số Adam mặc định thường là $\beta_1 = 0.9$, nhưng với GAN, đặt $\beta_1 = 0.5$ giúp giảm bớt quán tính và dao động trong cuộc chiến minimax.

### 6. Tránh lỗi Mode Collapse (Suy sụp mốt)
- **Hiện tượng**: Generator chỉ vẽ đúng 1 bức ảnh duy nhất lặp đi lặp lại vì phát hiện bức ảnh đó dễ lừa Discriminator nhất.
- **Cách trị**: Thêm nhiễu vào đầu vào của Discriminator, giảm tốc độ học của Cảnh sát, hoặc áp dụng hàm mất mát Wasserstein (WGAN).

### 7. U-Net Skip Connections cho bài toán sinh ảnh từ nét vẽ
- Khi làm bài toán Pix2Pix (từ nét vẽ sang tranh màu), luôn dùng kiến trúc U-Net để các đường biên sắc nét không bị nhòe mờ.

### 8. Vòng lặp Cycle Consistency Loss cho bài toán không có cặp ảnh
- Với CycleGAN, luôn đảm bảo $F(G(A)) \approx A$ để mô hình không tự do bịa ra các hình thù kỳ dị ngoài ý muốn.

### 9. Theo dõi biểu đồ Loss thường xuyên
- Trong GAN, $D$ Loss và $G$ Loss không nhất thiết phải giảm về 0. Trạng thái lý tưởng là cả 2 đường Loss dao động cân bằng quanh một giá trị ổn định (Cân bằng Nash - Nash Equilibrium).

### 10. Luôn tận dụng GPU Google Colab miễn phí
- Huấn luyện mạng nơ-ron tạo ảnh với GPU nhanh gấp 10-20 lần so với CPU thông thường!
