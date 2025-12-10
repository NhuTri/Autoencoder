# 📊 Hướng dẫn Trực quan hóa Quá trình Training

## Tổng quan

Sau khi training, chương trình sẽ tự động xuất file CSV chứa lịch sử training. Bạn có thể sử dụng script Python `visualize_training.py` để vẽ biểu đồ.

## File CSV được tạo tự động

| Phiên bản | File CSV | Nội dung |
|-----------|----------|----------|
| CPU | `training_history_cpu.csv` | epoch, train_loss, test_loss, time_sec, is_best |
| GPU | `training_history_gpu.csv` | epoch, train_loss, test_loss, time_sec, is_best |

## Cài đặt thư viện Python

```bash
pip install matplotlib pandas numpy
```

## Cách sử dụng

### 1. Tự động detect (khuyên dùng)

```bash
python visualize_training.py
```

Script sẽ tự động:
- Nếu chỉ có CPU CSV → vẽ biểu đồ CPU
- Nếu chỉ có GPU CSV → vẽ biểu đồ GPU  
- Nếu có cả 2 → vẽ biểu đồ so sánh

### 2. Chỉ định mode cụ thể

```bash
# Chỉ vẽ CPU
python visualize_training.py cpu

# Chỉ vẽ GPU
python visualize_training.py gpu

# So sánh CPU vs GPU
python visualize_training.py compare

# File CSV tùy chỉnh
python visualize_training.py path/to/custom.csv
```

## Biểu đồ được tạo

### Mode đơn (CPU hoặc GPU)

| Biểu đồ | Mô tả |
|---------|-------|
| **Loss Curve** | Train/Test loss qua từng epoch, đánh dấu epoch tốt nhất |
| **Time per Epoch** | Thời gian training mỗi epoch (bar chart) |

### Mode so sánh (Compare)

| Biểu đồ | Mô tả |
|---------|-------|
| **Loss Comparison** | So sánh train/test loss giữa CPU và GPU |
| **Time Comparison** | So sánh thời gian mỗi epoch |
| **Speedup** | Tỷ lệ tăng tốc GPU so với CPU |
| **Cumulative Time** | Tổng thời gian tích lũy, hiển thị thời gian tiết kiệm |

## Output files

```
training_plot_cpu.png       # Biểu đồ CPU
training_plot_gpu.png       # Biểu đồ GPU
training_comparison.png     # Biểu đồ so sánh
```

## Sử dụng trên Google Colab

```python
# Cell 1: Cài thư viện
!pip install matplotlib pandas numpy

# Cell 2: Sau khi train xong, vẽ biểu đồ
!python visualize_training.py

# Cell 3: Hiển thị ảnh trong notebook
from IPython.display import Image, display

# Hiển thị biểu đồ (chọn file phù hợp)
display(Image('training_plot_cpu.png'))
# hoặc
display(Image('training_plot_gpu.png'))
# hoặc
display(Image('training_comparison.png'))
```

## Ví dụ Output Console

```
==================================================
SUMMARY - CPU Autoencoder
==================================================
Total Epochs:      10
Final Train Loss:  0.012345
Final Test Loss:   0.015678
Best Test Loss:    0.014567 (Epoch 8)
Total Time:        1234.56 seconds
Average Time:      123.46 seconds/epoch
==================================================
```

## So sánh CPU vs GPU (Console Output)

```
============================================================
COMPARISON SUMMARY: CPU vs GPU
============================================================
Metric                            CPU             GPU
------------------------------------------------------------
Final Train Loss               0.012345        0.012389
Final Test Loss                0.015678        0.015701
Best Test Loss                 0.014567        0.014623
Total Time (s)                 1234.56          234.56
Avg Time/Epoch (s)              123.46           23.46
------------------------------------------------------------
Average Speedup                               5.26x
Max Speedup                                   6.12x
Min Speedup                                   4.89x
Time Saved                                 1000.00 seconds
============================================================
```

## Lưu ý

1. **Chạy training trước** - File CSV chỉ được tạo sau khi training hoàn thành
2. **Cùng số epoch** - Để so sánh chính xác, CPU và GPU nên train cùng số epoch
3. **Cùng dữ liệu** - Sử dụng cùng max_samples để so sánh công bằng

## Ví dụ workflow đầy đủ

```bash
# 1. Train CPU version
./autoencoder_cpu ./cifar-10-batches-bin 10 32 0 adam

# 2. Train GPU version (trên máy có CUDA)
./autoencoder_gpu ./cifar-10-batches-bin 10 32 0 adam

# 3. Copy file CSV về cùng thư mục (nếu cần)
# 4. Vẽ biểu đồ so sánh
python visualize_training.py compare
```

---

# 🖼️ Hướng dẫn Trực quan hóa Ảnh Reconstructed

## Tổng quan

Sau khi training, chương trình sẽ tự động export một số ảnh test và ảnh reconstructed ra file binary. Bạn có thể sử dụng script Python `visualize_reconstruction.py` để so sánh **Original vs Reconstructed**.

## File Binary được tạo tự động

| Phiên bản | File Binary | Nội dung |
|-----------|-------------|----------|
| CPU | `reconstructed_images_cpu.bin` | 10 ảnh test (original + reconstructed + labels + MSE) |
| GPU | `reconstructed_images_gpu.bin` | 10 ảnh test (original + reconstructed + labels + MSE) |

## Cài đặt thư viện Python

```bash
pip install matplotlib numpy
```

## Cách sử dụng

### 1. Tự động detect (khuyên dùng)

```bash
python visualize_reconstruction.py
```

Script sẽ tự động:
- Nếu có CPU file → vẽ ảnh CPU reconstruction
- Nếu có GPU file → vẽ ảnh GPU reconstruction

### 2. Chỉ định file cụ thể

```bash
# File CPU
python visualize_reconstruction.py reconstructed_images_cpu.bin

# File GPU  
python visualize_reconstruction.py reconstructed_images_gpu.bin

# File tùy chỉnh
python visualize_reconstruction.py path/to/custom.bin
```

### 3. So sánh CPU vs GPU

```bash
python visualize_reconstruction.py --compare
# hoặc
python visualize_reconstruction.py -c
```

## Biểu đồ được tạo

### Biểu đồ chính (3 hàng)

| Hàng | Nội dung |
|------|----------|
| **Original** | Ảnh gốc từ CIFAR-10 test set với label |
| **Reconstructed** | Ảnh được tái tạo qua Autoencoder với MSE |
| **Difference (5x)** | Sự khác biệt giữa 2 ảnh (phóng đại 5 lần) |

### Biểu đồ chi tiết (4 cột)

| Cột | Nội dung |
|-----|----------|
| **Original** | Ảnh gốc |
| **Reconstructed** | Ảnh reconstructed |
| **Difference (3x)** | Sự khác biệt màu (phóng đại 3 lần) |
| **Error Heatmap** | Bản đồ nhiệt hiển thị vùng có lỗi cao |

## Output files

```
reconstruction_cpu.png              # So sánh Original vs Reconstructed (CPU)
reconstruction_cpu_detailed.png     # Phân tích chi tiết với heatmap (CPU)
reconstruction_gpu.png              # So sánh Original vs Reconstructed (GPU)
reconstruction_gpu_detailed.png     # Phân tích chi tiết với heatmap (GPU)
reconstruction_comparison.png       # So sánh CPU vs GPU reconstruction
```

## Sử dụng trên Google Colab

```python
# Cell 1: Cài thư viện
!pip install matplotlib numpy

# Cell 2: Sau khi train xong, vẽ biểu đồ reconstruction
!python visualize_reconstruction.py

# Cell 3: Hiển thị ảnh trong notebook
from IPython.display import Image, display

# Hiển thị biểu đồ reconstruction
display(Image('reconstruction_cpu.png'))
display(Image('reconstruction_cpu_detailed.png'))

# Hoặc so sánh CPU vs GPU
# display(Image('reconstruction_comparison.png'))
```

## Ví dụ Output Console

```
Loading 10 images (32x32x3)

==================================================
RECONSTRUCTION STATISTICS - CPU Autoencoder
==================================================
Number of samples: 10
Image size: 32x32x3
Mean MSE: 0.012345
Min MSE:  0.008234
Max MSE:  0.018567
Std MSE:  0.003210
==================================================

Plot saved to: reconstruction_cpu.png
Detailed plot saved to: reconstruction_cpu_detailed.png
```

## Giải thích kết quả

### MSE (Mean Squared Error)
- **MSE thấp** (< 0.01): Reconstruction rất tốt, ảnh gần như giống hệt
- **MSE trung bình** (0.01 - 0.05): Reconstruction tốt, một số chi tiết nhỏ bị mất
- **MSE cao** (> 0.05): Reconstruction kém, nhiều thông tin bị mất

### Difference Image
- **Màu đen**: Không có sự khác biệt
- **Màu sáng**: Có sự khác biệt (càng sáng = khác biệt càng lớn)
- Thường thấy khác biệt ở các **edge** và **chi tiết nhỏ**

### Error Heatmap
- **Màu đỏ/vàng**: Vùng có lỗi cao (reconstruction kém)
- **Màu đen/tối**: Vùng có lỗi thấp (reconstruction tốt)

## Workflow đầy đủ

```bash
# 1. Train model
./autoencoder_cpu ./cifar-10-batches-bin 5 32 500 adam

# File được tạo tự động:
# - training_history_cpu.csv (training history)
# - reconstructed_images_cpu.bin (reconstruction samples)
# - autoencoder_weights.bin (model weights)
# - autoencoder_best.bin (best model weights)

# 2. Visualize training progress
python visualize_training.py

# 3. Visualize reconstruction quality
python visualize_reconstruction.py

# 4. (Optional) So sánh CPU vs GPU
# Sau khi có cả 2 file reconstruction
python visualize_reconstruction.py --compare
```

## Lưu ý

1. **Chạy training trước** - File binary chỉ được tạo sau khi training hoàn thành
2. **Ảnh test** - Script sử dụng 10 ảnh đầu tiên từ test set
3. **Clip values** - Pixel values được clip về [0, 1] để hiển thị đúng
4. **Loss tương quan MSE** - MSE của từng ảnh tương quan với overall loss
