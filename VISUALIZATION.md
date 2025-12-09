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
