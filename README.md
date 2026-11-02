# Dự báo nhu cầu bán lẻ theo ngày có xét yếu tố lịch: LightGBM và LSTM trên dữ liệu M5

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Punpunpuong22/m5-calendar-forecasting/blob/main/notebooks/m5_calendar_aware_lightgbm_lstm_v4.ipynb)

Mã nguồn cho bài báo cáo môn **Học sâu và Ứng dụng trong kinh doanh** (GV: TS. Tạ Hoàng Thắng; học viên: Nguyễn Vũ Mai Phương – 26848176), một phần của luận văn thạc sĩ về tác động của yếu tố lịch và sự kiện trong dự báo nhu cầu bán lẻ.

**Câu hỏi nghiên cứu:** thêm thông tin lịch và sự kiện vào đầu vào làm dự báo nhu cầu ngày tốt hơn bao nhiêu; mô hình cây tăng cường (LightGBM) hay mạng hồi quy (LSTM) tận dụng thông tin này tốt hơn; lợi ích rơi vào loại ngày nào; và thời gian huấn luyện dài hơn có đồng nghĩa với dự báo chính xác hơn không.

## 1. Thiết kế

| Bộ biến | Nội dung |
| --- | --- |
| **A** | Chỉ lịch sử nhu cầu (độ trễ, thống kê trượt) |
| **B** | A + lịch và sự kiện (thứ, tháng, cuối tuần, mã hóa sin/cos, loại sự kiện, ngày lễ, trước/sau sự kiện) |
| **C** | B + SNAP + giá bán |

So sánh A→B đo giá trị của thông tin lịch; B→C đo phần đóng góp thêm của SNAP và giá.

**Dữ liệu:** M5 (Walmart), cửa hàng CA_1, nhóm hàng FOODS, 100 mặt hàng lấy mẫu phân tầng theo doanh số. Dự báo 1 ngày tới.
Bảng dài 194.100 dòng; bỏ 35.014 dòng mặt hàng chưa lên kệ (chưa có giá) và 28 ngày đầu mỗi chuỗi chưa đủ lịch sử, còn 156.286 dòng; chia theo thời gian 70/15/15 → huấn luyện 99.319, kiểm định 28.267, kiểm tra 28.700 dòng.

**Mô hình**

- **LightGBM:** hàm mục tiêu Tweedie (variance power 1,2), learning rate 0,03, tối đa 3.000 cây, dừng sớm sau 100 vòng, subsample/colsample 0,8; chọn cấu hình trong 3 tổ hợp (num_leaves, min_child_samples) = (15, 50), (31, 20), (63, 20) theo tập kiểm định.
- **LSTM (Keras):** nhánh chuỗi 28 ngày gồm LSTM 64 → Dropout 0,2 → LSTM 32; nhúng mã mặt hàng (kích thước 4); nhánh biến lịch của ngày cần dự báo; Dense 32 (ReLU); đầu ra softplus. Hàm mất mát Poisson, Adam 1e-3, batch 256, tối đa 40 epoch, dừng sớm (patience 6), ReduceLROnPlateau. Nhu cầu biến đổi log1p và chuẩn hóa bằng thống kê của tập huấn luyện. Chạy 3 seed (42, 7, 2024) và lấy trung bình.
- **Mốc so sánh:** Naive, Seasonal Naive (7 ngày), trung bình trượt 28 ngày (MA28).

**Đánh giá:** WAPE, MAE, RMSE, RMSSE, Bias; phân tích sai số theo loại ngày (ngày thường, cuối tuần, trước sự kiện, sự kiện khác, ngày lễ); độ dao động của LSTM qua 3 seed; **thời gian huấn luyện** của từng lần huấn luyện (notebook v4).

## 2. Kết quả trên tập kiểm tra (notebook v2)

| Mô hình | WAPE (%) |
| --- | ---: |
| Naive | 74,12 |
| Seasonal Naive | 76,08 |
| MA28 | 64,90 |
| LightGBM-A | 61,68 |
| LightGBM-B | 59,63 |
| **LightGBM-C** | **59,57** |
| LSTM-A | 61,13 |
| LSTM-B | 60,63 |
| LSTM-C | 60,67 |

Độ lệch chuẩn WAPE của LSTM qua 3 seed: 0,26 / 0,26 / 0,38 điểm phần trăm (A / B / C).

Trên các ngày lễ/sự kiện, WAPE của LightGBM giảm từ 69,72% (bộ A) xuống 63,03% (bộ C).

**Tóm tắt:** khi chỉ có lịch sử bán (bộ A), LSTM có WAPE và MAE thấp hơn LightGBM (61,13% so với 61,68%), tức mạng hồi quy tự học được nhịp nhu cầu từ chuỗi; RMSE thì LightGBM thấp hơn (2,1667 so với 2,1830). Khi thêm lịch (bộ B), LightGBM giảm WAPE 2,05 điểm và vượt LSTM, còn LSTM chỉ giảm 0,50 điểm. SNAP và giá (bộ C) gần như không thay đổi kết quả ở tầm 1 ngày.

### Thời gian huấn luyện (notebook v4)

Thời gian không dùng để xếp hạng mô hình nhanh hay chậm, mà để kiểm tra xem huấn luyện lâu hơn có mua được độ chính xác hơn không.

| Bộ | LSTM: thời gian 3 seed (CPU Colab, notebook v2) | LSTM: epoch theo seed | LightGBM: thời gian 3 cấu hình |
| --- | ---: | ---: | ---: |
| A | 4.379 giây (1.577 / 1.241 / 1.561) | 40 / 32 / 40 | chạy notebook v4 để đo |
| B | 3.382 giây (1.379 / 882 / 1.121) | 36 / 23 / 29 | chạy notebook v4 để đo |
| C | 2.923 giây (1.250 / 661 / 1.012) | 32 / 17 / 26 | chạy notebook v4 để đo |

Notebook v2 không bấm giờ LightGBM; notebook v4 bấm giờ cả hai mô hình trên cùng một runtime và lưu `training_time_summary.csv`, `fig_time_vs_wape.png`.

## 3. Cấu trúc repo

```
notebooks/
  m5_calendar_aware_lightgbm_lstm_v4.ipynb   # bản dùng cho báo cáo cuối kỳ: v2 + đo thời gian huấn luyện, bỏ kiểm định DM
  m5_calendar_aware_lightgbm_lstm_v2.ipynb   # bản đã chạy, giữ nguyên kết quả (số liệu ở mục 2)
  m5_calendar_value_lightgbm_lstm_v3.ipynb   # v2 + mục 16: thí nghiệm bổ sung (tùy chọn)
src/m5cal/timing.py                          # đo thời gian huấn luyện, bảng thời gian × độ chính xác
src/m5cal/supplementary.py                   # độ đo, bootstrap khối tuần, permutation importance (dùng cho v3)
tests/                                       # kiểm thử trên dữ liệu giả lập (không cần M5)
```

Notebook v4 thêm mục 12.5 *Training time and accuracy*: bấm giờ từng lần huấn luyện LightGBM (3 cấu hình × 3 bộ) và LSTM (3 seed × 3 bộ) bằng `time.perf_counter()`, ghi riêng thời gian dự báo, in cấu hình phần cứng, rồi đặt thời gian cạnh WAPE/MAE/RMSE. Mã trong ô cấu hình chép nguyên văn từ `timing.py`. Muốn chỉ đo LightGBM nhanh, chạy từ mục 0 đến hết mục 10; toàn bộ notebook mất khoảng 3 giờ trên CPU, chủ yếu do LSTM (9 lần huấn luyện).

## 4. Cách chạy

**Trên Google Colab (khuyến nghị)**

1. Bấm nút *Open in Colab* ở đầu trang (hoặc *File → Open notebook → GitHub*, dán đường dẫn repo và chọn notebook).
2. Chọn runtime và **giữ nguyên runtime đó cho cả notebook** để thời gian hai mô hình so được với nhau. Kết quả thời gian LSTM trong báo cáo đo trên runtime CPU; chọn T4 GPU sẽ làm LSTM nhanh hơn nhiều, khi đó ghi rõ phần cứng.
3. *Runtime → Run all.* Notebook tải dữ liệu M5 qua `kagglehub` (bản sao công khai `aryayadav0513/m5-forecasting-accuracy`). Nếu Kaggle yêu cầu đăng nhập, thêm `KAGGLE_USERNAME` và `KAGGLE_KEY` vào **Colab Secrets** (biểu tượng chìa khóa ở thanh bên trái), không dán khóa vào mã.
4. Kết quả (bảng, hình, các tệp `training_time_*.csv`) được ghi vào thư mục `outputs` cạnh dữ liệu.

**Chạy kiểm thử trên máy**

```bash
pip install numpy pandas scipy pytest
python -m pytest -q
```

## 5. Hạn chế

- Chỉ một cửa hàng, một nhóm hàng, 100 mặt hàng và dự báo 1 ngày tới; kết luận chưa tổng quát cho toàn bộ M5 hay tầm dự báo dài hơn.
- LSTM chỉ chạy một cấu hình còn LightGBM được chọn trong 3 cấu hình; mục 16.4 (v3) có mã tinh chỉnh LSTM tương đương.
- Thời gian huấn luyện phụ thuộc phần cứng (CPU/GPU, số lõi); chỉ so sánh trong cùng một lần chạy.
- Bộ B gộp chung biến lịch nội sinh (thứ, tháng) và sự kiện, nên chưa tách được đóng góp riêng của từng nhóm.

## 6. Dữ liệu và bảo mật

Repo **không** chứa dữ liệu M5 (theo điều khoản của cuộc thi trên Kaggle) và không chứa khóa Kaggle hay token. `.gitignore` chặn `data/`, `*.csv`, `*.parquet`, `kaggle.json`, `.env`.

Dữ liệu: Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. *International Journal of Forecasting, 38*(4), 1346–1364.

## Giấy phép

MIT (xem `LICENSE`).
