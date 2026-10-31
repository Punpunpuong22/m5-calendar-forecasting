# Dự báo nhu cầu bán lẻ theo ngày có xét yếu tố lịch: LightGBM và LSTM trên dữ liệu M5

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Punpunpuong22/m5-calendar-forecasting/blob/main/notebooks/m5_calendar_aware_lightgbm_lstm_v2.ipynb)

Mã nguồn cho bài báo cáo môn **Học sâu và Ứng dụng trong kinh doanh** (GV: TS. Tạ Hoàng Thắng; học viên: Nguyễn Vũ Mai Phương – 26848176), một phần của luận văn thạc sĩ về tác động của yếu tố lịch và sự kiện trong dự báo nhu cầu bán lẻ.

**Câu hỏi nghiên cứu:** thêm thông tin lịch và sự kiện vào đầu vào có làm dự báo nhu cầu ngày tốt hơn không, mức cải thiện có ý nghĩa thống kê không, và mô hình cây tăng cường (LightGBM) hay mạng hồi quy (LSTM) tận dụng thông tin này tốt hơn?

## 1. Thiết kế

| Bộ biến | Nội dung |
| --- | --- |
| **A** | Chỉ lịch sử nhu cầu (độ trễ, thống kê trượt) |
| **B** | A + lịch và sự kiện (thứ, tháng, cuối tuần, mã hóa sin/cos, loại sự kiện, ngày lễ, trước/sau sự kiện) |
| **C** | B + SNAP + giá bán |

So sánh A→B đo giá trị của thông tin lịch; B→C đo phần đóng góp thêm của SNAP và giá.

**Dữ liệu:** M5 (Walmart), cửa hàng CA_1, nhóm hàng FOODS, 100 mặt hàng lấy mẫu phân tầng theo doanh số. Dự báo 1 ngày tới.
Bảng dài 194.100 dòng; bỏ 35.014 dòng đầu chuỗi chưa đủ độ trễ, còn 156.286 dòng; chia theo thời gian 70/15/15 → huấn luyện 99.319, kiểm định 28.267, kiểm tra 28.700 dòng.

**Mô hình**

- **LightGBM:** hàm mục tiêu Tweedie (variance power 1,2), learning rate 0,03, tối đa 3.000 cây, dừng sớm sau 100 vòng, subsample/colsample 0,8; chọn cấu hình trong 3 tổ hợp (num_leaves, min_child_samples) = (15, 50), (31, 20), (63, 20) theo tập kiểm định.
- **LSTM (Keras):** nhánh chuỗi 28 ngày gồm LSTM 64 → Dropout 0,2 → LSTM 32; nhúng mã mặt hàng (kích thước 4); nhánh biến lịch của ngày cần dự báo; Dense 32 (ReLU); đầu ra softplus. Hàm mất mát Poisson, Adam 1e-3, batch 256, tối đa 40 epoch, dừng sớm (patience 6), ReduceLROnPlateau. Nhu cầu biến đổi log1p và chuẩn hóa bằng thống kê của tập huấn luyện. Chạy 3 seed (42, 7, 2024) và lấy trung bình.
- **Mốc so sánh:** Naive, Seasonal Naive (7 ngày), trung bình trượt 28 ngày (MA28).

**Đánh giá:** WAPE, RMSE, RMSSE, Bias; kiểm định Diebold–Mariano trên sai số gộp theo ngày (phương sai Newey–West, trễ Bartlett 7); phân tích sai số theo loại ngày (ngày thường, cuối tuần, ngày lễ/sự kiện).

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

**Kiểm định Diebold–Mariano** (thống kê > 0 nghĩa là mô hình thứ hai tốt hơn)

| So sánh | Sai số tuyệt đối | Sai số bình phương |
| --- | --- | --- |
| LightGBM A → B | 6,90 (p < 0,0001) | 3,58 (p = 0,0003) |
| LSTM A → B | 2,52 (p = 0,012) | 0,49 (p = 0,627) |
| B → C (cả hai mô hình) | không có ý nghĩa | không có ý nghĩa |
| LightGBM-B so với LSTM-B | −3,29 (p = 0,001), LightGBM tốt hơn | |
| LightGBM-C so với LSTM-C | −4,39 (p < 0,0001), LightGBM tốt hơn | |

Trên các ngày lễ/sự kiện, WAPE của LightGBM giảm từ 69,72% (bộ A) xuống 63,03% (bộ C).

**Tóm tắt:** thông tin lịch giúp cả hai mô hình, rõ nhất với LightGBM; SNAP và giá không thêm cải thiện có ý nghĩa ở mức 1 ngày tới; LightGBM tốt hơn LSTM có ý nghĩa thống kê trong thiết lập này.

## 3. Cấu trúc repo

```
notebooks/
  m5_calendar_aware_lightgbm_lstm_v2.ipynb   # thí nghiệm chính, giữ nguyên kết quả đã chạy (số liệu ở mục 2)
  m5_calendar_value_lightgbm_lstm_v3.ipynb   # v2 + mục 16: thí nghiệm bổ sung
src/m5cal/supplementary.py                   # độ đo, Diebold–Mariano, bootstrap khối tuần, permutation importance
tests/test_supplementary.py                  # kiểm thử trên dữ liệu giả lập (không cần M5)
```

Mục 16 của notebook v3 gồm: kiểm định DM bổ sung và trung bình hai mô hình, khoảng tin cậy bootstrap khối tuần cho chênh lệch WAPE, chi phí tính toán, tinh chỉnh LSTM với cùng ngân sách như LightGBM, và permutation importance theo nhóm biến. Mã trong ô đầu mục 16 chép nguyên văn từ `supplementary.py`.

## 4. Cách chạy

**Trên Google Colab (khuyến nghị)**

1. Bấm nút *Open in Colab* ở đầu trang (hoặc *File → Open notebook → GitHub*, dán đường dẫn repo và chọn notebook).
2. Chọn *Runtime → Change runtime type → T4 GPU* để LSTM chạy nhanh hơn.
3. *Runtime → Run all.* Notebook tải dữ liệu M5 qua `kagglehub` (bản sao công khai `aryayadav0513/m5-forecasting-accuracy`). Nếu Kaggle yêu cầu đăng nhập, thêm `KAGGLE_USERNAME` và `KAGGLE_KEY` vào **Colab Secrets** (biểu tượng chìa khóa ở thanh bên trái), không dán khóa vào mã.
4. Kết quả (bảng, hình) được ghi vào thư mục `outputs` cạnh dữ liệu. Ở notebook v3, đặt `SAVE_TO_DRIVE = True` để chép sang Google Drive.

**Chạy kiểm thử trên máy**

```bash
pip install numpy pandas scipy pytest
python -m pytest -q
```

## 5. Hạn chế

- Chỉ một cửa hàng, một nhóm hàng, 100 mặt hàng và dự báo 1 ngày tới; kết luận chưa tổng quát cho toàn bộ M5 hay tầm dự báo dài hơn.
- Trong v2, LSTM chỉ chạy một cấu hình còn LightGBM được chọn trong 3 cấu hình; mục 16.4 (v3) bổ sung phần tinh chỉnh LSTM tương đương.
- Bộ B gộp chung biến lịch nội sinh (thứ, tháng) và sự kiện, nên chưa tách được đóng góp riêng của từng nhóm.

## 6. Dữ liệu và bảo mật

Repo **không** chứa dữ liệu M5 (theo điều khoản của cuộc thi trên Kaggle) và không chứa khóa Kaggle hay token. `.gitignore` chặn `data/`, `*.csv`, `*.parquet`, `kaggle.json`, `.env`.

Dữ liệu: Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. *International Journal of Forecasting, 38*(4), 1346–1364.

## Giấy phép

MIT (xem `LICENSE`).
