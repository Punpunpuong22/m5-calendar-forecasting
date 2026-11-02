"""Đo thời gian huấn luyện và đặt cạnh độ chính xác (notebook v4).

Mục đích không phải để xếp hạng mô hình theo tốc độ, mà để cho thấy thời gian huấn luyện
và độ chính xác là hai trục khác nhau: một mô hình tốn thời gian hơn nhiều lần chưa chắc
chính xác hơn, và ngược lại.

Notebook v4 chép nguyên văn tệp này vào một ô, nên mã trong notebook và mã được kiểm thử là một.
"""
from __future__ import annotations

import os
import platform
import time
from typing import Mapping, Sequence

import pandas as pd


class StopWatch:
    """Đồng hồ bấm giờ dùng với `with`: `with StopWatch() as sw: ...; sw.seconds`."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        self.seconds = float("nan")
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        return False


def hardware_info(gpu_devices: Sequence | None = None) -> dict:
    """Mô tả phần cứng để ghi kèm số đo thời gian (thời gian chỉ so được trên cùng phần cứng)."""
    return {"python": platform.python_version(), "machine": platform.machine(),
            "processor": platform.processor() or "n/a", "cpu_count": os.cpu_count(),
            "gpu": ", ".join(str(d) for d in gpu_devices) if gpu_devices else "none (CPU only)"}


def summarise_training_time(lgb_log: pd.DataFrame, lstm_log: pd.DataFrame,
                            results: pd.DataFrame | None = None) -> pd.DataFrame:
    """Gộp nhật ký thời gian thành một bảng theo mô hình × bộ biến.

    lgb_log  : một dòng cho mỗi lần huấn luyện LightGBM, cột `set`, `fit_sec`, `best_iter`, `selected` (bool).
    lstm_log : một dòng cho mỗi lần huấn luyện LSTM (mỗi seed), cột `set`, `sec`, `epochs`.
    results  : bảng chỉ số trên tập test, index là tên mô hình ("LightGBM-A", "LSTM-B", ...).

    Trả về các cột: model, set, fits, total_train_sec (tổng mọi lần huấn luyện của bộ đó: 3 cấu hình
    với LightGBM, 3 seed với LSTM), sec_per_fit, final_model_sec (thời gian của mô hình dùng để dự báo:
    cấu hình được chọn với LightGBM, trung bình một seed với LSTM), units (số cây / số epoch trung bình)
    và các chỉ số WAPE_%, MAE, RMSE nếu có `results`.
    """
    rows = []
    for fs, g in lgb_log.groupby("set", sort=True):
        sel = g[g["selected"]] if "selected" in g and g["selected"].any() else g.iloc[[0]]
        rows.append({"model": "LightGBM", "set": fs, "fits": len(g), "total_train_sec": g["fit_sec"].sum(),
                     "sec_per_fit": g["fit_sec"].mean(), "final_model_sec": float(sel["fit_sec"].iloc[0]),
                     "units": f"{int(sel['best_iter'].iloc[0])} cây"})
    for fs, g in lstm_log.groupby("set", sort=True):
        rows.append({"model": "LSTM", "set": fs, "fits": len(g), "total_train_sec": g["sec"].sum(),
                     "sec_per_fit": g["sec"].mean(), "final_model_sec": g["sec"].mean(),
                     "units": f"{g['epochs'].mean():.1f} epoch"})
    out = pd.DataFrame(rows)
    if results is not None and len(out):
        name = out["model"] + "-" + out["set"]
        for col in ["WAPE_%", "MAE", "RMSE"]:
            if col in results.columns:
                out[col] = name.map(results[col])
    return out


def time_ratio(summary: pd.DataFrame, col: str = "total_train_sec") -> pd.Series:
    """Tỉ lệ thời gian LSTM / LightGBM cho từng bộ biến."""
    p = summary.pivot(index="set", columns="model", values=col)
    if not {"LSTM", "LightGBM"} <= set(p.columns):
        return pd.Series(dtype=float)
    return (p["LSTM"] / p["LightGBM"]).rename(f"LSTM / LightGBM ({col})")
