"""Kiểm thử cho phần đo thời gian huấn luyện (không cần dữ liệu M5)."""
import math
import time

import pandas as pd

from m5cal.timing import StopWatch, hardware_info, summarise_training_time, time_ratio


def _logs():
    lgb = pd.DataFrame({"set": ["A"] * 3 + ["B"] * 3,
                        "fit_sec": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
                        "best_iter": [100, 120, 130, 300, 380, 200],
                        "selected": [False, False, True, False, True, False]})
    lstm = pd.DataFrame({"set": ["A"] * 3 + ["B"] * 3,
                         "sec": [60.0, 30.0, 30.0, 90.0, 60.0, 30.0],
                         "epochs": [40, 20, 20, 30, 20, 10]})
    res = pd.DataFrame({"WAPE_%": [61.7, 59.6, 61.1, 60.6], "MAE": [1.2, 1.16, 1.19, 1.18], "RMSE": [2.17, 2.09, 2.18, 2.17]},
                       index=["LightGBM-A", "LightGBM-B", "LSTM-A", "LSTM-B"])
    return lgb, lstm, res


def test_stopwatch_measures_elapsed_time():
    with StopWatch() as sw:
        time.sleep(0.01)
    assert 0.005 < sw.seconds < 1.0


def test_summary_totals_selected_model_and_metrics():
    lgb, lstm, res = _logs()
    s = summarise_training_time(lgb, lstm, res).set_index(["model", "set"])
    assert s.loc[("LightGBM", "A"), "total_train_sec"] == 6.0
    assert s.loc[("LightGBM", "A"), "final_model_sec"] == 3.0          # cấu hình được chọn
    assert s.loc[("LightGBM", "B"), "units"] == "380 cây"
    assert s.loc[("LSTM", "A"), "sec_per_fit"] == 40.0
    assert s.loc[("LSTM", "B"), "units"] == "20.0 epoch"
    assert s.loc[("LSTM", "A"), "WAPE_%"] == 61.1
    assert s.loc[("LightGBM", "B"), "fits"] == 3


def test_time_ratio_and_missing_family():
    lgb, lstm, res = _logs()
    s = summarise_training_time(lgb, lstm, res)
    r = time_ratio(s)
    assert math.isclose(r["A"], 120 / 6) and math.isclose(r["B"], 180 / 12)
    only_lgb = summarise_training_time(lgb, lstm.iloc[0:0])
    assert time_ratio(only_lgb).empty


def test_hardware_info_keys():
    info = hardware_info(["/physical_device:GPU:0"])
    assert info["gpu"].endswith("GPU:0") and info["cpu_count"] >= 1
    assert hardware_info()["gpu"].startswith("none")
