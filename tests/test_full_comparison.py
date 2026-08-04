import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _toy_comparison():
    return pd.DataFrame(
        {
            "model": ["sarimax", "seasonal_naive_weekly", "naive"],
            "MAE": [38.0, 43.5, 85.5],
            "RMSE": [65.7, 81.4, 110.4],
            "MASE": [0.943, 1.077, 2.121],
            "Bias": [-5.0, -13.2, 51.0],
        }
    )


def test_pct_improvement_and_beats_flag():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_full_comparison import compare_to_strongest_benchmark

    result = compare_to_strongest_benchmark(_toy_comparison())

    sarimax_row = result[result["model"] == "sarimax"].iloc[0]
    benchmark_row = result[result["model"] == "seasonal_naive_weekly"].iloc[0]
    naive_row = result[result["model"] == "naive"].iloc[0]

    # sarimax MAE 38.0 vs benchmark 43.5 -> ~12.6% improvement
    assert sarimax_row["pct_MAE_vs_strongest_benchmark"] == pytest.approx(12.6, abs=0.1)
    assert sarimax_row["beats_strongest_benchmark"] == True  # noqa: E712

    # the benchmark compared to itself -> 0% improvement, doesn't "beat" itself
    assert benchmark_row["pct_MAE_vs_strongest_benchmark"] == pytest.approx(0.0)
    assert benchmark_row["beats_strongest_benchmark"] == False  # noqa: E712

    # naive is much worse -> negative improvement, does not beat benchmark
    assert naive_row["pct_MAE_vs_strongest_benchmark"] < 0
    assert naive_row["beats_strongest_benchmark"] == False  # noqa: E712
