"""Unit tests for metrics, statistics, protocols, baselines and the backtest runner.

They run on CPU without the foundation-model packages: ``pytest -q``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ciexp import data as D  # noqa: E402
from ciexp import metrics as MT  # noqa: E402
from ciexp import models as M  # noqa: E402
from ciexp import protocols as P  # noqa: E402
from ciexp import runner as R  # noqa: E402
from ciexp import stats as S  # noqa: E402


# ----------------------------------------------------------------------------- metrics
def test_pinball_and_quantile_losses():
    y = np.array([1.0])
    q = np.array([[0.0]])
    assert MT.pinball(y, q, np.array([0.9]))[0, 0] == pytest.approx(0.9)
    assert MT.pinball(y, np.array([[2.0]]), np.array([0.9]))[0, 0] == pytest.approx(0.1)
    perfect = np.repeat(np.array([[1.0, 2.0, 3.0]]), 9, axis=0)
    assert MT.wql(np.array([1.0, 2.0, 3.0]), perfect, M.LEVELS) == pytest.approx(0.0)


def test_mase_and_sql_agree_for_the_median():
    hist = np.array([1.0, 2.0, 3.0, 4.0])
    y, point = np.array([5.0, 6.0]), np.array([5.0, 7.0])
    assert MT.mase(y, point, hist, m=1) == pytest.approx(0.5)
    assert MT.sql(y, point[None], np.array([0.5]), hist, m=1) == pytest.approx(0.5)


def test_coverage_interval_score_and_calibration():
    y = np.array([0.0, 1.0, 2.0, 10.0])
    q = np.stack([np.full(4, lvl * 4) for lvl in M.LEVELS])  # constant quantile lines
    assert MT.coverage(y, q, M.LEVELS, 0.1, 0.9) == pytest.approx(0.5)
    assert MT.interval_score(y, q, M.LEVELS, 0.1, 0.9) > MT.interval_width(q, M.LEVELS, 0.1, 0.9)
    curve = MT.calibration_curve(y, q, M.LEVELS)
    assert np.all(np.diff(curve) >= 0)


def test_qlike_kge_nan_handling():
    assert MT.qlike(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(0.0)
    assert MT.kge(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert MT.mae(np.array([1.0, np.nan]), np.array([2.0, 5.0])) == pytest.approx(1.0)


# ------------------------------------------------------------------------------- stats
def test_diebold_mariano_direction():
    rng = np.random.default_rng(0)
    e_good = rng.normal(scale=1.0, size=500)
    e_bad = rng.normal(scale=2.0, size=500)
    res = S.diebold_mariano(e_good, e_bad, horizon=1)
    assert res.statistic < 0 and res.p_less < 0.01
    res_h = S.diebold_mariano(e_good, e_bad, horizon=7, loss="sq")
    assert np.isfinite(res_h.statistic)


def test_multiplicity_corrections():
    p = np.array([0.01, 0.04, 0.03])
    np.testing.assert_allclose(S.holm(p), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(S.benjamini_hochberg(p), [0.03, 0.04, 0.04])


def test_model_confidence_set_keeps_best_drops_worst():
    rng = np.random.default_rng(1)
    n = 400
    losses = np.column_stack([rng.gamma(2, 1.0, n), rng.gamma(2, 1.02, n), rng.gamma(2, 3.0, n)])
    res = S.model_confidence_set(losses, ["best", "close", "bad"], alpha=0.1, reps=500)
    assert "best" in res.included and "bad" not in res.included


def test_skill_score_and_win_rate():
    errors = np.array([[1.0, 0.5], [2.0, 1.0], [4.0, 2.0]])
    np.testing.assert_allclose(S.skill_score(errors, baseline_col=0), [0.0, 0.5])
    np.testing.assert_allclose(S.win_rate(errors), [0.0, 1.0])
    filled = S.fill_failures(np.array([[1.0, np.nan]]), baseline_col=0)
    assert filled[0, 1] == 1.0
    point, lo, hi = S.paired_bootstrap_ci(errors, lambda e: S.skill_score(e, 0), reps=200)
    assert lo[1] <= point[1] <= hi[1]


def test_friedman_nemenyi_ranks():
    errors = np.array([[1, 2, 3], [1, 3, 2], [1, 2, 3], [2, 1, 3]], dtype=float)
    out = S.friedman_nemenyi(errors)
    assert out["avg_ranks"][0] < out["avg_ranks"][2] and out["cd"] > 0


# --------------------------------------------------------------------------- protocols
def test_rolling_origins_and_fev_windows():
    origins = P.rolling_origins(100, 10, n_windows=3)
    assert [o.cutoff for o in origins] == [70, 80, 90]
    assert P.fev_windows(5) == 20 and P.fev_windows(500) == 10 and P.fev_windows(5000) == 1


def test_date_split_origins_stay_inside_test_period():
    import pandas as pd

    idx = pd.date_range("2023-12-01", "2024-01-31", freq="D")
    origins = P.date_split_origins(idx, 7, test_start="2024-01-01", test_end="2024-01-31")
    assert idx[origins[0].cutoff] == pd.Timestamp("2024-01-01")
    assert all(idx[o.cutoff + 6] <= pd.Timestamp("2024-01-31") for o in origins)
    assert P.last_admissible_cutoff(idx, 7, "2023-12-31") + 7 - 1 == int(np.where(idx == "2023-12-31")[0][0])


def test_scarcity_subset_and_missing_injection():
    keep = P.scarcity_subset(1000, 0.2, seed=0)
    assert 190 <= keep.size <= 210 and keep.max() < 1000
    x = P.inject_missing(np.ones((2, 500)), 0.25, pattern="block", block=24, seed=0)
    assert 0.1 < np.isnan(x).mean() < 0.4 and np.isfinite(x[:, -1]).all()
    assert len(P.context_grid("pow2")) == 16 and P.context_grid("pow2")[-1] == 2**21


# ------------------------------------------------------------------------- baselines
def test_seasonal_naive_repeats_last_season_and_orders_quantiles():
    hist = np.tile(np.arange(24, dtype=float), 10)[None]
    q = M.SeasonalNaive(season=24).predict(hist, 48)
    np.testing.assert_allclose(q[0, 4], np.tile(np.arange(24.0), 2))
    assert np.all(np.diff(q[0], axis=0) >= -1e-12)


def test_to_levels_interpolation():
    src = [0.1, 0.5, 0.9]
    q = np.array([[[0.0], [1.0], [2.0]]])
    out = M.to_levels(q, src)
    assert out.shape == (1, 9, 1) and out[0, 2, 0] == pytest.approx(0.5)


def test_lear_and_gbm_fit_predict_shapes():
    rng = np.random.default_rng(0)
    y = 10 + np.sin(np.arange(600) * 2 * np.pi / 7) + rng.normal(scale=0.1, size=600)
    lear = M.LEAR(lags=[1, 2, 7], horizon=3).fit([y[:500]])
    q = lear.predict(y[None, :500], 3)
    assert q.shape == (1, 9, 3) and np.all(np.isfinite(q))
    gbm = M.GradientBoostingQuantile(lags=[1, 7], horizon=3, max_iter=20).fit([y[:500]])
    assert gbm.predict(y[None, :500], 3).shape == (1, 9, 3)
    har = M.HARRV().predict(np.abs(rng.normal(size=(1, 300))), 2)
    assert har.shape == (1, 9, 2) and (har >= 0).all()


# ---------------------------------------------------------------------------- runner
def test_runner_on_bundled_bike_sharing():
    ds = D.bike_sharing_example()
    ds.values = ds.values[:, :120]
    ds.index = ds.index[:120]
    origins = P.rolling_origins(120, 24, n_windows=2, min_history=48)
    fc = R.run_backtest(M.SeasonalNaive(season=24), ds, origins)
    assert len(fc) == 2 * 24 and set(R.QCOLS) <= set(fc.columns)
    met = R.metrics_table(fc, ds)
    assert met["MASE"].notna().all() and (met["status"] == "ok").all()
    summary = R.aggregate(met)
    assert summary["n_windows"].iloc[0] == 2


def test_runner_records_failures_instead_of_crashing():
    class Broken(M.BaseForecaster):
        name = "broken"

        def predict(self, history, horizon, **_):
            raise RuntimeError("boom")

    ds = D.synthetic_suite(length=512)
    fc = R.run_backtest(Broken(), ds, P.rolling_origins(512, 24, n_windows=1), rows=[0])
    assert fc["status"].str.startswith("error").all() and fc["q0.5"].isna().all()


def test_synthetic_suite_has_breaks_and_is_deterministic():
    a, b = D.synthetic_suite(length=1024, seed=3), D.synthetic_suite(length=1024, seed=3)
    np.testing.assert_allclose(a.values, b.values)
    assert any(s.startswith("break_") for s in a.ids)


# ------------------------------------------------------------- quick experiment runs
@pytest.mark.parametrize("script", ["e02_leakage.py", "e04_covariates.py", "e05_context.py", "e07_ablations.py",
                                    "e08_efficiency.py", "e09_volatility.py", "e01_benchmarks.py"])
def test_quick_experiment_scripts(script, tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT / "experiments" / script), "--quick", "--out", str(tmp_path)],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
