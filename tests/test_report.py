import numpy as np

from scripts.report import curve, step_to_threshold


def _evals(scores):
    return [{"step": 250_000 * (i + 1), "mean_score": s} for i, s in enumerate(scores)]


def test_step_to_threshold_is_first_hit_and_inclusive():
    assert step_to_threshold(_evals([10, 50, 90, 5])) == 500_000  # first hit, later collapse ignored
    assert step_to_threshold(_evals([10, 49.9, 20])) is None


def test_curve_is_mean_and_sample_std_over_seeds():
    runs = [{"evals": _evals([0, 10])}, {"evals": _evals([10, 30])}]
    steps, mean, std = curve(runs)
    assert list(steps) == [250_000, 500_000]
    assert list(mean) == [5, 20]
    assert np.allclose(std, [np.std([0, 10], ddof=1), np.std([10, 30], ddof=1)])


def test_main_table_rows_use_sample_std_and_percent():
    from scripts.report import TABLE_ROWS, main_table

    per_seed = [{"method": m, "step": st, "seed": i, "mean_score": 10.0 * (i + 1), "capped_rate": 0.1 * i,
                 "mean_length": 100.0 * (i + 1)} for m, st, _ in TABLE_ROWS for i in range(2)]
    first = main_table(per_seed).splitlines()[0]
    assert first == r"Random (uniform) & $15.0 \pm 7.1$ & $5.0 \pm 7.1$ & $150 \pm 71$ \\"
