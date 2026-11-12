import json

import pytest

from scripts.ablation_report import seed_scores, welch


def test_welch_matches_known_student_t_value():
    # a - b has t = -1 with df = 8 (equal variances, n=5 each); two-sided P(|T_8| > 1) = 0.3466
    t, df, p = welch([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    assert t == pytest.approx(-1.0) and df == pytest.approx(8.0)
    assert p == pytest.approx(0.3466, abs=5e-4)


def test_welch_unequal_variances_lowers_df_and_extreme_t_gives_tiny_p():
    t, df, p = welch([0, 0.1, 0, 0.1, 0], [10, 30, 20, 40, 0])
    assert df < 5 and p < 0.05
    assert welch([1, 2, 3], [1, 2, 3])[2] == pytest.approx(1.0)


def test_seed_scores_filters_method_and_step_and_orders_by_seed(tmp_path):
    rows = [{"method": "dqn", "step": 3_000_000, "seed": s, "mean_score": float(s)} for s in (2, 0, 1)]
    rows += [{"method": "dqn", "step": 1_000_000, "seed": 0, "mean_score": 99.0},
             {"method": "qlearning", "step": 3_000_000, "seed": 0, "mean_score": 98.0}]
    path = tmp_path / "final_eval.json"
    path.write_text(json.dumps({"per_seed": rows}))
    assert seed_scores(path, "dqn") == [0.0, 1.0, 2.0]
