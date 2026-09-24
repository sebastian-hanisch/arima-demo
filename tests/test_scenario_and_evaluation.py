"""Die Reihe (identisch zu den Vorgängern), die Sonderfälle in der Analyse, Ordnungswahl, Umrechnung Glättung -> ARIMA und die vier Experimente."""

import dataclasses

import numpy as np
import pytest

import ari_constants as C
import ari_ets as ETS
import ari_evaluation as E
import ari_forecast as F
import ari_sarima as SAR
import ari_scenario as S


def test_scenario_is_the_same_vehicle_as_in_the_predecessors():
    ser = S.generate(seed=3)
    assert ser.y[:8].tolist() == [103.0, 131.0, 117.0, 117.0, 127.0, 64.0, 46.0, 132.0] and float(ser.y.sum()) == 126867.0 and float(ser.mu.sum()) == pytest.approx(126496.28723546621)


def test_analyse_is_consistent_and_special_cases_equal_the_predecessors():
    s = E.Settings(seed=2)
    a = E.analyse(s)
    assert set(a.summary) == set(C.METHODS) and a.best in C.METHODS and len(a.origins) == len(range(C.FIRST_TEST, C.N_DAYS - s.horizon + 1, s.step))
    assert a.oracle["mase"] < min(v["mase"] for v in a.summary.values())
    y = a.series.y
    org = a.origins
    naive = np.repeat(y[org - 1][:, None], s.horizon, axis=1)
    snaive = y[org[:, None] - 7 + (np.arange(s.horizon) % 7)[None, :]]
    assert a.errors["naive"] + a.actual == pytest.approx(naive) and a.errors["snaive"] + a.actual == pytest.approx(snaive)
    assert a.summary["arima011"]["mase"] == pytest.approx(a.summary["ses"]["mase"], abs=3e-3)     # einfache Glättung = ARIMA(0,1,1), beide unabhängig geschätzt
    loop = np.stack([np.mean([y[t - 7 * (i + 1) + (np.arange(s.horizon) % 7)] for i in range(s.k)], axis=0) for t in org])
    assert a.errors["snaive_k"] + a.actual == pytest.approx(loop)


def test_analyse_with_a_subset_of_methods_gives_the_same_numbers():
    full = E.analyse(E.Settings(seed=2))
    part = E.analyse(E.Settings(seed=2), ("user", "snaive_k"))
    assert set(part.summary) == {"user", "snaive_k"} and part.summary["user"]["mase"] == full.summary["user"]["mase"]


def test_settings_map_to_the_spec_and_the_cache_is_shared():
    s = E.Settings(p=2, d=1, q=0, P=1, D=1, Q=0, log=False)
    assert s.spec == SAR.Spec(2, 1, 0, 1, 1, 0, False)
    E._fit_arima.cache_clear()
    E.analyse(E.Settings(seed=9), ("user",))
    n = E._fit_arima.cache_info().misses
    E.analyse(E.Settings(seed=9, horizon=7, k=8), ("user",))
    assert E._fit_arima.cache_info().misses == n


def test_horizon_and_step_change_the_result():
    base = E.analyse(E.Settings(seed=2))
    assert len(E.analyse(E.Settings(seed=2, step=7)).origins) == pytest.approx(len(base.origins) / 7, abs=1)
    assert E.analyse(E.Settings(seed=2, horizon=28)).summary["user"]["mae"] > base.summary["user"]["mae"]


def test_event_classes():
    ser = S.generate(seed=1)
    cls = E.event_classes(ser)
    ev = (ser.holiday + np.roll(ser.holiday, 1) + ser.promo) > 0
    assert np.array_equal(cls == 0, ev) and set(np.unique(cls)) == {0, 1, 2}
    far = np.flatnonzero(cls == 2)
    assert all(not ev[d - C.EVENT_AFTER_DAYS:d + 1].any() for d in far[far > 20][:50])


def test_auto_search_is_sorted_and_comparable():
    rows = E.auto_search(E.Settings(seed=2))
    assert len(rows) == len(E.auto_grid()) == 24 and [r[2] for r in rows] == sorted(r[2] for r in rows)
    assert len({r[1].n_fit for r in rows}) == 1                                    # alle Fehlersummen laufen über dieselben Tage
    assert all(r[0].D == 1 and r[0].Q == 1 and r[0].log for r in rows) and {r[0].d for r in rows} == set(C.AUTO_D)


# --- Umrechnung Glättungsmodell -> ARIMA ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["ses", "holt", "damped"])
def test_converted_arima_parameters_reproduce_the_smoothing_forecast(key):
    y = S.generate(seed=5).y
    fitted = ETS.fit(y[:C.FIRST_TEST], key)
    model = dataclasses.replace(fitted, alpha=0.3, beta=0.05 if key != "ses" else 0.0, phi=0.9 if key == "damped" else 1.0)     # kräftige Parameter: der Unterschied der Anfangszustände verklingt rasch
    states = ETS.filter_states(model, y)
    org = np.arange(600, 1000, 37)
    f_ets = ETS.forecast_origins(model, states, org, 14)
    p, d, q = dict(C.EQUIVALENCES)[key]
    phi, theta = E.ets_to_arima(model, key)
    conv = SAR.Model(SAR.Spec(p, d, q), phi, (), theta, (), 0.0, 1.0, 100, 1.0)
    assert SAR.forecast_origins(conv, SAR.filter_series(conv, y), org, 14) == pytest.approx(f_ets, abs=1e-6)


def test_ets_to_arima_by_hand():
    m = dataclasses.replace(ETS.fit(S.generate(seed=1).y[:C.FIRST_TEST], "damped"), alpha=0.3, beta=0.1, phi=0.9)
    assert E.ets_to_arima(m, "ses") == ((), (-0.7,))
    assert E.ets_to_arima(m, "holt") == ((), (pytest.approx(0.3 + 0.1 - 2), pytest.approx(0.7)))
    phi, theta = E.ets_to_arima(m, "damped")
    assert phi == (0.9,) and theta == (pytest.approx(0.3 + 0.09 - 1.9), pytest.approx(0.63))
    with pytest.raises(ValueError):
        E.ets_to_arima(m, "hw_mult")


# --- Experimente --------------------------------------------------------------------------------------------------------------------------------


def test_experiments_return_consistent_rows():
    eq = E.equivalence_experiment(seeds=(0, 1))
    assert [r["key"] for r in eq] == ["ses", "holt", "damped"] and eq[0]["max_diff"] < 1e-8 and abs(eq[0]["ets"] - eq[0]["arima"]) < 2e-3 and eq[0]["theta_arima"] < 0 and 0.95 < eq[1]["theta_ets"] < 1.0
    sp = E.spec_experiment(seeds=(0, 1), specs=E.SPEC_TABLE[:2], noises=(0.14,))
    assert [r["label"] for r in sp["rows"]] == ["(0,1,1)(0,1,1)", "(0,1,1)(0,1,1) log"] and sp["rows"][1]["mase"] < sp["rows"][0]["mase"] and len(sp["bias"]) == 1 and sp["hw_mult"] > 0
    ev = E.events_experiment(levels=(0.0, 1.0), seeds=(0, 1))
    assert [x["events"] for x in ev] == [0.0, 1.0] and sum(ev[0]["share"]) == pytest.approx(1.0) and ev[1]["ar_log"][0] > 2 * ev[0]["ar_log"][0] and ev[0]["share"] == ev[1]["share"]
    au = E.auto_experiment(seeds=(0,))
    assert au["grid_size"] == 24 and len(au["picks"]) == 1 and au["best"] <= au["selected"][0] + 1e-12


def test_diagnostics_keys_and_shapes():
    a = E.analyse(E.Settings(seed=2), ("user",))
    dg = E.diagnostics(a)
    assert len(dg["acf_z"]) == len(dg["acf_w"]) == len(dg["pacf_w"]) == len(dg["acf_res"]) == C.ACF_LAGS and 0 <= dg["lb_p"] <= 1 and len(dg["w"]) == C.FIRST_TEST - 7 and len(dg["resid"]) == C.FIRST_TEST - SAR.BURN
    raw = E.analyse(E.Settings(seed=2, p=0, d=0, q=0, P=0, D=0, Q=0, log=False), ("user",))
    assert E.diagnostics(raw)["acf_z"][6] > 0.6                                    # Wochenmuster: hohe Autokorrelation bei Lag 7
