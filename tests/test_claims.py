"""Jede Zahl aus README und PRESET_HELP als Test. Reihen und Schätzung sind deterministisch (fester Seed der Parametersuche); die Bänder sind trotzdem großzügiger als die Rundung, damit andere numpy-/BLAS-Versionen nicht stören."""

import numpy as np
import pytest

import ari_constants as C
import ari_evaluation as E
import ari_presets as P
import ari_sarima as SAR

STD = "Standardfall: (1,0,1)(1,1,1) im Log"


def _preset(name, methods=C.METHODS, **over):
    p = dict(P.PRESETS[name])
    p.update(over)
    return E.analyse(E.Settings(p["trend"], p["weekly"], p["yearly"], p["noise"], p["shift"], p["events"], p["horizon"], p["step"], p["k"], p["seed"], p["p"], p["d"], p["q"], p["P"], p["D"], p["Q"], p["log"]), methods)


def _m(a):
    return {k: v["mase"] for k, v in a.summary.items()}


def test_standard_preset_and_agreement_with_the_predecessors():
    a = _preset(STD)
    m = _m(a)
    assert len(a.origins) == 352 and a.oracle["mase"] == pytest.approx(0.747, abs=0.01)
    assert m["user"] == pytest.approx(0.839, abs=0.02) and m["hw_mult"] == pytest.approx(0.841, abs=0.015) and m["snaive_k"] == pytest.approx(0.878, abs=0.006) and m["snaive"] == pytest.approx(1.125, abs=0.006) and m["naive"] == pytest.approx(2.677, abs=0.006)
    model = a.user[0]
    assert model.phi[0] == pytest.approx(0.99, abs=0.01) and model.Theta[0] == pytest.approx(-0.953, abs=0.03) and model.theta[0] == pytest.approx(-0.89, abs=0.06)


def test_airline_and_raw_presets():
    m = _m(_preset("Airline-Modell (0,1,1)(0,1,1) im Log"))
    assert m["user"] == pytest.approx(0.847, abs=0.02) and m["hw_mult"] == pytest.approx(0.841, abs=0.015)
    a = _preset("Ohne Log (rohe Werte)")
    assert _m(a)["user"] == pytest.approx(0.88, abs=0.02) and _m(a)["user"] > m["user"]
    assert _preset("Airline-Modell (0,1,1)(0,1,1) im Log").user[0].theta[0] == pytest.approx(-0.895, abs=0.04) and _preset("Airline-Modell (0,1,1)(0,1,1) im Log").user[0].Theta[0] == pytest.approx(-0.929, abs=0.04)


def test_special_case_presets_reproduce_the_predecessors():
    a = _preset("Einfache Glättung als ARIMA(0,1,1)")
    assert _m(a)["user"] == pytest.approx(2.021, abs=0.005) and _m(a)["user"] == pytest.approx(_m(a)["ses"], abs=2e-3) and a.user[0].theta[0] == pytest.approx(-0.959, abs=0.02)
    assert 1 + a.user[0].theta[0] == pytest.approx(a.ets["ses"][0].alpha, abs=0.005)                         # theta = alpha - 1
    b = _preset("Saisonal naiv als SARIMA")
    assert _m(b)["user"] == pytest.approx(1.125, abs=0.006) and _m(b)["user"] == pytest.approx(_m(b)["snaive"], abs=1e-12) and b.user[0].n_params == 1


def test_strong_events_preset():
    a = _preset("Feiertage und Aktionen stark")
    m = _m(a)
    assert m["user"] == pytest.approx(0.793, abs=0.02) and m["hw_mult"] == pytest.approx(0.808, abs=0.015) and m["snaive_k"] == pytest.approx(0.844, abs=0.006) and a.oracle["mase"] == pytest.approx(0.616, abs=0.01)


@pytest.fixture(scope="module")
def twelve():
    return [E.analyse(E.Settings(seed=s)) for s in C.EXP_SEEDS]


def test_default_model_over_twelve_series(twelve):
    mean = {k: float(np.mean([a.summary[k]["mase"] for a in twelve])) for k in C.METHODS}
    assert mean["user"] == pytest.approx(0.879, abs=0.02) and mean["hw_mult"] == pytest.approx(0.880, abs=0.02) and abs(mean["user"] - mean["hw_mult"]) < 0.02
    assert mean["snaive_k"] == pytest.approx(0.948, abs=0.02) and mean["naive"] == pytest.approx(2.73, abs=0.03) and mean["snaive"] == pytest.approx(1.165, abs=0.02) and mean["ses"] == pytest.approx(2.068, abs=0.03)
    assert float(np.mean([a.oracle["mase"] for a in twelve])) == pytest.approx(0.735, abs=0.01) and all(a.summary["user"]["mase"] < a.summary["snaive_k"]["mase"] for a in twelve)
    gaps = [abs(a.summary["arima011"]["mase"] - a.summary["ses"]["mase"]) for a in twelve]
    assert float(np.mean(gaps)) < 2e-3 and max(gaps) < 0.02


def test_fitted_parameters_over_twelve_series(twelve):
    models = [a.user[0] for a in twelve]
    assert float(np.mean([m.phi[0] for m in models])) == pytest.approx(0.986, abs=0.02) and float(np.mean([m.theta[0] for m in models])) == pytest.approx(-0.852, abs=0.05) and float(np.mean([m.Theta[0] for m in models])) == pytest.approx(-0.937, abs=0.03)
    assert all(abs(m.Phi[0]) < 0.15 for m in models)


def test_ljung_box_does_not_see_the_events():
    def share(events):
        ps = [E.diagnostics(E.analyse(E.Settings(seed=s, events=events), ("user",)))["lb_p"] for s in C.EXP_SEEDS]
        return float(np.mean(np.array(ps) < 0.05)), float(np.median(ps))
    (lo, med_lo), (hi, med_hi) = share(0.0), share(1.0)
    assert lo <= 0.34 and hi <= 0.34 and abs(lo - hi) <= 0.17 and med_lo > 0.05 and med_hi > 0.05


def test_acf_signatures():
    raw = E.diagnostics(E.analyse(E.Settings(p=0, d=0, q=0, P=0, D=0, Q=0, log=False), ("user",)))
    assert raw["acf_z"][[0, 6, 13, 20, 27]] == pytest.approx([0.23, 0.81, 0.80, 0.80, 0.77], abs=0.04)
    d1 = E.diagnostics(E.analyse(E.Settings(p=0, d=0, q=0, P=0, D=1, Q=0, log=True), ("user",)))
    assert d1["acf_w"][[0, 6, 13]] == pytest.approx([0.12, -0.48, -0.04], abs=0.05) and d1["pacf_w"][[6, 13]] == pytest.approx([-0.48, -0.34], abs=0.05)


def test_equivalence_experiment():
    ses, holt, damped = E.equivalence_experiment()
    assert ses["ets"] == pytest.approx(2.068, abs=0.01) and ses["arima"] == pytest.approx(2.068, abs=0.01) and abs(ses["ets"] - ses["arima"]) < 1e-3 and ses["max_diff"] < 1e-8
    assert holt["ets"] == pytest.approx(2.052, abs=0.02) and holt["arima"] == pytest.approx(2.098, abs=0.03) and holt["arima"] - holt["ets"] == pytest.approx(0.046, abs=0.03)
    assert damped["ets"] == pytest.approx(2.044, abs=0.02) and damped["arima"] == pytest.approx(2.063, abs=0.03) and damped["arima"] > damped["ets"]
    assert holt["theta_ets"] == pytest.approx(0.99, abs=0.01) and holt["theta_arima"] == pytest.approx(0.95, abs=0.03)


def test_spec_experiment():
    r = E.spec_experiment()
    by = {x["label"]: x["mase"] for x in r["rows"]}
    labels = [x["label"] for x in r["rows"]]
    assert [round(by[lab], 2) for lab in labels] == pytest.approx([0.93, 0.89, 0.89, 0.88, 0.91, 0.88], abs=0.02) and r["hw_mult"] == pytest.approx(0.880, abs=0.01)
    assert by[labels[0]] > by[labels[1]] + 0.03 and by[labels[4]] > by[labels[5]] + 0.03 and by[labels[3]] <= by[labels[2]] + 0.003
    lo, hi = r["bias"]
    assert lo["mase_mean"] == pytest.approx(0.878, abs=0.015) and abs(lo["mase_mean"] - lo["mase_median"]) < 0.003
    assert hi["mase_mean"] == pytest.approx(0.856, abs=0.015) and hi["mase_median"] == pytest.approx(0.846, abs=0.015) and hi["mase_median"] < hi["mase_mean"]
    assert hi["rmse_mean"] == pytest.approx(59.0, abs=1.5) and hi["rmse_median"] == pytest.approx(60.3, abs=1.5) and hi["rmse_mean"] < hi["rmse_median"]


def test_events_experiment():
    rows = {r["events"]: r for r in E.events_experiment()}
    assert [round(x, 2) for x in rows[1.0]["share"]] == pytest.approx([0.09, 0.27, 0.64], abs=0.01)
    hi, lo = rows[1.0], rows[0.0]
    assert hi["hw_mult"] == pytest.approx([54.2, 20.3, 15.5], abs=0.5) and hi["ar_log"] == pytest.approx([54.0, 18.2, 15.9], abs=0.5) and hi["diff_log"] == pytest.approx([54.2, 19.3, 15.8], abs=0.5)
    assert hi["snaive_k"] == pytest.approx([55.0, 18.8, 17.8], abs=0.5) and hi["oracle"] == pytest.approx([17.3, 13.7, 13.9], abs=0.3) and hi["ar_log"][0] / hi["oracle"][0] == pytest.approx(3.1, abs=0.15)
    assert hi["ar_log"][1] < hi["diff_log"][1] < hi["hw_mult"][1] and hi["ar_log"][2] > hi["hw_mult"][2]
    assert lo["hw_mult"] == pytest.approx([14.9, 14.5, 14.8], abs=0.4) and lo["ar_log"] == pytest.approx([15.0, 14.7, 15.1], abs=0.4)


def test_events_mean_advantage_over_holt_winters():
    rows = [E.analyse(E.Settings(seed=s, events=1.0), ("user", "hw_mult")) for s in C.EXP_SEEDS]
    mu, mh = (float(np.mean([a.summary[k]["mase"] for a in rows])) for k in ("user", "hw_mult"))
    assert mu == pytest.approx(0.882, abs=0.02) and mh == pytest.approx(0.895, abs=0.02) and mu < mh and sum(a.summary["user"]["mase"] < a.summary["hw_mult"]["mase"] for a in rows) >= 9
    assert float(np.mean([a.user[0].phi[0] for a in rows])) == pytest.approx(0.956, abs=0.03)


def test_auto_experiment():
    r = E.auto_experiment()
    assert r["grid_size"] == 24 and r["selected"][0] == pytest.approx(0.880, abs=0.02) and r["best"] == pytest.approx(0.874, abs=0.02) and r["gap"] == pytest.approx(0.006, abs=0.01) and r["airline"][0] == pytest.approx(0.891, abs=0.02) and r["hw_mult"][0] == pytest.approx(0.880, abs=0.02)
    assert len(set(r["picks"])) >= 5 and 3 <= sum(sp.d == 0 for sp in r["picks"]) <= 9 and r["selected"][0] < r["airline"][0]


def test_burn_in_improves_the_conditional_estimate():
    import ari_forecast as F
    spec = SAR.Spec(0, 1, 1, 0, 1, 1)
    res = {0: [], 200: []}
    for sd in range(6):
        y = E._series(E.Settings(seed=sd).series_key).y
        org = F.origins(len(y), 14)
        act = np.stack([y[t:t + 14] for t in org])
        for burn in res:
            m = SAR.fit(y[:C.FIRST_TEST], spec, burn=burn)
            res[burn].append(np.abs(SAR.forecast_origins(m, SAR.filter_series(m, y), org, 14) - act).mean() / F.mase_scale(y, C.FIRST_TEST))
    assert float(np.mean(res[0])) == pytest.approx(0.908, abs=0.02) and float(np.mean(res[200])) == pytest.approx(0.886, abs=0.02) and np.mean(res[200]) < np.mean(res[0])
