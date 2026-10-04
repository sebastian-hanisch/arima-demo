"""Zufällige Instanzen gegen unabhängige Orakel (ergänzt die festen Fälle aus test_sarima.py / test_diagnostics.py).

- SARIMA-Prognosen (zufällige Ordnungen, Parameter, Ursprünge, Horizonte, mit und ohne Log) gegen statsmodels bei festen Parametern.
- Bedingte Fehlerrekursion (`_css`) gegen eine Schleife mit `np.convolve` als Polynomprodukt (ohne statsmodels).
- Glättungskern (`ari_ets`, alle sechs Modelltypen) gegen eine zustandsweise Python-Schleife (ohne statsmodels) und gegen statsmodels `ETSModel` (feste Parameter und Anfangszustände).
- ACF, PACF, Ljung-Box und Chi-Quadrat-Verteilung gegen statsmodels/scipy auf zufälligen Reihen.
"""

import warnings

import numpy as np
import pytest

import ari_constants as C
import ari_diagnostics as D
import ari_ets as ETS
import ari_sarima as SAR
import ari_scenario as S

_SERIES = None


def _series():
    global _SERIES
    if _SERIES is None:
        _SERIES = [S.generate(seed=sd).y.astype(float) for sd in range(1, 5)]
    return _SERIES


def _stable(rng, k, sign):
    """Zufällige Koeffizienten mit allen Wurzeln deutlich außerhalb des Einheitskreises (Anfangsstörung der bedingten Rekursion klingt ab)."""
    while True:
        c = rng.uniform(-0.6, 0.6, size=k)
        if k == 0 or np.all(np.abs(np.roots(np.concatenate([(sign * c)[::-1], [1.0]]))) > 1.25):
            return c


def test_sarima_forecasts_match_statsmodels_on_random_instances():
    arima = pytest.importorskip("statsmodels.tsa.arima.model").ARIMA
    rng = np.random.default_rng(2026)
    for it in range(60):
        p, q, d, P, Q, D_ = (int(rng.integers(0, 3)), int(rng.integers(0, 3)), int(rng.integers(0, 3)), int(rng.integers(0, 2)), int(rng.integers(0, 2)), int(rng.integers(0, 2)))
        if d + D_ > 2:
            D_ = 0
        spec = SAR.Spec(p, d, q, P, D_, Q, bool(rng.integers(0, 2)))
        y = _series()[it % 4][:int(rng.integers(600, 1000))]
        phi, theta, Phi, Theta = _stable(rng, p, -1.0), _stable(rng, q, 1.0), _stable(rng, P, -1.0), _stable(rng, Q, 1.0)
        z = np.log(np.maximum(y, 1.0)) if spec.log else y
        nodiff = d + D_ == 0
        mean = float(SAR.difference(z[:C.FIRST_TEST], d, D_).mean()) if nodiff else 0.0
        sigma2 = float(rng.uniform(0.01, 0.1)) if spec.log else 1.0
        model = SAR.Model(spec, tuple(phi), tuple(Phi), tuple(theta), tuple(Theta), mean, 1.0, 100, sigma2)
        t, h = int(rng.integers(max(300, spec.lost + 200), len(y) - 14)), int(rng.integers(1, 15))
        mine = SAR.forecast_origins(model, SAR.filter_series(model, y), np.array([t]), h)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = arima(z[:t], order=(p, d, q), seasonal_order=(P, D_, Q, 7), trend="c" if nodiff else "n")
            vals = {"sigma2": sigma2, "const": mean}
            vals.update({f"ar.L{i}": v for i, v in enumerate(phi, 1)})
            vals.update({f"ma.L{i}": v for i, v in enumerate(theta, 1)})
            vals.update({f"ar.S.L{7 * i}": v for i, v in enumerate(Phi, 1)})
            vals.update({f"ma.S.L{7 * i}": v for i, v in enumerate(Theta, 1)})
            ref = np.asarray(mod.smooth([vals[nm] for nm in mod.param_names]).forecast(h))
        if spec.log:
            ref = np.exp(ref + 0.5 * sigma2)                                    # Mittelwert einer Lognormalverteilung
        ref = np.maximum(ref, 0.0)
        assert np.max(np.abs(mine - ref) / np.maximum(1.0, np.abs(ref))) < 1e-6, (it, spec.label, t, h)


def test_css_recursion_matches_a_plain_loop():
    rng = np.random.default_rng(5)
    for it in range(150):
        p, q, P, Q = (int(rng.integers(0, 3)), int(rng.integers(0, 3)), int(rng.integers(0, 2)), int(rng.integers(0, 2)))
        phi, theta, Phi, Theta = (rng.uniform(-0.5, 0.5, k) for k in (p, q, P, Q))
        w = rng.normal(size=int(rng.integers(40, 200)))
        A, B = SAR.ar_ma_polys(phi[None, :], Phi[None, :], theta[None, :], Theta[None, :])
        e, sse = SAR._css(w, A, B, 0)
        ar = np.convolve(np.r_[1.0, -phi] if p else [1.0], np.r_[1.0, np.zeros(6), -Phi[0]] if P else [1.0])      # 1 - phi(B) Phi(B^7) als Polynomprodukt
        ma = np.convolve(np.r_[1.0, theta] if q else [1.0], np.r_[1.0, np.zeros(6), Theta[0]] if Q else [1.0])
        la = len(ar) - 1
        ref = np.zeros(len(w))
        for t in range(la, len(w)):
            ref[t] = sum(ar[k] * w[t - k] for k in range(len(ar))) - sum(ma[k] * ref[t - k] for k in range(1, len(ma)) if t - k >= la)
        assert np.allclose(e[0], ref, atol=1e-10), it
        assert sse[0] == pytest.approx(float(np.sum(ref[la:] ** 2)))


def _ets_loop(y, trend, season, alpha, beta, gamma, phi, init):
    """Zustandsweise Schleife (Hyndman et al. 2008, additiver Fehler); gibt die Ein-Schritt-Fehler und die Zustände vor jedem Tag zurück."""
    l, b = init[0], init[1]
    s = list(init[2]) if season != "none" else None
    errs, levels, trends, seas = [], [l], [b], [list(s) if s else None]
    for t, yt in enumerate(y):
        base = l + phi * b
        st = s[t % 7] if s else None
        yhat = base if s is None else (base + st if season == "add" else base * st)
        e = yt - yhat
        if season == "mul":
            l_new, b_new = base + alpha * e / st, phi * b + beta * e / st
            s[t % 7] = st + gamma * e / l_new
        else:
            l_new, b_new = base + alpha * e, phi * b + beta * e
            if season == "add":
                s[t % 7] = st + gamma * e
        l, b = l_new, b_new
        errs.append(e)
        levels.append(l)
        trends.append(b)
        seas.append(list(s) if s else None)
    return np.array(errs), np.array(levels), np.array(trends), seas


def _random_ets(rng, key, y):
    trend, season = C.ETS_MODELS[key]
    al = float(rng.uniform(0.02, 0.5))
    be = float(rng.uniform(0.0, al)) if trend != "none" else 0.0
    ga = float(rng.uniform(0.0, min(0.4, 1 - al))) if season != "none" else 0.0
    ph = float(rng.uniform(C.PHI_MIN, C.PHI_MAX)) if trend == "damped" else 1.0
    init = ETS.init_states(y[:int(rng.integers(200, 400))], trend, season)
    return ETS.Model(key, trend, season, al, be, ga, ph, init, 1.0, 100)


def test_ets_core_matches_a_state_loop_and_hand_forecast_formula():
    rng = np.random.default_rng(31)
    compared = 0
    for it in range(60):
        key = list(C.ETS_MODELS)[it % 6]
        y = _series()[(it // 6) % 4]
        m = _random_ets(rng, key, y)
        st = ETS.filter_states(m, y)
        if st.level.min() <= 0.5 or (m.season == "mul" and st.season.min() <= 0.05):
            continue                                                            # entartete Zustände: die Demo klemmt Nenner bei EPS, die Schleife nicht
        errs, levels, trends, seas = _ets_loop(y, m.trend, m.season, m.alpha, m.beta, m.gamma, m.phi, m.init)
        assert np.allclose(st.error, errs, atol=1e-8) and np.allclose(st.level, levels, atol=1e-8) and np.allclose(st.trend, trends, atol=1e-8), key
        t, h = int(rng.integers(300, len(y) - 14)), int(rng.integers(1, 15))
        got = ETS.forecast_origins(m, st, np.array([t]), h)[0]
        want = []
        for j in range(h):                                                      # (l + (phi + ... + phi^(j+1)) b) combined with the season slot of day t + j
            base = levels[t] + sum(m.phi ** i for i in range(1, j + 2)) * trends[t]
            if m.season != "none":
                sv = seas[t][(t + j) % 7]
                base = base + sv if m.season == "add" else base * sv
            want.append(max(base, 0.0))
        assert np.allclose(got, want, atol=1e-8), key
        compared += 1
    assert compared >= 45


def test_ets_core_matches_statsmodels_for_fixed_parameters_and_states():
    ets_model = pytest.importorskip("statsmodels.tsa.exponential_smoothing.ets").ETSModel
    rng = np.random.default_rng(32)
    compared = 0
    for it in range(60):
        key = list(C.ETS_MODELS)[it % 6]
        y = _series()[(it // 6) % 4]
        m = _random_ets(rng, key, y)
        st = ETS.filter_states(m, y)
        if st.level.min() <= 0.5 or (m.season == "mul" and st.season.min() <= 0.05):
            continue                                                            # entartete Zustände: die Demo klemmt Nenner bei EPS, statsmodels nicht
        t, h = int(rng.integers(300, len(y) - 14)), int(rng.integers(1, 15))
        kw = dict(error="add", trend=None if m.trend == "none" else "add", damped_trend=(m.trend == "damped"), seasonal=None if m.season == "none" else m.season,
                  seasonal_periods=7 if m.season != "none" else None, initialization_method="known", initial_level=m.init[0])
        if m.trend != "none":
            kw["initial_trend"] = m.init[1]
        if m.season != "none":
            kw["initial_seasonal"] = np.array(m.init[2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm = ets_model(y[:t], **kw)
            vals = {"smoothing_level": m.alpha, "smoothing_trend": m.beta, "smoothing_seasonal": m.gamma, "damping_trend": m.phi}
            res = sm.smooth([vals[nm] for nm in sm.param_names])
        assert np.allclose(res.resid, st.error[:t], atol=1e-8), key
        assert np.allclose(np.maximum(np.asarray(res.forecast(h)), 0.0), ETS.forecast_origins(m, st, np.array([t]), h)[0], atol=1e-7), key
        compared += 1
    assert compared >= 45


def test_diagnostics_match_statsmodels_and_scipy_on_random_series():
    stattools = pytest.importorskip("statsmodels.tsa.stattools")
    ljung = pytest.importorskip("statsmodels.stats.diagnostic").acorr_ljungbox
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(31)
    for it in range(80):
        n = int(rng.integers(30, 800))
        x = rng.normal(size=n).cumsum() * float(rng.choice([0, 0.3, 1])) + rng.normal(size=n)
        nl = int(rng.integers(1, min(25, n // 2)))
        assert np.max(np.abs(D.acf(x, nl) - stattools.acf(x, nlags=nl, fft=False)[1:])) < 1e-12
        assert np.max(np.abs(D.pacf(x, nl) - stattools.pacf(x, nlags=nl, method="ywm")[1:])) < 1e-10
        lags = int(rng.integers(2, min(30, n // 2) + 1))
        k = int(rng.integers(0, min(lags, 5)))
        q, pv = D.ljung_box(x, lags, k)
        ref = ljung(x, lags=[lags], model_df=k, return_df=True)
        assert q == pytest.approx(float(ref["lb_stat"].iloc[0]), rel=1e-9) and pv == pytest.approx(float(ref["lb_pvalue"].iloc[0]), abs=1e-10)
        xx, df = float(rng.uniform(0, 100)), int(rng.integers(1, 40))
        assert D.chi2_sf(xx, df) == pytest.approx(float(stats.chi2.sf(xx, df)), abs=1e-12)
