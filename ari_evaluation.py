"""Auswertung: ein frei wählbares ARIMA-Modell im Rolling-Origin-Vergleich mit den Sonderfällen (naiv, saisonal naiv, ARIMA(0,1,1)), den Glättungsmodellen aus Stück 2 und dem Wochenmittel; automatische Ordnungswahl
nach AICc und vier Experimente (Sonderfälle, Log und Differenzieren, Ereignisse, Ordnungswahl)."""

import dataclasses
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import ari_constants as C
import ari_diagnostics as D
import ari_ets as ETS
import ari_forecast as F
import ari_sarima as SAR
import ari_scenario as S

SPEC_NAIVE = SAR.Spec(0, 1, 0)
SPEC_SNAIVE = SAR.Spec(0, 0, 0, 0, 1, 0)
SPEC_011 = SAR.Spec(0, 1, 1)


@dataclass(frozen=True)
class Settings:
    trend: int = C.DEFAULT_TREND
    weekly: float = C.DEFAULT_WEEKLY
    yearly: float = C.DEFAULT_YEARLY
    noise: float = C.DEFAULT_NOISE
    shift: int = C.DEFAULT_SHIFT
    events: float = C.DEFAULT_EVENTS
    horizon: int = C.DEFAULT_HORIZON
    step: int = C.DEFAULT_STEP
    k: int = C.DEFAULT_K_WEEKS
    seed: int = 3
    p: int = C.DEFAULT_SPEC[0]
    d: int = C.DEFAULT_SPEC[1]
    q: int = C.DEFAULT_SPEC[2]
    P: int = C.DEFAULT_SPEC[3]
    D: int = C.DEFAULT_SPEC[4]
    Q: int = C.DEFAULT_SPEC[5]
    log: bool = C.DEFAULT_SPEC[6]

    @property
    def spec(self):
        return SAR.Spec(self.p, self.d, self.q, self.P, self.D, self.Q, self.log)

    @property
    def series_key(self):
        return (self.trend, self.weekly, self.yearly, self.noise, self.shift, self.events, self.seed)


@dataclass
class Analysis:
    settings: Settings
    series: S.Series
    user: tuple                # (Modell, Filtered) des gewählten ARIMA-Modells
    ets: dict                  # Schlüssel -> (Modell, Zustände)
    origins: np.ndarray
    actual: np.ndarray
    errors: dict
    summary: dict
    oracle: dict
    horizon_mae: dict
    origin_mae: dict

    @property
    def best(self):
        return min(self.summary, key=lambda m: self.summary[m]["mase"])

    @property
    def train_resid(self):
        """Ein-Schritt-Fehler des gewählten Modells auf den Trainingstagen (ohne Burn-in), in der Skala der modellierten Reihe (bei log: Log-Skala)."""
        model, filt = self.user
        lo = max(SAR.BURN - model.spec.lost, model.spec.p + SAR.M * model.spec.P)
        hi = C.FIRST_TEST - model.spec.lost
        return filt.e[lo:hi]


def make_series(key):
    trend, weekly, yearly, noise, shift, events, seed = key
    return S.generate(trend, weekly, yearly, noise, shift, events, seed=seed)


@lru_cache(maxsize=128)
def _series(key):
    return make_series(key)


@lru_cache(maxsize=2048)
def _fit_arima(key, spec):
    y = _series(key).y
    model = SAR.fit(y[:C.FIRST_TEST], spec)
    return model, SAR.filter_series(model, y)


@lru_cache(maxsize=1024)
def _fit_ets(key, ets_key):
    y = _series(key).y
    model = ETS.fit(y[:C.FIRST_TEST], ets_key)
    return model, ETS.filter_states(model, y)


def oracle_summary(series, org, h):
    """Kennzahlen der Orakel-Prognose: der wahre Erwartungswert als Prognose, gemessen am beobachteten Wert (untere Grenze für jedes Verfahren im Mittel)."""
    E = np.stack([series.mu[t:t + h] - series.y[t:t + h] for t in org])
    scale = F.mase_scale(series.y, C.FIRST_TEST)
    mae = float(np.abs(E).mean())
    return {"mae": mae, "rmse": float(np.sqrt((E ** 2).mean())), "me": float(E.mean()), "mase": mae / scale}


def analyse(s, methods=C.METHODS):
    series = _series(s.series_key)
    y = series.y
    org = F.origins(series.n, s.horizon, step=s.step)
    actual = np.stack([y[t:t + s.horizon] for t in org])
    user = _fit_arima(s.series_key, s.spec)
    ets = {k: _fit_ets(s.series_key, k) for k in ("ses", "hw_mult") if k in methods}
    errors = {}
    for m in methods:
        if m == "user":
            f = F.arima_forecast(user, org, s.horizon)
        elif m == "naive":
            f = F.arima_forecast(_fit_arima(s.series_key, SPEC_NAIVE), org, s.horizon)
        elif m == "snaive":
            f = F.arima_forecast(_fit_arima(s.series_key, SPEC_SNAIVE), org, s.horizon)
        elif m == "arima011":
            f = F.arima_forecast(_fit_arima(s.series_key, SPEC_011), org, s.horizon)
        elif m in ("ses", "hw_mult"):
            f = F.ets_forecast(ets[m], org, s.horizon)
        elif m == "snaive_k":
            f = F.snaive_k_origins(y, org, s.horizon, s.k)
        else:
            raise ValueError(m)
        errors[m] = f - actual
    return Analysis(s, series, user, ets, org, actual, errors, F.summarize(errors, y), oracle_summary(series, org, s.horizon), F.per_horizon(errors), F.per_origin(errors))


def _mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


def _replace(base, **kw):
    d = dict(base.__dict__)
    d.update(kw)
    return Settings(**d)


def _spec_kw(spec):
    return dict(p=spec.p, d=spec.d, q=spec.q, P=spec.P, D=spec.D, Q=spec.Q, log=spec.log)


def spec_mase(base, spec, seed, methods=("user",), **kw):
    """MASE eines ARIMA-Modells auf der Reihe mit dem angegebenen Seed (Einstellungen der Reihe aus base)."""
    a = analyse(_replace(base, seed=seed, **_spec_kw(spec), **kw), methods)
    return a.summary["user"]["mase"]


# --- Automatische Ordnungswahl ------------------------------------------------------------------------------------------------------------------


def auto_grid():
    return [SAR.Spec(p, d, q, P, 1, 1, True) for d in C.AUTO_D for p in C.AUTO_P for q in C.AUTO_Q for P in C.AUTO_SP]


def auto_search(s):
    """Alle Ordnungen des Gitters (log, D = 1, Q = 1) auf den Trainingstagen der Reihe schätzen und nach AICc ordnen: [(Spec, Modell, AICc)]. Die Fehler werden auf denselben Tagen der Reihe summiert, die AICc-Werte sind daher
    auch zwischen verschiedenen Differenzierungen vergleichbar."""
    rows = []
    for spec in auto_grid():
        model, _ = _fit_arima(s.series_key, spec)
        rows.append((spec, model, model.aicc))
    return sorted(rows, key=lambda r: r[2])


# --- Experiment 1: Sonderfälle -------------------------------------------------------------------------------------------------------------------


def ets_to_arima(model, key):
    """Parameter des gleichwertigen ARIMA-Modells zu einem geschätzten Glättungsmodell (Hyndman et al. 2008): einfache Glättung -> (0,1,1), Holt -> (0,2,2), gedämpft -> (1,1,2); (phi, theta) als Tupel."""
    a, b, ph = model.alpha, model.beta, model.phi
    if key == "ses":
        return (), (a - 1.0,)
    if key == "holt":
        return (), (a + b - 2.0, 1.0 - a)
    if key == "damped":
        return (ph,), (a + ph * b - 1.0 - ph, (1.0 - a) * ph)
    raise ValueError(key)


def equivalence_experiment(seeds=None, base=None):
    """Glättungsmodell gegen das gleichwertige ARIMA-Modell (Ordnung nach C.EQUIVALENCES), beide unabhängig geschätzt; dazu der größte Prognoseunterschied, wenn die ARIMA-Parameter aus den geschätzten Glättungsparametern umgerechnet werden."""
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for key, (p, d, q) in C.EQUIVALENCES:
        spec = SAR.Spec(p, d, q)
        m_ets, m_ari, diff, th_ets, th_ari = [], [], [], [], []
        for sd in seeds:
            st = _replace(base, seed=sd, **_spec_kw(spec))
            series = _series(st.series_key)
            org = F.origins(series.n, base.horizon, step=base.step)
            act = np.stack([series.y[t:t + base.horizon] for t in org])
            scale = F.mase_scale(series.y, C.FIRST_TEST)
            model, states = _fit_ets(st.series_key, key)
            f_ets = ETS.forecast_origins(model, states, org, base.horizon)
            m_ets.append(np.abs(f_ets - act).mean() / scale)
            m_ari.append(analyse(st, ("user",)).summary["user"]["mase"])
            phi, theta = ets_to_arima(model, key)
            conv = SAR.Model(spec, phi, (), theta, (), 0.0, 1.0, 100, 1.0)
            th_ets.append(theta[-1])
            th_ari.append(_fit_arima(st.series_key, spec)[0].theta[-1])
            f_conv = SAR.forecast_origins(conv, SAR.filter_series(conv, series.y), org, base.horizon)
            diff.append(float(np.abs(f_conv - f_ets).max()))
        rows.append({"key": key, "spec": spec, "n_seeds": len(seeds), "ets": float(np.mean(m_ets)), "arima": float(np.mean(m_ari)), "max_diff": max(diff), "median_diff": float(np.median(diff)), "theta_ets": float(np.mean(th_ets)), "theta_arima": float(np.mean(th_ari))})
    return rows


# --- Experiment 2: Log und Differenzieren --------------------------------------------------------------------------------------------------------


SPEC_TABLE = (SAR.Spec(0, 1, 1, 0, 1, 1), SAR.Spec(0, 1, 1, 0, 1, 1, log=True), SAR.Spec(1, 1, 1, 0, 1, 1, log=True), SAR.Spec(1, 0, 1, 0, 1, 1, log=True), SAR.Spec(1, 0, 1, 1, 1, 1), SAR.Spec(1, 0, 1, 1, 1, 1, log=True))


def spec_experiment(seeds=None, base=None, specs=SPEC_TABLE, noises=None):
    """MASE einiger ARIMA-Ordnungen (mit und ohne Log) im Vergleich mit Holt-Winters multiplikativ; dazu für das Modell (1,0,1)(1,1,1) im Log die Rücktransformation: Mittelwert (exp(f + s²/2)) gegen Median (exp(f)) in MAE und RMSE."""
    seeds = C.EXP_SEEDS if seeds is None else seeds
    noises = C.BIAS_NOISE if noises is None else noises
    base = Settings() if base is None else base
    res = {sp.label: [] for sp in specs}
    hw = []
    for sd in seeds:
        for sp in specs:
            res[sp.label].append(spec_mase(base, sp, sd))
        hw.append(analyse(_replace(base, seed=sd), ("hw_mult",)).summary["hw_mult"]["mase"])
    rows = [{"spec": sp, "label": sp.label, **dict(zip(("mase", "se"), _mean_se(res[sp.label])))} for sp in specs]
    bias = []
    ref = SAR.Spec(1, 0, 1, 1, 1, 1, True)
    for nz in noises:
        mean_m, med_m, mean_r, med_r = [], [], [], []
        for sd in seeds:
            st = _replace(base, seed=sd, noise=nz, **_spec_kw(ref))
            series = _series(st.series_key)
            model, filt = _fit_arima(st.series_key, ref)
            org = F.origins(series.n, base.horizon, step=base.step)
            act = np.stack([series.y[t:t + base.horizon] for t in org])
            scale = F.mase_scale(series.y, C.FIRST_TEST)
            f1, f0 = SAR.forecast_origins(model, filt, org, base.horizon, True), SAR.forecast_origins(model, filt, org, base.horizon, False)
            mean_m.append(np.abs(f1 - act).mean() / scale)
            med_m.append(np.abs(f0 - act).mean() / scale)
            mean_r.append(float(np.sqrt(((f1 - act) ** 2).mean())))
            med_r.append(float(np.sqrt(((f0 - act) ** 2).mean())))
        bias.append({"noise": nz, "mase_mean": float(np.mean(mean_m)), "mase_median": float(np.mean(med_m)), "rmse_mean": float(np.mean(mean_r)), "rmse_median": float(np.mean(med_r))})
    return {"rows": rows, "hw_mult": _mean_se(hw)[0], "bias": bias, "n_seeds": len(seeds)}


# --- Experiment 3: Ereignisse --------------------------------------------------------------------------------------------------------------------


def event_classes(series):
    """0 = Ereignistag (Feiertag, Tag danach, Aktion), 1 = höchstens 14 Tage nach dem letzten Ereignistag, 2 = sonst."""
    ev = (series.holiday + np.roll(series.holiday, 1) + series.promo) > 0
    cls = np.full(series.n, 2)
    last = -10 ** 6
    for d in range(series.n):
        if ev[d]:
            last = d
            cls[d] = 0
        elif d - last <= C.EVENT_AFTER_DAYS:
            cls[d] = 1
    return cls


SPEC_AR = SAR.Spec(1, 0, 1, 1, 1, 1, True)              # stationäre ARMA-Fehler um die Saisondifferenz
SPEC_DIFF = SAR.Spec(1, 1, 1, 1, 1, 1, True)            # zusätzlich ein Trend-Differenzieren (Niveau als Irrfahrt)
EVENT_METHODS = ("hw_mult", "ar_log", "diff_log", "snaive_k")


def events_experiment(levels=None, seeds=None, base=None):
    """Fehler (MAE in Aufträgen) an Ereignistagen, in den Tagen danach und sonst - für Holt-Winters, SARIMA(1,0,1)(1,1,1) im Log, dasselbe mit d = 1 und das Wochenmittel."""
    levels = C.EVENT_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for ev in levels:
        acc = {m: [[], [], []] for m in EVENT_METHODS}
        orc = [[], [], []]
        share = []
        for sd in seeds:
            st = _replace(base, events=ev, seed=sd)
            a = analyse(_replace(st, **_spec_kw(SPEC_AR)), ("user", "hw_mult", "snaive_k"))
            b = analyse(_replace(st, **_spec_kw(SPEC_DIFF)), ("user",))
            errs = {"hw_mult": a.errors["hw_mult"], "ar_log": a.errors["user"], "diff_log": b.errors["user"], "snaive_k": a.errors["snaive_k"]}
            days = a.origins[:, None] + np.arange(base.horizon)[None, :]
            cm = event_classes(a.series)[days]
            oe = np.abs(a.series.mu[days] - a.series.y[days])
            share.append([float(np.mean(cm == c)) for c in range(3)])
            for c in range(3):
                orc[c].append(float(oe[cm == c].mean()))
                for m in EVENT_METHODS:
                    acc[m][c].append(float(np.abs(errs[m])[cm == c].mean()))
        row = {"events": ev, "n_seeds": len(seeds), "share": list(np.mean(share, axis=0)), "oracle": [float(np.mean(v)) for v in orc]}
        for m in EVENT_METHODS:
            row[m] = [float(np.mean(v)) for v in acc[m]]
        rows.append(row)
    return rows


# --- Experiment 4: Ordnungswahl ------------------------------------------------------------------------------------------------------------------


def auto_experiment(seeds=None, base=None):
    """Je Reihe alle Ordnungen des Gitters nach AICc ordnen; MASE der gewählten Ordnung gegen die im Nachhinein beste Ordnung des Gitters, das Airline-Modell im Log und Holt-Winters multiplikativ."""
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    sel, best, airline, hw, picks = [], [], [], [], []
    for sd in seeds:
        st = _replace(base, seed=sd)
        rows = auto_search(st)
        mases = {r[0]: spec_mase(base, r[0], sd) for r in rows}
        pick = rows[0][0]
        picks.append(pick)
        sel.append(mases[pick])
        best.append(min(mases.values()))
        airline.append(mases[SAR.Spec(0, 1, 1, 0, 1, 1, True)] if SAR.Spec(0, 1, 1, 0, 1, 1, True) in mases else spec_mase(base, SAR.Spec(0, 1, 1, 0, 1, 1, True), sd))
        hw.append(analyse(st, ("hw_mult",)).summary["hw_mult"]["mase"])
    return {"n_seeds": len(seeds), "grid_size": len(auto_grid()), "picks": picks, "selected": _mean_se(sel), "best": float(np.mean(best)), "airline": _mean_se(airline), "hw_mult": _mean_se(hw), "gap": float(np.mean(np.array(sel) - np.array(best)))}


# --- Diagnose -----------------------------------------------------------------------------------------------------------------------------------


def diagnostics(a):
    """ACF/PACF der Reihe, der differenzierten Reihe und der Fehler des gewählten Modells (Trainingstage) sowie der Ljung-Box-Test der Fehler."""
    model, _ = a.user
    spec = model.spec
    y_train = a.series.y[:C.FIRST_TEST]
    z = SAR._transform(spec, y_train)
    w = SAR.difference(z, spec.d, spec.D)
    resid = a.train_resid
    lb_q, lb_p = D.ljung_box(resid, C.LJUNG_LAGS, model.spec.n_arma)
    return {"z": z, "w": w, "resid": resid, "acf_z": D.acf(z, C.ACF_LAGS), "acf_w": D.acf(w, C.ACF_LAGS), "pacf_w": D.pacf(w, C.ACF_LAGS), "acf_res": D.acf(resid, C.ACF_LAGS), "lb_q": lb_q, "lb_p": lb_p}
