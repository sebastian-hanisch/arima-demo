"""Plotly-Abbildungen der ARIMA-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ari_constants as C
import ari_forecast as F

COLORS = {"user": "#c2185b", "naive": "#7f7f7f", "snaive": "#4c78a8", "ses": "#cab2d6", "arima011": "#8e6bbf", "hw_mult": "#e6550d", "snaive_k": "#17becf", "ar_log": "#c2185b", "diff_log": "#f06292"}
SHORT = {"user": "Ihr Modell", "naive": "Naiv (0,1,0)", "snaive": "Saisonal naiv (0,0,0)(0,1,0)", "ses": "Einfache Glättung", "arima011": "ARIMA(0,1,1)", "hw_mult": "Holt-Winters mult.", "snaive_k": "Wochenmittel"}
ACTUAL = "#14233B"
ORACLE = "#54a24b"
WARN = "#f58518"
CLASS_NAMES = ("Ereignistag", f"bis {C.EVENT_AFTER_DAYS} Tage danach", "sonst")


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def method_label(m, k, spec_label=None):
    if m == "user":
        return f"Ihr Modell: ARIMA{spec_label}" if spec_label else C.METHOD_NAMES[m]
    if m == "snaive_k":
        return f"Wochenmittel der letzten {k} Wochen (Stück 1)"
    return C.METHOD_NAMES[m]


def build_series(a, origin):
    """Die ganze Reihe (drei Jahre) mit Erwartungswert, dem Testjahr und dem gewählten Ursprung."""
    s = a.series
    t = np.arange(s.n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=s.y, mode="lines", name="Tagesaufträge", line=dict(color=ACTUAL, width=1)))
    fig.add_trace(go.Scatter(x=t, y=s.mu, mode="lines", name="Erwartungswert (ohne Rauschen)", line=dict(color=ORACLE, width=1.5, dash="dot")))
    fig.add_vrect(x0=C.FIRST_TEST, x1=s.n, fillcolor="rgba(245,133,24,0.08)", line_width=0, annotation_text="Testjahr (Ursprünge)", annotation_position="top left")
    fig.add_vline(x=origin, line=dict(color=WARN, dash="dash"))
    if s.shift_day >= 0:
        fig.add_vline(x=s.shift_day, line=dict(color="#e45756", dash="dot"), annotation_text="Niveausprung", annotation_position="bottom right")
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Aufträge je Tag", rangemode="tozero")
    return _base(fig, 300)


def build_origin(a, origin):
    """Die 42 Tage vor dem Ursprung und die nächsten h Tage: Ist, Erwartungswert, das gewählte ARIMA-Modell und Vergleichsverfahren."""
    s, st = a.series, a.settings
    h = st.horizon
    org = np.array([origin])
    x_hist = np.arange(origin - 42, origin)
    x_fut = np.arange(origin, origin + h)
    spec_label = a.user[0].spec.label
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_hist, y=s.y[origin - 42:origin], mode="lines+markers", name="bekannt", line=dict(color=ACTUAL, width=1.5), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=x_fut, y=s.y[origin:origin + h], mode="lines+markers", name="tatsächlich", line=dict(color=ACTUAL, width=2), marker=dict(size=6, symbol="circle-open")))
    fig.add_trace(go.Scatter(x=x_fut, y=s.mu[origin:origin + h], mode="lines", name="Erwartungswert", line=dict(color=ORACLE, width=2, dash="dot")))
    fc = {"user": F.arima_forecast(a.user, org, h)[0], "hw_mult": F.ets_forecast(a.ets["hw_mult"], org, h)[0], "snaive_k": F.snaive_k_origins(s.y, org, h, st.k)[0], "ses": F.ets_forecast(a.ets["ses"], org, h)[0]}
    for m, f in fc.items():
        fig.add_trace(go.Scatter(x=x_fut, y=f, mode="lines", name=method_label(m, st.k, spec_label), line=dict(color=COLORS[m], width=3 if m == "user" else 1.4, dash="solid" if m == "user" else "dash")))
    fig.add_vline(x=origin - 0.5, line=dict(color=WARN, dash="dash"))
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Aufträge je Tag", rangemode="tozero")
    return _base(fig, 360).update_layout(legend=dict(orientation="h", y=-0.35))


def build_acf(values, title, n, color="#4c78a8", partial=False):
    """Balkendiagramm der (partiellen) Autokorrelationen mit dem 95-%-Band 1,96 / sqrt(n); Lags 7, 14, 21, 28 sind hervorgehoben."""
    lags = np.arange(1, len(values) + 1)
    colors = [WARN if lag % 7 == 0 else color for lag in lags]
    fig = go.Figure(go.Bar(x=lags, y=values, marker=dict(color=colors), showlegend=False))
    band = 1.96 / np.sqrt(n)
    fig.add_hline(y=band, line=dict(color="#7f7f7f", dash="dot"))
    fig.add_hline(y=-band, line=dict(color="#7f7f7f", dash="dot"))
    fig.update_xaxes(title_text="Lag (Tage)", dtick=7)
    fig.update_yaxes(title_text=("PACF" if partial else "ACF"), range=[-1.05, 1.05])
    fig.update_layout(title=dict(text=title, font=dict(size=13)))
    return _base(fig, 260)


def build_bars(a):
    """MASE je Verfahren über alle Ursprünge und Horizonte, dazu die Orakel-Untergrenze."""
    ms = sorted(a.summary, key=lambda m: a.summary[m]["mase"])
    fig = go.Figure(go.Bar(x=[SHORT[m] for m in ms], y=[a.summary[m]["mase"] for m in ms], marker=dict(color=[COLORS[m] for m in ms]),
                           text=[f"{a.summary[m]['mase']:.2f}".replace(".", ",") for m in ms], textposition="outside", showlegend=False))
    fig.add_hline(y=a.oracle["mase"], line=dict(color=ORACLE, dash="dot"), annotation_text="Orakel (wahrer Erwartungswert)", annotation_position="top right")
    fig.add_hline(y=1.0, line=dict(color="#7f7f7f", dash="dash"), annotation_text="MASE 1 = saisonal naiv im Training", annotation_position="bottom right")
    fig.update_yaxes(title_text="MASE (kleiner ist besser)", rangemode="tozero")
    return _base(fig, 380)


def build_horizon(a):
    scale = F.mase_scale(a.series.y, C.FIRST_TEST)
    xs = list(range(1, a.settings.horizon + 1))
    fig = go.Figure()
    for m in a.summary:
        fig.add_trace(go.Scatter(x=xs, y=a.horizon_mae[m] / scale, mode="lines+markers", name=method_label(m, a.settings.k, a.user[0].spec.label), line=dict(color=COLORS[m], width=2.5 if m in ("user", "hw_mult") else 1.3), marker=dict(size=4)))
    fig.update_xaxes(title_text="Prognosehorizont (Tage)", dtick=1 if a.settings.horizon <= 14 else 2)
    fig.update_yaxes(title_text="MASE je Horizont", rangemode="tozero")
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.35))


# --- Experimente ------------------------------------------------------------------------------------------------------------------------------


def build_equivalence(rows):
    labels = [f"{r['key']} / ARIMA{r['spec'].label}".replace("ses", "Einfache Glättung").replace("holt", "Holt").replace("damped", "Holt gedämpft") for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["ets"] for r in rows], name="Glättungsmodell (Stück 2)", marker=dict(color=COLORS["ses"]), text=[f"{r['ets']:.3f}".replace(".", ",") for r in rows], textposition="outside"))
    fig.add_trace(go.Bar(x=labels, y=[r["arima"] for r in rows], name="gleichwertiges ARIMA-Modell", marker=dict(color=COLORS["user"]), text=[f"{r['arima']:.3f}".replace(".", ",") for r in rows], textposition="outside"))
    fig.update_yaxes(title_text="MASE (kleiner ist besser)", rangemode="tozero")
    fig.update_layout(barmode="group")
    return _base(fig, 340)


def build_specs(res):
    rows = res["rows"]
    labels = [r["label"] for r in rows] + ["Holt-Winters mult. (Stück 2)"]
    vals = [r["mase"] for r in rows] + [res["hw_mult"]]
    errs = [r["se"] for r in rows] + [0.0]
    colors = [COLORS["user"] if r["spec"].log else "#e8a0bd" for r in rows] + [COLORS["hw_mult"]]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], subplot_titles=("MASE einiger Ordnungen (dunkel: im Log)", "Rücktransformation aus dem Log"))
    fig.add_trace(go.Bar(x=labels, y=vals, error_y=dict(type="data", array=errs), marker=dict(color=colors), text=[f"{v:.3f}".replace(".", ",") for v in vals], textposition="outside", showlegend=False), row=1, col=1)
    xs = [f"Rauschen {b['noise']:.2f}".replace(".", ",") for b in res["bias"]]
    fig.add_trace(go.Bar(x=xs, y=[b["mase_mean"] for b in res["bias"]], name="Mittelwert exp(f + s²/2): MASE", marker=dict(color=COLORS["user"])), row=1, col=2)
    fig.add_trace(go.Bar(x=xs, y=[b["mase_median"] for b in res["bias"]], name="Median exp(f): MASE", marker=dict(color="#e8a0bd")), row=1, col=2)
    fig.update_yaxes(title_text="MASE (kleiner ist besser)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(rangemode="tozero", row=1, col=2)
    fig.update_layout(barmode="group")
    return _base(fig, 400).update_layout(legend=dict(orientation="h", y=-0.45))


def build_events(rows):
    fig = make_subplots(rows=1, cols=len(rows), shared_yaxes=True, subplot_titles=[f"Ereignisstärke {r['events']:.2f}".replace(".", ",") for r in rows])
    series = (("Orakel (Rauschen)", "oracle", ORACLE), ("Holt-Winters mult.", "hw_mult", COLORS["hw_mult"]), ("SARIMA (1,0,1)(1,1,1) log", "ar_log", COLORS["ar_log"]), ("SARIMA (1,1,1)(1,1,1) log", "diff_log", COLORS["diff_log"]), ("Wochenmittel", "snaive_k", COLORS["snaive_k"]))
    for i, r in enumerate(rows, start=1):
        for name, key, color in series:
            vals = r[key]
            fig.add_trace(go.Bar(x=list(CLASS_NAMES), y=vals, name=name, marker=dict(color=color), showlegend=(i == 1)), row=1, col=i)
    fig.update_yaxes(title_text="mittlerer absoluter Fehler (Aufträge je Tag)", rangemode="tozero", col=1)
    fig.update_layout(barmode="group")
    return _base(fig, 400).update_layout(legend=dict(orientation="h", y=-0.3))


def build_auto(res):
    labels = ["nach AICc gewählt", "im Nachhinein beste Ordnung des Gitters", "Airline-Modell (0,1,1)(0,1,1) log", "Holt-Winters mult. (Stück 2)"]
    vals = [res["selected"][0], res["best"], res["airline"][0], res["hw_mult"][0]]
    errs = [res["selected"][1], 0.0, res["airline"][1], res["hw_mult"][1]]
    fig = go.Figure(go.Bar(x=labels, y=vals, error_y=dict(type="data", array=errs), marker=dict(color=[COLORS["user"], "#e8a0bd", COLORS["arima011"], COLORS["hw_mult"]]), text=[f"{v:.3f}".replace(".", ",") for v in vals], textposition="outside", showlegend=False))
    fig.update_yaxes(title_text="MASE (kleiner ist besser)", rangemode="tozero")
    return _base(fig, 340)
