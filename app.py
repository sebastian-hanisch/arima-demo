"""ARIMA und SARIMA - das Baukasten-Gegenstück zur Glättung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Drittes Stück der Zeitreihen-Prognose-Linie der "Konzepte"-Reihe (Kontrast zu Stück 2): dieselben Tagesaufträge eines Depots, aber das Modell wird aus Differenzieren, AR- und MA-Termen selbst zusammengesetzt und nach dem
Verfahren von Box und Jenkins gewählt.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import ari_constants as C
import ari_forecast as F
import ari_sarima as SAR
from ari_evaluation import Settings, analyse, auto_experiment, auto_search, diagnostics, equivalence_experiment, events_experiment, spec_experiment
from ari_presets import PRESET_HELP, PRESETS, SPEC_STATE_KEYS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from ari_visualization import SHORT, build_acf, build_auto, build_bars, build_equivalence, build_events, build_horizon, build_origin, build_series, build_specs, method_label

st.set_page_config(page_title="ARIMA – Sebastian Hanisch", layout="wide")


def de(x, digits=1):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


def current_settings():
    g = st.session_state
    return Settings(int(g["trend_slider"]), round(float(g["weekly_slider"]), 2), round(float(g["yearly_slider"]), 2), round(float(g["noise_slider"]), 2), int(g["shift_slider"]), round(float(g["events_slider"]), 2),
                    int(g["horizon_slider"]), int(g["step_slider"]), int(g["k_slider"]), int(g["seed_input"]), int(g["p_slider"]), int(g["d_slider"]), int(g["q_slider"]), int(g["sp_slider"]), int(g["sd_slider"]), int(g["sq_slider"]),
                    bool(g["log_check"]))


def apply_auto_order():
    """Knopf-Rückruf: Ordnung nach AICc aus dem Gitter wählen und die Regler setzen."""
    s = current_settings()
    rows = auto_search(s)
    spec = rows[0][0]
    for key, val in zip(SPEC_STATE_KEYS, (spec.p, spec.d, spec.q, spec.P, spec.D, spec.Q, spec.log)):
        st.session_state[key] = val
    st.session_state["auto_top"] = {"key": s.series_key, "rows": [(r[0].label, r[2]) for r in rows[:5]], "n": len(rows)}


@st.cache_data(show_spinner=False)
def _equivalence(seeds):
    return equivalence_experiment(seeds=seeds)


@st.cache_data(show_spinner=False)
def _spec(seeds):
    return spec_experiment(seeds=seeds)


@st.cache_data(show_spinner=False)
def _events(levels, seeds):
    return events_experiment(levels=levels, seeds=seeds)


@st.cache_data(show_spinner=False)
def _auto(seeds):
    return auto_experiment(seeds=seeds)


st.title("🧩 ARIMA – Modelle aus Bausteinen")
st.markdown(
    """
Die exponentielle Glättung aus dem Vorgänger-Stück hat feste Bauformen (Niveau, Trend, Saison). **ARIMA** geht den anderen Weg: ein Baukasten aus **Differenzieren** (I), **Autoregression** (AR: der Wert hängt von den letzten Werten ab) und **gleitenden Fehlern** (MA: der Wert hängt von den
letzten Prognosefehlern ab), dazu dieselben Bausteine noch einmal im Wochentakt (**SARIMA**). Die Ordnung $(p,d,q)(P,D,Q)$ wählt man selbst - nach Box und Jenkins aus Autokorrelationen, oder automatisch nach AICc. Die Demo läuft auf **denselben Tagesaufträgen eines Depots** wie die beiden
Vorgänger und zeigt, dass **naiv, saisonal naiv und einfache Glättung Sonderfälle sind**, was die zusätzliche Freiheit bringt - und wo sie nichts bringt.
"""
)
st.caption(
    "Drittes Stück der **Zeitreihen-Prognose-Linie** der \"Konzepte\"-Reihe, Kontrast zur Exponentiellen Glättung (gleicher Zweck, andere Bauweise); alle Daten sind erzeugt, die Rechnung ist in numpy geschrieben (die Kreuzprobe im Test läuft gegen statsmodels). "
    "**Bezug zu OR:** Nachfrageprognosen für Bestand, Personal und Touren; ARIMA ist der klassische Baukasten dafür und die Grundlage der Dynamischen Regression (nächstes Stück)."
)

with st.expander("So funktioniert ARIMA", expanded=True):
    st.markdown(
        """
1. **Differenzieren (I).** Statt der Reihe selbst betrachtet man Änderungen: $d = 1$ heißt $y_t - y_{t-1}$, $D = 1$ heißt $y_t - y_{t-7}$ (die Änderung gegenüber der Vorwoche). Das entfernt Niveau und Wochenmuster - übrig bleibt eine Reihe, die um Null schwankt.
2. **AR und MA.** Auf den Änderungen liegt ein ARMA-Modell: **AR($p$)** rechnet die letzten $p$ Änderungen ein, **MA($q$)** die letzten $q$ Prognosefehler. Die saisonalen Bausteine $P$ und $Q$ tun dasselbe im Abstand von 7, 14, ... Tagen.
3. **Log.** Das Wochenmuster des Vehikels ist multiplikativ (Freitag = +20 %). Auf $\\log y$ wird daraus ein additives Muster; die Prognose wird mit $e^{f + \\sigma^2/2}$ zurückgerechnet.
4. **Sonderfälle.** $(0,1,0)$ ist die naive Prognose, $(0,0,0)(0,1,0)$ die saisonal naive, $(0,1,1)$ die einfache Glättung mit $\\theta = \\alpha - 1$.
5. **Ordnung wählen.** Autokorrelation und partielle Autokorrelation der (differenzierten) Reihe verraten, welche Bausteine fehlen; die Fehler des Modells sollten keine Autokorrelation mehr zeigen. Oder: alle Ordnungen eines Gitters schätzen und nach AICc wählen.
6. **Bewertung.** Wie in den Vorgängern: viele Ursprünge (Rolling-Origin), MASE, Orakel-Untergrenze; die Parameter stammen aus den ersten zwei Jahren.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario mit Modell laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Das Modell**")
    p = st.slider("AR-Ordnung p", *bounds("p_slider"), key="p_slider", help="Wie viele letzte Änderungen das Modell einrechnet.")
    d = st.slider("Differenzieren d", *bounds("d_slider"), key="d_slider", help="Wie oft die Reihe differenziert wird (0 = gar nicht, 1 = Änderungen zum Vortag, 2 = Änderungen der Änderungen).")
    q = st.slider("MA-Ordnung q", *bounds("q_slider"), key="q_slider", help="Wie viele letzte Prognosefehler das Modell einrechnet.")
    sp = st.slider("Saisonale AR-Ordnung P", *bounds("sp_slider"), key="sp_slider", help="Wie die AR-Ordnung, im Abstand von 7 Tagen.")
    sd = st.slider("Saisonales Differenzieren D", *bounds("sd_slider"), key="sd_slider", help="1 = Änderung gegenüber demselben Wochentag der Vorwoche.")
    sq = st.slider("Saisonale MA-Ordnung Q", *bounds("sq_slider"), key="sq_slider", help="Wie die MA-Ordnung, im Abstand von 7 Tagen.")
    logt = st.checkbox("Log-Transformation", key="log_check", help="Das Modell rechnet mit log(Aufträge); die Prognose wird mit exp(f + s²/2) zurückgerechnet. Macht das multiplikative Wochenmuster additiv.")
    st.button("🔎 Ordnung automatisch wählen (AICc)", width="stretch", on_click=apply_auto_order,
              help=f"Schätzt {len(C.AUTO_D) * len(C.AUTO_P) * len(C.AUTO_Q) * len(C.AUTO_SP)} Ordnungen (im Log, mit saisonalem Differenzieren und saisonalem MA) auf den Trainingstagen und setzt die Regler auf die mit dem kleinsten AICc. Dauer wenige Sekunden.")
    st.markdown("**Die Reihe**")
    trend = st.slider("Trend (% je Jahr)", *bounds("trend_slider"), key="trend_slider", step=C.TREND_STEP, help="Lineares Wachstum (oder Schrumpfen) der Aufträge, in Prozent des Ausgangsniveaus je Jahr.")
    weekly = st.slider("Wochenmuster", *bounds("weekly_slider"), key="weekly_slider", step=C.WEEKLY_STEP, help="Stärke des Wochentagsmusters (1 = Standard: Freitag am stärksten, Sonntag am schwächsten; 0 = keines).")
    yearly = st.slider("Jahresmuster", *bounds("yearly_slider"), key="yearly_slider", step=C.YEARLY_STEP, help="Amplitude der jahreszeitlichen Schwankung (Anteil des Niveaus).")
    noise = st.slider("Rauschen", *bounds("noise_slider"), key="noise_slider", step=C.NOISE_STEP, help="Streuung des multiplikativen Rauschens (log-normal, im Mittel unverzerrt).")
    shift = st.slider("Niveausprung (%)", *bounds("shift_slider"), key="shift_slider", step=C.SHIFT_STEP, help="Sprung des Niveaus an einem zufälligen Tag im letzten Jahr (0 = keiner), z. B. ein neuer Großkunde.")
    events = st.slider("Feiertage und Aktionen", *bounds("events_slider"), key="events_slider", step=C.EVENTS_STEP, help="Stärke der Effekte: am Feiertag ruht das Depot (bis -50 %), am Folgetag Nachholeffekt, dazu drei Aktionswochen je Jahr (bis +50 %). ARIMA kennt sie nicht.")
    st.markdown("**Der Vergleich**")
    horizon = st.slider("Prognosehorizont (Tage)", *bounds("horizon_slider"), key="horizon_slider", help="Wie viele Tage im Voraus prognostiziert wird.")
    step = st.slider("Abstand der Ursprünge (Tage)", *bounds("step_slider"), key="step_slider", help="Alle wie viele Tage ein neuer Ursprung beginnt. 1 = jeder Tag des letzten Jahres.")
    k = st.slider("Wochen im Wochenmittel (k)", *bounds("k_slider"), key="k_slider", help="Über wie viele letzte Wochen das Vergleichsverfahren 'Wochenmittel' den Wochentag mittelt.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt Rauschen, Aktionstage, Phase des Jahresmusters und den Tag des Niveausprungs fest.")
    st.button("🎲 Neue Reihe generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"trend_slider": int(trend), "weekly_slider": round(float(weekly), 2), "yearly_slider": round(float(yearly), 2), "noise_slider": round(float(noise), 2), "shift_slider": int(shift), "events_slider": round(float(events), 2),
                   "horizon_slider": int(horizon), "step_slider": int(step), "k_slider": int(k), "seed_input": int(seed), "p_slider": int(p), "d_slider": int(d), "q_slider": int(q), "sp_slider": int(sp), "sd_slider": int(sd),
                   "sq_slider": int(sq), "log_check": bool(logt)})

settings = current_settings()
a = analyse(settings)
s = a.series
K = settings.k
model = a.user[0]
spec = model.spec
label = spec.label

auto = st.session_state.get("auto_top")
if auto and auto["key"] == settings.series_key:
    st.success(f"🔎 Automatische Ordnungswahl: von {auto['n']} Ordnungen hat {auto['rows'][0][0]} das kleinste AICc ({de(auto['rows'][0][1], 1)}); danach {', '.join(f'{lab} ({de(v, 1)})' for lab, v in auto['rows'][1:])}. Die Regler sind gesetzt.")

# --- Die Reihe und das Modell -----------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Reihe und das Modell an einem Ursprung")
lo_o, hi_o = int(a.origins[0]), int(a.origins[-1])
st.session_state["origin_slider"] = min(hi_o, max(lo_o, st.session_state.get("origin_slider", 900)))
origin = int(st.slider("Ursprung (Tag)", lo_o, hi_o, key="origin_slider", help="Ab diesem Tag wird prognostiziert; bekannt ist alles davor. Alle Ursprünge des Testjahres gehen in die Auswertung ein."))
st.plotly_chart(build_series(a, origin), width="stretch", key="series_chart")
st.plotly_chart(build_origin(a, origin), width="stretch", key="origin_chart")
weekday_name = C.WEEKDAYS[(origin - 1) % 7]
st.caption(
    f"Reihe mit {s.n} Tagen (Mittel {de(s.y.mean())} Aufträge je Tag); Ursprung an Tag {origin} (letzter bekannter Tag: ein {weekday_name}). Die dicke Linie ist die Prognose von ARIMA{label} für die nächsten {settings.horizon} Tage, gestrichelt Holt-Winters, "
    "Wochenmittel und einfache Glättung; die grüne gepunktete Linie ist der wahre Erwartungswert."
)

dg = diagnostics(a)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modell", f"ARIMA{label}", help="(p,d,q)(P,D,Q) mit Periode 7; 'log' heißt: das Modell rechnet mit log(Aufträge).")
c2.metric("AICc", de(model.aicc, 1), help="Bestraft Parameter; nur zwischen Modellen mit derselben Transformation vergleichbar (die Fehler werden auf denselben Tagen der Reihe summiert).")
c3.metric("Streuung der Fehler σ", (f"{de(np.sqrt(model.sigma2), 3)} (Log)" if spec.log else f"{de(np.sqrt(model.sigma2), 1)} Aufträge"), help="Wurzel der mittleren Fehlerquadrate auf den Trainingstagen (ohne die ersten 200 Tage).")
c4.metric("Ljung-Box p (14 Lags)", de(dg["lb_p"], 3), help="Nullhypothese: die Fehler sind unkorreliert. Ein kleiner p-Wert (unter 0,05) zeigt, dass im Fehler noch Struktur steckt.")
params = []
if spec.p:
    params.append("φ = " + ", ".join(de(v, 3) for v in model.phi))
if spec.q:
    params.append("θ = " + ", ".join(de(v, 3) for v in model.theta))
if spec.P:
    params.append("Φ = " + ", ".join(de(v, 3) for v in model.Phi))
if spec.Q:
    params.append("Θ = " + ", ".join(de(v, 3) for v in model.Theta))
st.caption("Geschätzte Parameter: " + ("; ".join(params) if params else "keine (reines Differenzieren)") + f". Schätzung: kleinste bedingte Fehlerquadrate auf den Tagen {SAR.BURN} bis {C.FIRST_TEST - 1} (die ersten {SAR.BURN} Tage zählen nicht).")

st.markdown("---")

# --- Werkstatt --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Werkstatt von Box und Jenkins: Autokorrelationen lesen")
n_z, n_w, n_r = len(dg["z"]), len(dg["w"]), len(dg["resid"])
diff_txt = "nicht differenziert" if spec.lost == 0 else f"differenziert mit d = {spec.d}, D = {spec.D}"
w1, w2 = st.columns(2)
with w1:
    st.plotly_chart(build_acf(dg["acf_z"], f"ACF der Reihe ({'log, ' if spec.log else ''}Trainingstage)", n_z, "#4c78a8"), width="stretch", key="acf_z_chart")
with w2:
    st.plotly_chart(build_acf(dg["acf_w"], f"ACF der Reihe, {diff_txt}", n_w, "#4c78a8"), width="stretch", key="acf_w_chart")
w3, w4 = st.columns(2)
with w3:
    st.plotly_chart(build_acf(dg["pacf_w"], f"PACF der Reihe, {diff_txt}", n_w, "#8e6bbf", partial=True), width="stretch", key="pacf_w_chart")
with w4:
    st.plotly_chart(build_acf(dg["acf_res"], f"ACF der Fehler von ARIMA{label}", n_r, "#c2185b"), width="stretch", key="acf_res_chart")
verdict_lb = "unterscheiden sich auf dem 5-%-Niveau **nicht** von unkorreliertem Rauschen" if dg["lb_p"] >= 0.05 else "zeigen auf dem 5-%-Niveau noch **Autokorrelation**"
st.caption(
    f"Die Balken sind Autokorrelationen der Trainingstage, das gestrichelte Band ist ±1,96/√n; die Lags 7, 14, 21 und 28 sind orange. **So liest man es (Standardreihe):** Die Reihe selbst hat bei den Wochenlags dauerhaft hohe Werte (Wochenmuster): sie ist nicht stationär, also wird saisonal differenziert. "
    f"Nach dem Differenzieren steht ein negativer Ausschlag bei Lag 7 (dort sitzt ein saisonaler MA-Term) und die PACF klingt im Wochentakt ab. Die Fehler von ARIMA{label} {verdict_lb} (Ljung-Box, 14 Lags: Q = {de(dg['lb_q'], 1)}, p = {de(dg['lb_p'], 3)}). "
    "Der Test prüft nur Autokorrelation: fehlende Einflussgrößen wie Feiertage findet er nicht (Experiment unten)."
)

st.markdown("---")

# --- Auswertung -----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Auswertung über alle Ursprünge")
n_org = len(a.origins)
sm = a.summary
u, hw, wm = sm["user"]["mase"], sm["hw_mult"]["mase"], sm["snaive_k"]["mase"]
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"ARIMA{label}", f"MASE {de(u, 2)}", delta=f"{u - hw:+.2f}".replace(".", ",") + " gegen Holt-Winters", delta_color="inverse", help="Der Pfeil vergleicht mit Holt-Winters multiplikativ aus Stück 2 (grün: besser).")
m2.metric("Holt-Winters multiplikativ", f"MASE {de(hw, 2)}", help="Das beste Glättungsmodell aus Stück 2.")
m3.metric(f"Wochenmittel (k = {K})", f"MASE {de(wm, 2)}", help="Derselbe Wochentag, gemittelt über die letzten k Wochen (Stück 1).")
m4.metric("Orakel-Untergrenze", f"MASE {de(a.oracle['mase'], 2)}", help="Fehler der Prognose 'wahrer Erwartungswert' gegen die beobachteten Werte: das Rauschen der Reihe. Kein Verfahren liegt im Mittel darunter.")
st.plotly_chart(build_bars(a), width="stretch", key="bars_chart")
rows = [{"Verfahren": method_label(m, K, label), "MASE": de(sm[m]["mase"], 2), "MAE (Aufträge)": de(sm[m]["mae"], 1), "RMSE": de(sm[m]["rmse"], 1), "Verzerrung (Prognose minus Ist)": de(sm[m]["me"], 1)} for m in sorted(sm, key=lambda m: sm[m]["mase"])]
st.dataframe(rows, hide_index=True)
gap_or = u / a.oracle["mase"] - 1
if u < hw - 0.005:
    st.success(f"✅ ARIMA{label} schlägt Holt-Winters: MASE {de(u, 3)} gegen {de(hw, 3)} (Wochenmittel {de(wm, 2)}, Orakel {de(a.oracle['mase'], 2)}); die Orakel-Untergrenze liegt noch {pct(gap_or)} darunter. Bei einer einzelnen Reihe ist der Abstand klein - die Experimente unten zeigen, wie er über zwölf Reihen aussieht.")
elif u <= hw + 0.005:
    st.info(f"ARIMA{label} und Holt-Winters liegen gleichauf: MASE {de(u, 3)} gegen {de(hw, 3)} (Wochenmittel {de(wm, 2)}, Orakel {de(a.oracle['mase'], 2)}). Das ist der Normalfall: die beiden Familien erreichen dieselbe Genauigkeit - die Experimente unten zeigen, wo sie sich unterscheiden.")
elif u < wm:
    st.info(f"ARIMA{label} liegt zwischen Holt-Winters ({de(hw, 2)}) und dem Wochenmittel ({de(wm, 2)}): MASE {de(u, 2)}; Orakel {de(a.oracle['mase'], 2)}.")
else:
    st.warning(f"⚠️ ARIMA{label} schlägt nicht einmal das Wochenmittel: MASE {de(u, 2)} gegen {de(wm, 2)} (Holt-Winters {de(hw, 2)}). Hier fehlt dem Modell ein Baustein - zum Beispiel das saisonale Differenzieren oder das Log; probieren Sie die automatische Ordnungswahl.")
st.caption(f"{n_org} Ursprünge im Testjahr (Abstand {settings.step} Tage), je {settings.horizon} Tage Horizont; MASE-Nenner: saisonal naiver Fehler in den ersten {C.FIRST_TEST} Tagen ({de(F.mase_scale(s.y, C.FIRST_TEST), 1)} Aufträge). "
           "Naiv, saisonal naiv und ARIMA(0,1,1) laufen hier als ARIMA-Modelle; ihre Zahlen sind dieselben wie in den Vorgängern (naiv und saisonal naiv in Stück 1, einfache Glättung in Stück 2).")

st.markdown("##### Wie der Fehler mit dem Horizont wächst")
st.plotly_chart(build_horizon(a), width="stretch", key="horizon_chart")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Sind naiv, saisonal naiv und Glättung wirklich Sonderfälle?")
st.caption(f"Einfache Glättung gegen ARIMA(0,1,1), Holt gegen ARIMA(0,2,2), Holt gedämpft gegen ARIMA(1,1,2); beide Seiten unabhängig geschätzt, dazu die Prognosen mit umgerechneten Parametern (θ = α − 1 für die einfache Glättung). Standardreihe, Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer wenige Sekunden.")
if st.button("Sonderfälle durchrechnen", key="equiv_start"):
    st.session_state["equiv_on"] = True
if st.session_state.get("equiv_on"):
    re_ = _equivalence(C.EXP_SEEDS)
    st.plotly_chart(build_equivalence(re_), width="stretch", key="equiv_chart")
    r0, r1, r2 = re_
    st.warning(
        f"**Befund:** Die einfache Glättung und ARIMA(0,1,1) sind **dasselbe Modell**: unabhängig geschätzt erreichen beide MASE {de(r0['ets'], 3)} und {de(r0['arima'], 3)}, und rechnet man die Parameter um, unterscheiden sich die Prognosen in keinem Ursprung um mehr als {r0['max_diff']:.0e} Aufträge. "
        f"Naiv und saisonal naiv fallen zusammen mit ARIMA(0,1,0) und (0,0,0)(0,1,0) (Tabelle oben: dieselben Zahlen wie in Stück 1). Bei Holt ({de(r1['ets'], 2)} gegen {de(r1['arima'], 2)}) und Holt gedämpft ({de(r2['ets'], 2)} gegen {de(r2['arima'], 2)}) sind die Modelle in der Theorie gleichwertig, "
        f"die Schätzung landet aber nicht auf demselben Punkt: Holt entspricht im Mittel θ₂ = 1 − α = {de(r1['theta_ets'], 2)}, also dem Rand der Umkehrbarkeit (θ₂ < 1), die ARIMA-Schätzung bleibt bei {de(r1['theta_arima'], 2)} stehen, und die MASE ist dabei um {de(r1['arima'] - r1['ets'], 2)} schlechter. Wer Trend über d = 2 modelliert, arbeitet am Rand des zulässigen Bereichs."
    )

st.markdown("---")

st.subheader("🔬 Log oder nicht - wie viel Differenzieren?")
st.caption(f"Standardreihe; sechs Ordnungen, mit und ohne Log, gegen Holt-Winters multiplikativ; rechts die Rücktransformation aus dem Log bei Rauschen {', '.join(de(x, 2) for x in C.BIAS_NOISE)}. Mittel über {len(C.EXP_SEEDS)} feste Seeds (Fehlerbalken: Standardfehler). Dauer etwa 10 bis 20 Sekunden.")
if st.button("Ordnungen durchrechnen", key="spec_start"):
    st.session_state["spec_on"] = True
if st.session_state.get("spec_on"):
    rs = _spec(C.EXP_SEEDS)
    st.plotly_chart(build_specs(rs), width="stretch", key="spec_chart")
    by = {r["label"]: r["mase"] for r in rs["rows"]}
    labs = [r["label"] for r in rs["rows"]]
    lo_b, hi_b = rs["bias"][0], rs["bias"][-1]
    st.warning(
        f"**Befund:** Das Log ist für dieses Vehikel nötig: {labs[0]} erreicht roh {de(by[labs[0]], 3)}, im Log {de(by[labs[1]], 3)}, denn das Wochenmuster ist multiplikativ. Holt-Winters multiplikativ liegt bei {de(rs['hw_mult'], 3)}; die beste der sechs Ordnungen ({min(by, key=by.get)}) bei {de(min(by.values()), 3)}: "
        f"ARIMA erreicht die Glättung, überholt sie aber nicht deutlich. Ohne Differenzieren im Trend ($d = 0$ mit AR-Term: {de(by[labs[3]], 3)}) statt $d = 1$ ({de(by[labs[2]], 3)}) ist es minimal besser. "
        f"Die Rücktransformation hängt vom Fehlermaß ab: bei Rauschen {de(lo_b['noise'], 2)} ist es fast gleich ({de(lo_b['mase_mean'], 3)} gegen {de(lo_b['mase_median'], 3)}), bei {de(hi_b['noise'], 2)} liefert der Median exp(f) die kleinere MASE ({de(hi_b['mase_median'], 3)} gegen {de(hi_b['mase_mean'], 3)}), "
        f"der Mittelwert exp(f + s²/2) den kleineren RMSE ({de(hi_b['rmse_mean'], 1)} gegen {de(hi_b['rmse_median'], 1)} Aufträge)."
    )

st.markdown("---")

st.subheader("🔬 Was passiert an Feiertagen und Aktionen?")
st.caption(f"Standardreihe mit Ereignisstärke {', '.join(de(x, 2) for x in C.EVENT_LEVELS)}; Fehler je Tagesart für Holt-Winters, SARIMA(1,0,1)(1,1,1) im Log, dasselbe mit d = 1 und das Wochenmittel; Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer wenige Sekunden.")
if st.button("Ereignisse durchrechnen", key="events_start"):
    st.session_state["events_on"] = True
if st.session_state.get("events_on"):
    rv = _events(C.EVENT_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_events(rv), width="stretch", key="events_chart")
    top = rv[-1]
    st.warning(
        f"**Befund:** Auch ARIMA kennt Feiertage und Aktionen nicht: bei Ereignisstärke {de(top['events'], 1)} liegt der Fehler an den Ereignistagen ({pct(top['share'][0])} der Prognosetage) bei {de(top['ar_log'][0], 1)} Aufträgen, {de(top['ar_log'][0] / top['oracle'][0], 1)}-mal so hoch wie beim Orakel ({de(top['oracle'][0], 1)}); "
        f"Holt-Winters {de(top['hw_mult'][0], 1)}, Wochenmittel {de(top['snaive_k'][0], 1)}. In den {C.EVENT_AFTER_DAYS} Tagen danach schneidet das stationäre Modell besser ab ({de(top['ar_log'][1], 1)}) als Holt-Winters ({de(top['hw_mult'][1], 1)}) und als dieselbe Ordnung mit d = 1 ({de(top['diff_log'][1], 1)}): "
        f"die Autoregression zieht das Niveau nach einem Ereignis zurück, das Glättungsmodell (und d = 1) trägt es als neues Niveau weiter. An den übrigen Tagen ist Holt-Winters ({de(top['hw_mult'][2], 1)}) dafür etwas besser als das stationäre Modell ({de(top['ar_log'][2], 1)})."
    )

st.markdown("---")

st.subheader("🔬 Hilft die automatische Ordnungswahl?")
st.caption(f"Standardreihe; je Reihe werden {len(C.AUTO_D) * len(C.AUTO_P) * len(C.AUTO_Q) * len(C.AUTO_SP)} Ordnungen (d ∈ {{{', '.join(str(x) for x in C.AUTO_D)}}}, p ∈ {{{', '.join(str(x) for x in C.AUTO_P)}}}, q ∈ {{{', '.join(str(x) for x in C.AUTO_Q)}}}, P ∈ {{{', '.join(str(x) for x in C.AUTO_SP)}}}, mit Log, D = 1, Q = 1) "
           f"auf den Trainingstagen geschätzt und nach AICc geordnet. Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa eine halbe Minute.")
if st.button("Ordnungswahl durchrechnen", key="auto_start"):
    st.session_state["auto_on"] = True
if st.session_state.get("auto_on"):
    ra = _auto(C.EXP_SEEDS)
    st.plotly_chart(build_auto(ra), width="stretch", key="auto_chart")
    n_d0 = sum(sp_.d == 0 for sp_ in ra["picks"])
    st.warning(
        f"**Befund:** Die nach AICc gewählte Ordnung erreicht MASE {de(ra['selected'][0], 3)} ± {de(ra['selected'][1], 3)}, die im Nachhinein beste Ordnung des Gitters {de(ra['best'], 3)} (Abstand {de(ra['gap'], 3)}), das Airline-Modell {de(ra['airline'][0], 3)} und Holt-Winters {de(ra['hw_mult'][0], 3)}. "
        f"Gewählt wurden {len(set(ra['picks']))} verschiedene Ordnungen in {ra['n_seeds']} Reihen ({n_d0}-mal ohne, {ra['n_seeds'] - n_d0}-mal mit d = 1): die Wahl ist bei dieser Reihenlänge nicht stabil - und weil die Ordnungen fast gleich gut sind, kostet das wenig."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nur die Reihe selbst zählt** | Feiertage und Aktionen sind ARIMA unbekannt: an den Ereignistagen so schlecht wie Holt-Winters und das Wochenmittel; der Ljung-Box-Test findet die fehlende Erklärung nicht. | Dynamische Regression (Stück 4) |
| **Die Ordnung ist bekannt** | Sie muss gewählt werden - von Hand oder nach AICc; bei dieser Reihenlänge wechselt die Wahl von Reihe zu Reihe. | Glättungsmodelle mit weniger Wahlfreiheit (Stück 2), Kombination (Stück 9) |
| **Linear und stationär nach dem Differenzieren** | Trend braucht d = 2 (Optimum am Rand der Umkehrbarkeit, empfindliche Schätzung); nichtlineare Muster oder Regimewechsel bleiben unerklärt. | Boosting (Stück 6) |
| **Der Bedarf ist nie null** | Das Log verlangt positive Werte; bei vielen Nullen brechen Log und Differenzieren. | Croston, SBA, TSB (Stück 5) |
| **Eine Reihe genügt** | Jedes Depot bekommt seine eigenen Parameter. | Globale Modelle: Boosting, Vortrainiertes Netz |
| **Es gibt eine Punktprognose** | Die Modellstreuung σ liefert Intervalle, aber nur unter der Annahme normalverteilter Fehler. | Prognoseintervalle (Stück 7) |
| **Erzeugte Reihe, zwölf Seeds** | Das Vehikel kennt genau die Muster, die es erzeugt; echte Reihen sind unordentlicher. Die Zahlen gelten für diese Reihen. | – |
"""
)
st.caption("Die Linie: Naive Prognose → Exponentielle Glättung → **ARIMA** → Dynamische Regression, dazu Croston, Boosting, Prognoseintervalle, Hierarchie, Kombination, Bestand und ein vortrainiertes Netz (die übrigen Stücke noch nicht gebaut).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Mit dem Rückwärtsschiebeoperator $B\,y_t = y_{t-1}$ gilt
$$\phi(B)\,\Phi(B^7)\,(1-B)^d\,(1-B^7)^D\,y_t = \theta(B)\,\Theta(B^7)\,\varepsilon_t,$$
mit $\phi(B) = 1 - \phi_1 B - \dots - \phi_p B^p$, $\theta(B) = 1 + \theta_1 B + \dots + \theta_q B^q$ und den saisonalen Gegenstücken $\Phi$, $\Theta$ in $B^7$; $\varepsilon_t$ sind unkorrelierte Fehler mit Varianz $\sigma^2$. Mit $w_t = (1-B)^d (1-B^7)^D y_t$ ist das ein ARMA-Modell für $w$.
Stationarität und Umkehrbarkeit werden erzwungen, indem die Koeffizienten aus Partial-Autokorrelationen in $(-1, 1)$ aufgebaut werden (Durbin-Levinson).

**Sonderfälle.** $(0,1,0)$: $\hat y_{t+h} = y_t$ (naiv). $(0,0,0)(0,1,0)$: $\hat y_{t+h} = y_{t+h-7}$ (saisonal naiv). $(0,1,1)$: $\hat y_{t+1} = y_t + \theta\,\varepsilon_t$, also einfache Glättung mit $\alpha = 1 + \theta$. Holt entspricht $(0,2,2)$ mit $\theta_1 = \alpha + \beta - 2$, $\theta_2 = 1 - \alpha$;
das gedämpfte Holt-Modell $(1,1,2)$ mit $\phi_1 = \phi$, $\theta_1 = \alpha + \phi\beta - 1 - \phi$, $\theta_2 = (1-\alpha)\phi$.

**Schätzen.** Bedingte Fehlerquadrate: $e_t = w_t - \sum_k A_k w_{t-k} - \sum_k B_k e_{t-k}$ mit $e_t = 0$ vor dem ersten Tag, an dem alle AR-Rückgriffe Daten haben; $A_k$, $B_k$ sind die Koeffizienten der ausmultiplizierten Polynome. Minimiert wird $\sum e_t^2$ über die Tage ab Tag 200
(die ersten Fehler tragen die Anfangsannahme $e = 0$ und verzerren sonst die Schätzung); Suche: 2 000 feste Zufallspunkte, dann sechs Runden lokaler Verfeinerung, deterministisch. $e_t$ ist zugleich der Ein-Schritt-Prognosefehler, deshalb sind die Fehlersummen und das
**AICc** $= n \ln(\mathrm{SSE}/n) + 2k + 2k(k+1)/(n-k-1)$ auch zwischen verschiedenen Differenzierungen vergleichbar, solange sie über dieselben Tage laufen.

**Prognose.** Die Fehler $\varepsilon$ der Zukunft sind 0; die Differenzen werden rekursiv fortgeschrieben und zu $y$ zurückintegriert. Im Log: $\hat y = \exp(f + \sigma^2/2)$ (Mittelwert) bzw. $\exp(f)$ (Median). **Ljung-Box:** $Q = n(n+2)\sum_{k=1}^{m} r_k^2/(n-k)$, Chi-Quadrat mit $m - (p+q+P+Q)$ Freiheitsgraden.

Implementiert in `ari_sarima.py` (Modell, Schätzung, Prognose), `ari_diagnostics.py` (ACF, PACF, Ljung-Box), `ari_ets.py` (Glättung zum Vergleich), `ari_forecast.py` (Rolling-Origin, Kennzahlen), `ari_evaluation.py` (Analyse, Ordnungswahl, vier Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
