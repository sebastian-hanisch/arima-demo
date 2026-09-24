# 🧩 ARIMA – Modelle aus Bausteinen

Drittes Stück der **Zeitreihen-Prognose-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning – und der **Kontrast** zur [Exponentiellen Glättung](https://github.com/sebastian-hanisch/exponential-smoothing-demo) (gleicher Zweck, andere Bauweise).
Geplant sind acht weitere Stücke (Dynamische Regression, Croston-Verfahren, Boosting, Prognoseintervalle, Hierarchische Abstimmung, Kombination, Prognose → Bestand, ein vortrainiertes Netz; noch nicht gebaut).

Die Glättung hat feste Bauformen (Niveau, Trend, Saison). **ARIMA** ist ein Baukasten: **Differenzieren** (I), **Autoregression** (AR: der Wert hängt von den letzten Werten ab) und **gleitende Fehler** (MA: er hängt von den letzten Prognosefehlern ab), dazu dieselben Bausteine im Wochentakt (**SARIMA**).
Die Ordnung $(p,d,q)(P,D,Q)$ wählt man selbst – nach Box und Jenkins aus Autokorrelationen – oder automatisch nach AICc. Die Demo läuft auf **denselben Tagesaufträgen eines Depots** wie die beiden Vorgänger (dieselbe Reihe, im Test auf denselben Fingerabdruck geprüft), im selben Rolling-Origin-Vergleich mit MASE und Orakel-Untergrenze.
Alle Daten sind erzeugt, die Rechnung ist in numpy geschrieben; die Kreuzprobe im Test läuft gegen statsmodels.

**Bezug zu OR:** Nachfrageprognosen für Bestand, Personal und Touren; ARIMA ist der klassische Baukasten dafür und die Grundlage der Dynamischen Regression (Regression mit ARIMA-Fehlern, nächstes Stück).

## Warum dieses Problem – und was sich gegenüber dem Plan geändert hat

Der Plan der Linie erwartete für diesen Kontrast: "ETS ≈ ARIMA". Das bestätigt sich – und die Messung zeigt, wo genau die beiden Familien zusammenfallen und wo nicht:

1. **Naiv, saisonal naiv und einfache Glättung sind Sonderfälle.** ARIMA(0,1,0) ist die naive Prognose, (0,0,0)(0,1,0) die saisonal naive (im Test identisch zu Stück 1), (0,1,1) die einfache Glättung mit $\theta = \alpha - 1$: mit umgerechneten Parametern unterscheiden sich die Prognosen um höchstens 6e-12 Aufträge; unabhängig geschätzt erreichen beide MASE 2,068.
2. **Bei Holt und gedämpftem Holt ist die Gleichwertigkeit nur theoretisch.** Holt (2,05) und (0,2,2) (2,10), gedämpft (2,04) und (1,1,2) (2,06): der geschätzte Holt-Trend entspricht $\theta_2 = 1 - \alpha \approx 0{,}99$, dem Rand der Umkehrbarkeit; die ARIMA-Schätzung bleibt bei 0,95 stehen, und die MASE ist dabei um 0,05 schlechter.
3. **Die Genauigkeit ist gleich.** SARIMA(1,0,1)(1,1,1) im Log erreicht im Mittel über zwölf Reihen 0,879, Holt-Winters multiplikativ 0,880 (Wochenmittel 0,948). Die zusätzliche Freiheit bringt hier keinen Vorsprung.
4. **Das Log ist der ARIMA-Weg zur multiplikativen Saison.** Ohne Log erreicht das Airline-Modell 0,933, mit Log 0,891; bei der Glättung war das ein eigener Modelltyp.
5. **Nach einem Ereignis erholt sich das stationäre Modell schneller.** Bei starken Feiertagen und Aktionen liegt SARIMA(1,0,1)(1,1,1) im Log vor Holt-Winters (0,882 gegen 0,895): die Autoregression zieht das Niveau zurück, die Glättung (und $d = 1$) trägt das Ereignis als neues Niveau weiter. Das war nicht geplant. An den Ereignistagen selbst ist keines der Modelle besser als das Wochenmittel – die Ereignisse selbst bleiben unerklärt (Stück 4).
6. **Der Ljung-Box-Test findet die Ereignisse nicht.** Er prüft Autokorrelation, nicht fehlende Einflussgrößen: er lehnt bei einer Minderheit der Reihen ab, mit und ohne Ereignisse etwa gleich oft.

## Modell

- **Die Reihe** (`ari_scenario.py`): wortgleich zu den Vorgängern: 1 095 Tage, Niveau, Trend, Wochenmuster, Jahresmuster, Niveausprung, Feiertage, Aktionen, multiplikatives Rauschen; Ursprünge im letzten Jahr (ab Tag 730).
- **Modell** (`ari_sarima.py`): $\phi(B)\,\Phi(B^7)\,(1-B)^d\,(1-B^7)^D\,y_t = \theta(B)\,\Theta(B^7)\,\varepsilon_t$. Ordnungen $p, q \le 2$, $d \le 2$, $P, D, Q \le 1$, dazu ein Schalter für $\log y$ (Rücktransformation $e^{f + \sigma^2/2}$, der Mittelwert; der Median $e^f$ ist getestet).
  Stationarität und Umkehrbarkeit werden erzwungen, indem die Koeffizienten aus Partial-Autokorrelationen in $(-1,1)$ aufgebaut werden (Durbin-Levinson).
- **Schätzen:** bedingte Fehlerquadrate der ersten 730 Tage; die ersten 200 Tage zählen nicht in die Summe (die Anfangsannahme $e = 0$ verzerrt sie – im Test: Airline-Modell 0,886 statt 0,908 auf sechs Reihen). Suche: 2 000 feste Zufallspunkte, dann sechs Runden lokaler Verfeinerung, alle Kandidaten gleichzeitig durch die Reihe gerechnet, deterministisch.
  Der Ein-Schritt-Fehler $e_t$ ist zugleich der Prognosefehler; die Fehlersummen laufen über dieselben Tage der Reihe, das **AICc** ist deshalb auch zwischen verschiedenen Differenzierungen vergleichbar.
- **Werkstatt:** ACF und PACF der Reihe, der differenzierten Reihe und der Fehler (95-%-Band $1{,}96/\sqrt n$), Ljung-Box-Test (Chi-Quadrat in numpy).
- **Vergleich:** naiv, saisonal naiv und ARIMA(0,1,1) laufen als ARIMA-Modelle; dazu einfache Glättung und Holt-Winters multiplikativ (`ari_ets.py`, der Kern aus Stück 2), das Wochenmittel (Stück 1), die Orakel-Untergrenze.

## Methodik

- **Handrechnungen:** Polynome $(1-0{,}5B)(1-0{,}4B^7)$, Partial-Autokorrelationen $\to$ Koeffizienten, Differenzen (auch $d = 2$ und $(1-B)(1-B^7)$), die Fehlerrekursion, einfache Glättung als ARIMA(0,1,1) ($10 \to 12 \to 11 \to 10{,}88$), naiv und saisonal naiv als Sonderfälle, $d = 2$ setzt eine Gerade fort, Rückkehr zum Mittel bei AR(1), Log mit und ohne Bias-Korrektur, Abschneiden bei 0.
- **Kreuzprobe gegen statsmodels:** sechs Ordnungen (mit $d = 0/1/2$, $D = 0/1$, AR- und MA-Termen, saisonalen Termen und dem Mittel-Modell) stimmen bei festen Parametern in der Prognose auf 1e-6 mit `statsmodels` überein (gemessen: 1e-8). Die geschätzten Fehlerquadrate sind höchstens 0,1 % größer als bei den statsmodels-Parametern (meist kleiner).
  ACF, PACF und Ljung-Box stimmen mit statsmodels überein; die Chi-Quadrat-Verteilung mit scipy.
- **Umrechnung Glättung → ARIMA** (Hyndman et al. 2008): für einfache Glättung, Holt und gedämpftes Holt reproduzieren die umgerechneten Parameter die Glättungsprognose auf 1e-6 (mit kräftigen Parametern, weil sich die Anfangszustände sonst nur langsam angleichen).
- **Kein Blick in die Zukunft:** Ändert man die Tage ab einem Ursprung, bleibt die Prognose früherer Ursprünge unverändert. **Wiedererkennung:** ein simulierter ARMA(1,1)-Prozess wird auf die Parameter und die Fehlervarianz zurückgerechnet.
- **Statistik:** zwölf feste Seeds, Fehlerbalken = Standardfehler.
- **Literatur** (nicht nachgebaut): Box/Jenkins, *Time Series Analysis* (Identifikation, Schätzung, Diagnose); Hyndman/Athanasopoulos, *Forecasting: Principles and Practice* (3. Aufl., Kap. 9); Hyndman et al., *Forecasting with Exponential Smoothing* (2008; die Äquivalenzen).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Standardreihen** (12 Seeds, Horizont 14, 352 Ursprünge) | MASE: SARIMA(1,0,1)(1,1,1) im Log **0,879**, Holt-Winters multiplikativ **0,880**, Wochenmittel (k = 4) 0,948, saisonal naiv 1,165, einfache Glättung 2,068, naiv 2,73; Orakel 0,735. Geschätzt im Mittel: φ = 0,99, θ = −0,85, Θ = −0,94, Φ um 0. | `test_default_model_over_twelve_series`, `test_fitted_parameters_over_twelve_series` |
| Standardfall (Preset, Seed 3) | ARIMA(1,0,1)(1,1,1) im Log 0,84, Holt-Winters 0,84, Wochenmittel 0,88, saisonal naiv 1,13, naiv 2,68, Orakel 0,75; φ = 0,99, Θ = −0,95. | `test_standard_preset_and_agreement_with_the_predecessors` |
| **Sonderfälle** | Einfache Glättung und ARIMA(0,1,1): MASE 2,068 und 2,068, Prognosen mit umgerechneten Parametern höchstens 6e-12 auseinander. Holt 2,05 / ARIMA(0,2,2) 2,10 (Holt entspricht $\theta_2 = 0{,}99$, die Schätzung bleibt bei 0,95); gedämpft 2,04 / (1,1,2) 2,06. | `test_equivalence_experiment` |
| Presets: Sonderfälle (Seed 3) | ARIMA(0,1,1): 2,02 (θ = −0,96 ↔ α = 0,04), dieselbe Zahl wie die einfache Glättung; SARIMA(0,0,0)(0,1,0): 1,13, identisch zu saisonal naiv, kein Parameter. | `test_special_case_presets_reproduce_the_predecessors` |
| **Log und Differenzieren** (12 Seeds) | (0,1,1)(0,1,1) roh **0,93**, im Log **0,89**; (1,1,1)(0,1,1) log 0,89; (1,0,1)(0,1,1) log 0,88; (1,0,1)(1,1,1) roh 0,91, im Log **0,88**; Holt-Winters 0,88. Ohne Differenzieren im Trend (mit AR-Term) minimal besser. | `test_spec_experiment` |
| **Rücktransformation aus dem Log** (Rauschen 0,14 / 0,40) | Bei 0,14 fast gleich (MASE 0,878 mit Mittelwert und Median); bei 0,40 liefert der **Median** exp(f) die kleinere MASE (0,846 gegen 0,856), der **Mittelwert** exp(f + s²/2) den kleineren RMSE (59,0 gegen 60,3 Aufträge). | `test_spec_experiment` |
| Presets: Airline und ohne Log (Seed 3) | Airline im Log 0,85 (θ = −0,90, Θ = −0,93); roh 0,88; Holt-Winters 0,84. | `test_airline_and_raw_presets` |
| **Ereignisse** (Stärke 0 / 0,5 / 1,0; 9 % Ereignistage, 27 % bis 14 Tage danach, 64 % sonst) | Stärke 1,0, Fehler in Aufträgen (Ereignistag / danach / sonst): Holt-Winters 54,2 / 20,3 / 15,5; SARIMA(1,0,1)(1,1,1) log **54,0 / 18,2 / 15,9**; dasselbe mit d = 1: 54,2 / 19,3 / 15,8; Wochenmittel 55,0 / 18,8 / 17,8; Orakel 17,3 / 13,7 / 13,9. An Ereignistagen das 3,1-Fache des Orakels; nach dem Ereignis ist das stationäre Modell am besten, an normalen Tagen ist Holt-Winters etwas besser. Ohne Ereignisse: 14,9 / 14,5 / 14,8 (Holt-Winters) gegen 15,0 / 14,7 / 15,1. | `test_events_experiment` |
| Ereignisse: Mittel über 12 Reihen | SARIMA 0,882 gegen Holt-Winters 0,895, in mindestens 9 von 12 Reihen vorn; φ im Mittel 0,956 (Standard: 0,986). Preset (Seed 3): 0,79 gegen 0,81, Wochenmittel 0,84, Orakel 0,62. | `test_events_mean_advantage_over_holt_winters`, `test_strong_events_preset` |
| **Werkstatt: Autokorrelationen** | ACF der rohen Reihe bei Lag 1 / 7 / 14 / 21 / 28: 0,23 / 0,81 / 0,80 / 0,80 / 0,77 (das Wochenmuster hält an); nach saisonalem Differenzieren im Log: Lag 1 0,12, **Lag 7 −0,48**, PACF bei Lag 7 und 14: −0,48 und −0,34 (Signatur eines saisonalen MA-Terms). | `test_acf_signatures` |
| **Ljung-Box-Test** | Lehnt bei höchstens einem Drittel der zwölf Reihen ab (Niveau 5 %), mit Ereignisstärke 0 und mit 1,0 etwa gleich oft; der Median der p-Werte liegt in beiden Fällen über 0,05: der Test sieht die Ereignisse nicht. | `test_ljung_box_does_not_see_the_events` |
| **Automatische Ordnungswahl** (24 Ordnungen nach AICc, 12 Seeds) | Gewählte Ordnung MASE **0,880 ± 0,024**, im Nachhinein beste Ordnung des Gitters 0,874, Airline-Modell 0,891, Holt-Winters 0,880. Die Wahl wechselt von Reihe zu Reihe (mindestens fünf verschiedene Ordnungen, etwa je zur Hälfte mit und ohne d = 1). | `test_auto_experiment` |
| Burn-in der Schätzung | Airline-Modell roh, sechs Reihen: 0,886 mit 200 Tagen Burn-in, 0,908 ohne. | `test_burn_in_improves_the_conditional_estimate` |

Die Preset-Zeilen sind **Einzelreihen** (Seed 3); belastbar sind die Zeilen über zwölf Seeds.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nur die Reihe selbst zählt** | Feiertage und Aktionen sind ARIMA unbekannt; an den Ereignistagen so schlecht wie Holt-Winters und das Wochenmittel, und der Ljung-Box-Test findet die fehlende Erklärung nicht. | Dynamische Regression (geplant) |
| **Die Ordnung ist bekannt** | Sie muss gewählt werden – von Hand oder nach AICc; bei dieser Reihenlänge wechselt die Wahl von Reihe zu Reihe, kostet aber wenig, weil die Ordnungen fast gleich gut sind. | Glättungsmodelle mit weniger Wahlfreiheit, Kombination (geplant) |
| **Linear und stationär nach dem Differenzieren** | Trend über $d = 2$ arbeitet am Rand der Umkehrbarkeit (empfindliche Schätzung); nichtlineare Muster oder Regimewechsel bleiben unerklärt. | Boosting (geplant) |
| **Der Bedarf ist nie null** | Das Log verlangt positive Werte; bei vielen Nullen brechen Log und Differenzieren. | Croston, SBA, TSB (geplant) |
| **Eine Reihe genügt** | Jedes Depot bekommt seine eigenen Parameter. | Globale Modelle: Boosting, Vortrainiertes Netz (geplant) |
| **Es gibt eine Punktprognose** | Die Fehlerstreuung σ liefert Intervalle, aber nur unter der Annahme normalverteilter Fehler. | Prognoseintervalle (geplant) |
| **Erzeugte Reihe, zwölf Seeds** | Das Vehikel kennt genau die Muster, die es erzeugt; echte Reihen sind unordentlicher. Die Zahlen gelten für diese Reihen. | – |

## Tests

Pytest-Suite (`pytest tests/ -v`, rund anderthalb Minuten, 96 Tests): der Kern von Hand (Polynome, Differenzen, Fehlerrekursion, Sonderfälle), Kreuzprobe gegen statsmodels (sechs Ordnungen, Schätzung, ACF/PACF/Ljung-Box) und scipy (Chi-Quadrat), Schätzung (deterministisch, zulässig, kein Blick in die Zukunft, Wiedererkennung eines simulierten ARMA-Prozesses),
Umrechnung Glättung → ARIMA, die Reihe (Fingerabdruck wie in den Vorgängern), Auswertung, Ordnungswahl und Experimentzeilen, Preset- und Permalink-Klemmen, AppTest-Rauchtests (Standard, jedes Preset, automatische Ordnungswahl, Extremwerte, vier Experimente auf Abruf, keine unaufgelösten Platzhalter) und `test_claims.py` (jede Zahl aus diesem README und aus den Preset-Hinweisen; Reihen und Schätzung sind deterministisch, die Bänder großzügiger als die Rundung).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `ari_constants.py` | Regler-Grenzen, Verfahren, Ordnungsgitter, Experiment-Seeds |
| `ari_presets.py` | Permalink/Presets-Mechanik, `PRESET_HELP` |
| `ari_scenario.py` | Die Reihe (wortgleich zu den Vorgängern) |
| `ari_sarima.py` | SARIMA: Polynome, Differenzieren, Schätzung, Prognose |
| `ari_ets.py` | Glättungsmodelle aus Stück 2 (zum Vergleich) |
| `ari_diagnostics.py` | ACF, PACF, Ljung-Box, Chi-Quadrat |
| `ari_forecast.py` | Rolling-Origin, Wochenmittel, MAE/RMSE/ME/MASE |
| `ari_evaluation.py` | Analyse, Ordnungswahl, Umrechnung, vier Experimente, Diagnose |
| `ari_visualization.py` | Plotly-Abbildungen |

## Bewusst nicht umgesetzt

- Exakte Likelihood (Kalman-Filter) und Konstanten im differenzierten Modell (Drift); die bedingten Fehlerquadrate mit Burn-in genügen für die Aussagen hier.
- Höhere Ordnungen und andere Saisonlängen; Modellwahl über ein größeres Gitter oder mit Einheitswurzeltests.
- Regression auf Kalendermerkmale (ARIMAX): das ist Stück 4. Ein PDF-Export gehört nicht zur Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy (Kreuzproben im Test: statsmodels, scipy).
