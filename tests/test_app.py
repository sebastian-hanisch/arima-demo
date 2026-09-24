"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Ordnungsregler, automatische Ordnungswahl, Würfel-Knopf, Permalink-Grenzen, Extremwerte, vier Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ari_constants as C
import ari_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value, el.value[:120]


def test_default_run_shows_metrics_charts_and_a_verdict():
    at = _run()
    _ok(at)
    assert len(at.metric) == 8 and len(at.get("plotly_chart")) == 8 and len(at.success) + len(at.info) + len(at.warning) >= 1
    assert any(m.value == "ARIMA(1,0,1)(1,1,1) log" for m in at.metric)


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == p[key]


def test_origin_slider_survives_a_shorter_test_range():
    at = _run(horizon_slider=1, origin_slider=1090)
    _ok(at)
    at.slider(key="horizon_slider").set_value(28).run()
    _ok(at)
    assert at.session_state["origin_slider"] <= C.N_DAYS - 28


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Reihe generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_automatic_order_selection_sets_the_sliders_and_reports():
    at = _run(p_slider=0, d_slider=0, q_slider=0, sp_slider=0, sd_slider=0, sq_slider=0, log_check=False)
    _ok(at)
    next(b for b in at.button if "automatisch" in b.label).click().run()
    _ok(at)
    assert at.session_state["log_check"] is True and at.session_state["sd_slider"] == 1 and at.session_state["sq_slider"] == 1 and at.session_state["d_slider"] in C.AUTO_D
    assert any("Automatische Ordnungswahl" in s.value for s in at.success)


def test_permalink_values_are_snapped_clamped_and_validated():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["trend"] = "999"
    at.query_params["noise"] = "0.31"
    at.query_params["weekly"] = "abc"
    at.query_params["p"] = "7"
    at.query_params["log"] = "0"
    at.query_params["horizon"] = "0"
    at.query_params["shift"] = "-35"
    at.run()
    _ok(at)
    assert at.session_state["trend_slider"] == C.TREND_MAX and at.session_state["noise_slider"] == 0.3 and at.session_state["weekly_slider"] == C.DEFAULT_WEEKLY
    assert at.session_state["p_slider"] == C.P_MAX and at.session_state["log_check"] is False and at.session_state["horizon_slider"] == C.HORIZON_MIN and at.session_state["shift_slider"] == -40


def test_invalid_log_permalink_falls_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["log"] = "vielleicht"
    at.run()
    _ok(at)
    assert at.session_state["log_check"] is C.DEFAULT_SPEC[6]


@pytest.mark.parametrize("kw", [dict(p_slider=2, d_slider=2, q_slider=2, sp_slider=1, sd_slider=1, sq_slider=1, log_check=True), dict(p_slider=0, d_slider=0, q_slider=0, sp_slider=0, sd_slider=0, sq_slider=0, log_check=False),
                                dict(p_slider=0, d_slider=0, q_slider=0, sp_slider=0, sd_slider=0, sq_slider=0, log_check=True), dict(p_slider=2, d_slider=0, q_slider=0, sp_slider=0, sd_slider=0, sq_slider=0, log_check=False),
                                dict(trend_slider=C.TREND_MIN, noise_slider=C.NOISE_MAX), dict(trend_slider=C.TREND_MAX, weekly_slider=0.0, yearly_slider=0.0), dict(horizon_slider=C.HORIZON_MAX, step_slider=C.STEP_MAX),
                                dict(shift_slider=-40, k_slider=12), dict(shift_slider=40, k_slider=2, events_slider=1.0), dict(noise_slider=C.NOISE_MIN, events_slider=0.0, horizon_slider=1)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def _small(monkeypatch):
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1))


def test_equivalence_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    next(b for b in at.button if b.key == "equiv_start").click().run()
    _ok(at)
    assert at.session_state["equiv_on"] and any("dasselbe Modell" in w.value for w in at.warning)


def test_spec_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    monkeypatch.setattr(C, "BIAS_NOISE", (0.14, 0.4))
    at = _run()
    next(b for b in at.button if b.key == "spec_start").click().run()
    _ok(at)
    assert at.session_state["spec_on"] and any("Das Log ist für dieses Vehikel nötig" in w.value for w in at.warning)


def test_events_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    next(b for b in at.button if b.key == "events_start").click().run()
    _ok(at)
    assert at.session_state["events_on"] and any("Auch ARIMA kennt Feiertage und Aktionen nicht" in w.value for w in at.warning)


def test_auto_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    monkeypatch.setattr(C, "AUTO_D", (0, 1))
    monkeypatch.setattr(C, "AUTO_P", (0, 1))
    monkeypatch.setattr(C, "AUTO_Q", (1,))
    monkeypatch.setattr(C, "AUTO_SP", (0,))
    at = _run()
    next(b for b in at.button if b.key == "auto_start").click().run()
    _ok(at)
    assert at.session_state["auto_on"] and any("Die nach AICc gewählte Ordnung" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present_and_no_unresolved_f_strings():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value and "{signed(" not in el.value
