"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios, vgl. es_presets.py)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import ari_constants as C


def _bool(value):
    v = str(value).strip().lower()
    if v in ("1", "true", "ja", "yes"):
        return True
    if v in ("0", "false", "nein", "no"):
        return False
    raise ValueError(value)


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "trend_slider": SettingSpec("trend", int, C.DEFAULT_TREND, C.TREND_MIN, C.TREND_MAX),
    "weekly_slider": SettingSpec("weekly", float, C.DEFAULT_WEEKLY, C.WEEKLY_MIN, C.WEEKLY_MAX),
    "yearly_slider": SettingSpec("yearly", float, C.DEFAULT_YEARLY, C.YEARLY_MIN, C.YEARLY_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "shift_slider": SettingSpec("shift", int, C.DEFAULT_SHIFT, C.SHIFT_MIN, C.SHIFT_MAX),
    "events_slider": SettingSpec("events", float, C.DEFAULT_EVENTS, C.EVENTS_MIN, C.EVENTS_MAX),
    "horizon_slider": SettingSpec("horizon", int, C.DEFAULT_HORIZON, C.HORIZON_MIN, C.HORIZON_MAX),
    "step_slider": SettingSpec("step", int, C.DEFAULT_STEP, C.STEP_MIN, C.STEP_MAX),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K_WEEKS, C.K_WEEKS_MIN, C.K_WEEKS_MAX),
    "p_slider": SettingSpec("p", int, C.DEFAULT_SPEC[0], 0, C.P_MAX),
    "d_slider": SettingSpec("d", int, C.DEFAULT_SPEC[1], 0, C.D_MAX),
    "q_slider": SettingSpec("q", int, C.DEFAULT_SPEC[2], 0, C.Q_MAX),
    "sp_slider": SettingSpec("sp", int, C.DEFAULT_SPEC[3], 0, C.SP_MAX),
    "sd_slider": SettingSpec("sd", int, C.DEFAULT_SPEC[4], 0, C.SD_MAX),
    "sq_slider": SettingSpec("sq", int, C.DEFAULT_SPEC[5], 0, C.SQ_MAX),
    "log_check": SettingSpec("log", _bool, C.DEFAULT_SPEC[6]),
    "seed_input": SettingSpec("seed", int, 3, 0, C.SEED_MAX),
}
SPEC_STATE_KEYS = ("p_slider", "d_slider", "q_slider", "sp_slider", "sd_slider", "sq_slider", "log_check")
PRESET_KEYS = {"trend": "trend_slider", "weekly": "weekly_slider", "yearly": "yearly_slider", "noise": "noise_slider", "shift": "shift_slider", "events": "events_slider", "horizon": "horizon_slider",
               "step": "step_slider", "k": "k_slider", "seed": "seed_input", "p": "p_slider", "d": "d_slider", "q": "q_slider", "P": "sp_slider", "D": "sd_slider", "Q": "sq_slider", "log": "log_check"}
STEPS = {"trend_slider": C.TREND_STEP, "weekly_slider": C.WEEKLY_STEP, "yearly_slider": C.YEARLY_STEP, "noise_slider": C.NOISE_STEP, "shift_slider": C.SHIFT_STEP, "events_slider": C.EVENTS_STEP}


def _p(**kw):
    base = {"trend": C.DEFAULT_TREND, "weekly": C.DEFAULT_WEEKLY, "yearly": C.DEFAULT_YEARLY, "noise": C.DEFAULT_NOISE, "shift": C.DEFAULT_SHIFT, "events": C.DEFAULT_EVENTS, "horizon": C.DEFAULT_HORIZON,
            "step": C.DEFAULT_STEP, "k": C.DEFAULT_K_WEEKS, "seed": 3, "p": C.DEFAULT_SPEC[0], "d": C.DEFAULT_SPEC[1], "q": C.DEFAULT_SPEC[2], "P": C.DEFAULT_SPEC[3], "D": C.DEFAULT_SPEC[4], "Q": C.DEFAULT_SPEC[5],
            "log": C.DEFAULT_SPEC[6]}
    base.update(kw)
    return base


PRESETS = {
    "Standardfall: (1,0,1)(1,1,1) im Log": _p(),
    "Airline-Modell (0,1,1)(0,1,1) im Log": _p(p=0, d=1, q=1, P=0, D=1, Q=1),
    "Ohne Log (rohe Werte)": _p(p=0, d=1, q=1, P=0, D=1, Q=1, log=False),
    "Einfache Glättung als ARIMA(0,1,1)": _p(p=0, d=1, q=1, P=0, D=0, Q=0, log=False),
    "Saisonal naiv als SARIMA": _p(p=0, d=0, q=0, P=0, D=1, Q=0, log=False),
    "Feiertage und Aktionen stark": _p(events=1.0),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else round(float(snapped), 2)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(int(value)) if isinstance(value, bool) else str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


PRESET_HELP = {
    "Standardfall: (1,0,1)(1,1,1) im Log": "Seed 3: MASE ARIMA(1,0,1)(1,1,1) im Log 0,84, Holt-Winters multiplikativ 0,84, Wochenmittel (k = 4) 0,88, saisonal naiv 1,13, naiv 2,68; Orakel-Untergrenze 0,75. φ liegt mit 0,99 nahe an der Einheitswurzel, Θ bei −0,95. "
                                           "Im Mittel über 12 Reihen: 0,88 gegen 0,88 bei Holt-Winters - ein Gleichstand.",
    "Airline-Modell (0,1,1)(0,1,1) im Log": "Seed 3: MASE 0,85 (θ = −0,90, Θ = −0,93) gegen 0,84 bei Holt-Winters; im Mittel über 12 Reihen 0,89 gegen 0,88. Das klassische Modell für Reihen mit Trend und Saison.",
    "Ohne Log (rohe Werte)": "Seed 3: MASE 0,88 gegen 0,85 im Log; im Mittel über 12 Reihen 0,93 gegen 0,89. Das Wochenmuster des Vehikels ist multiplikativ (Freitag = +20 % vom Niveau); ohne Log muss ein fester Aufschlag reichen.",
    "Einfache Glättung als ARIMA(0,1,1)": "Seed 3: MASE 2,02 - dieselbe Zahl wie die einfache Glättung aus Stück 2 (θ = −0,96 entspricht α = 0,04); beide Seiten sind unabhängig geschätzt. Ein Modell ohne Wochenmuster; die Ljung-Box-Probe lehnt es ab.",
    "Saisonal naiv als SARIMA": "Seed 3: MASE 1,13 - dieselbe Zahl wie die saisonal naive Prognose aus Stück 1. Ein Modell ohne Parameter: nur das saisonale Differenzieren.",
    "Feiertage und Aktionen stark": "Seed 3, Ereignisstärke 1,0: ARIMA(1,0,1)(1,1,1) im Log 0,79, Holt-Winters 0,81, Wochenmittel 0,84, Orakel 0,62. Im Mittel über 12 Reihen 0,882 gegen 0,895 bei Holt-Winters (in 11 von 12 Reihen vorn): das stationäre Modell erholt sich schneller von einem Ereignis.",
}
