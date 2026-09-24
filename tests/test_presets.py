"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen und Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import pytest

import ari_constants as C
import ari_evaluation as E
import ari_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi
        for key, state_key in (("trend", "trend_slider"), ("weekly", "weekly_slider"), ("yearly", "yearly_slider"), ("noise", "noise_slider"), ("shift", "shift_slider"), ("events", "events_slider")):
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-9


def test_standard_preset_equals_the_default_settings():
    p = P.PRESETS["Standardfall: (1,0,1)(1,1,1) im Log"]
    assert E.Settings(p["trend"], p["weekly"], p["yearly"], p["noise"], p["shift"], p["events"], p["horizon"], p["step"], p["k"], p["seed"], p["p"], p["d"], p["q"], p["P"], p["D"], p["Q"], p["log"]) == E.Settings()


def test_bounds_steps_and_unique_url_params():
    assert P.bounds("trend_slider") == (C.TREND_MIN, C.TREND_MAX) and P.bounds("p_slider") == (0, C.P_MAX) and P.bounds("sd_slider") == (0, C.SD_MAX)
    assert set(P.STEPS) == {"trend_slider", "weekly_slider", "yearly_slider", "noise_slider", "shift_slider", "events_slider"}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS) and set(P.SPEC_STATE_KEYS) <= set(P.SETTING_SPECS)


def test_log_permalink_value_is_validated():
    caster = P.SETTING_SPECS["log_check"].caster
    assert caster("1") is True and caster("0") is False and caster("True") is True and caster("nein") is False
    with pytest.raises(ValueError):
        caster("vielleicht")


def test_default_spec_is_inside_the_slider_bounds():
    for value, key in zip(C.DEFAULT_SPEC[:6], ("p_slider", "d_slider", "q_slider", "sp_slider", "sd_slider", "sq_slider")):
        lo, hi = P.bounds(key)
        assert lo <= value <= hi
