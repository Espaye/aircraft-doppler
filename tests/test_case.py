import json

import pytest

from adoppler import case, locate

BASE = {'audio': 'a.wav', 'audio_start_utc': '2026-06-01T12:00:00Z', 'search_center': [52.0, 5.0]}


def make(tmp_path, **over):
    (tmp_path / 'case.json').write_text(json.dumps({**BASE, **over}))
    return case.Case(tmp_path)


def test_defaults_and_time(tmp_path):
    c = make(tmp_path)
    assert c.t0 == 1780315200.0
    assert c.s['terrain'] == 'flat' and c.s['match']['min_score'] == 0.5


@pytest.mark.parametrize('over,msg', [
    ({'audio_start_utc': '2026-06-01 12:00'}, 'time zone'),
    ({'audio_start_utc': 'yesterday'}, 'ISO 8601'),
    ({'search_center': [95, 5]}, 'search_center'),
    ({'search_radius_km': 0}, 'search_radius_km'),
    ({'terrain': 'lidar'}, 'terrain'),
    ({'adsb_source': 'file'}, 'adsb_file'),
    ({'delay_prior_s': [3, 1]}, 'delay_prior_s'),
])
def test_invalid_settings_are_explained(tmp_path, over, msg):
    with pytest.raises(case.CaseError, match=msg):
        make(tmp_path, **over)


def test_missing_audio_is_a_case_error(tmp_path):
    with pytest.raises(case.CaseError):
        make(tmp_path).audio_info()


def _result(n_explained=5, shifted_total=60.0, stability=1.0, moved=False, alt_total=40.0):
    return {'best': {'n_explained': n_explained, 'n_events': 7, 'E_m': 100.0, 'N_m': 0.0, 'local_total': 20.0},
            'uncertainty': {'mode_stability': stability, 'bootstrap_units': 7, 'bootstrap_edge_fraction': 0.0,
                            'posterior_95_touches_edge': False, 'bootstrap_68_m': 20.0, 'bootstrap_90_m': 40.0,
                            'loo': [{'left_out': 'E01', 'coarse_mode_kept': not moved}]},
            'null_test': {'real_coarse_total': 30.0, 'real_coarse_n_explained': n_explained,
                          'shifted': [{'total': shifted_total, 'n_explained': 1}]},
            'alternatives': [{'E_m': 5000.0, 'N_m': 0.0, 'local_total': alt_total, 'n_explained': 2}],
            'settings': {'search_radius_m': 20000.0}}


@pytest.mark.parametrize('kw,verdict', [
    ({}, 'ok'),
    ({'n_explained': 1}, 'failed'),
    ({'n_explained': 2}, 'weak'),
    ({'moved': True}, 'weak'),
    ({'alt_total': 21.0}, 'ambiguous'),
    ({'stability': 0.5}, 'ambiguous'),
])
def test_assessment(kw, verdict):
    assert locate.assess(_result(**kw))['verdict'] == verdict


def test_null_test_close_to_real_is_ambiguous():
    r = _result(shifted_total=31.0)
    r['null_test']['shifted'][0]['n_explained'] = 5
    assert locate.assess(r)['verdict'] == 'ambiguous'
