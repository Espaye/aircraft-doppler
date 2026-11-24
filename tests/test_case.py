import json

import numpy as np
import pytest

from adoppler import case, cli, locate, pipeline

BASE = {'audio': 'a.wav', 'audio_start_utc': '2026-06-01T12:00:00Z', 'search_center': [52.0, 5.0]}
FLIGHTS = ('icao24,time,lat,lon,alt_m\n'                 # one aircraft at 12:00 UTC, another at 14:00 UTC
           'aaaaaa,2026-06-01T12:00:00Z,52.00,5.00,500\naaaaaa,2026-06-01T12:00:10Z,52.01,5.00,500\n'
           'bbbbbb,2026-06-01T14:00:00Z,52.00,5.00,500\nbbbbbb,2026-06-01T14:00:10Z,52.01,5.00,500\n')


def make(tmp_path, **over):
    (tmp_path / 'case.json').write_text(json.dumps({**BASE, **over}))
    return case.Case(tmp_path)


def wav(p, seconds=2.0, fs=8000):
    import soundfile as sf
    sf.write(str(p), np.zeros(int(seconds * fs), np.float32), fs)
    return p


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


@pytest.mark.parametrize('write', [False, True])
def test_new_case_with_bad_audio_writes_nothing(tmp_path, write):
    if write:
        (tmp_path / 'a.wav').write_bytes(b'not audio')
    with pytest.raises(case.CaseError, match='cannot read' if write else 'not found'):
        pipeline.new_case(tmp_path / 'c', tmp_path / 'a.wav', BASE['audio_start_utc'], BASE['search_center'])
    assert not (tmp_path / 'c').exists()


def test_cli_new_takes_paths_from_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wav(tmp_path / 'rec.wav'); (tmp_path / 'flights.csv').write_text(FLIGHTS)
    cli.main(['new', 'cases/c', '--audio', 'rec.wav', '--adsb-file', 'flights.csv',
              '--start', BASE['audio_start_utc'], '--center', '52,5'])
    c = case.Case(tmp_path / 'cases' / 'c')
    assert c.audio_path.exists() and (c.dir / c.s['adsb_file']).exists()


def test_flight_data_is_fetched_again_after_the_start_time_changes(tmp_path):
    wav(tmp_path / 'a.wav'); (tmp_path / 'flights.csv').write_text(FLIGHTS)
    c = make(tmp_path, adsb_source='file', adsb_file='flights.csv', geoid_undulation_m=0.0)
    quiet = lambda *a: None
    assert [t.icao24 for t in pipeline.fetch_adsb(c, log=quiet)] == ['aaaaaa']
    c = case.update(c, {'audio_start_utc': '2026-06-01T14:00:00Z'})
    assert [t.icao24 for t in pipeline.fetch_adsb(c, log=quiet)] == ['bbbbbb']
    assert [t.icao24 for t in pipeline.fetch_adsb(c, log=quiet)] == ['bbbbbb']        # now from the cache


def test_cli_refusals_are_messages(tmp_path):
    make(tmp_path)
    before = (tmp_path / 'case.json').read_text()
    with pytest.raises(SystemExit, match='already holds a case'):
        cli.main(['synth', str(tmp_path)])
    assert (tmp_path / 'case.json').read_text() == before
    with pytest.raises(SystemExit, match='not synthetic'):
        cli.main(['export', str(tmp_path / 'site'), str(tmp_path)])


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


def test_recording_gap_from_the_end_time(tmp_path):
    wav(tmp_path / 'a.wav', seconds=100)
    c = make(tmp_path)
    assert c.clock_gap_s is None and c.delay_range == c.delay_prior
    assert len(c.warnings()) == 1 and 'ground_temp_c is not set' in c.warnings()[0]
    c = make(tmp_path, audio_end_utc='2026-06-01T12:01:45Z', ground_temp_c=8)     # 5 s of file finalisation
    assert c.clock_gap_s == 0 and c.delay_range == c.delay_prior and c.warnings() == []
    c = make(tmp_path, audio_end_utc='2026-06-01T12:02:40Z', ground_temp_c=8)
    assert c.clock_gap_s == pytest.approx(60) and c.delay_range == (-62.0, 10.0)
    assert len(c.warnings()) == 1 and 'pause or dropout' in c.warnings()[0]
    c = make(tmp_path, audio_end_utc='2026-06-01T12:01:00Z', ground_temp_c=8)     # before start + duration
    with pytest.raises(case.CaseError, match='one of the two times is wrong'):
        c.clock_gap_s
    assert 'one of the two times is wrong' in c.warnings()[0]
    with pytest.raises(case.CaseError, match='later than audio_start_utc'):
        make(tmp_path, audio_end_utc='2026-06-01T11:00:00Z')


def test_cli_new_stores_the_end_time_and_warns(tmp_path, capsys):
    wav(tmp_path / 'rec.wav', seconds=100)
    cli.main(['new', str(tmp_path / 'c'), '--audio', str(tmp_path / 'rec.wav'), '--start', BASE['audio_start_utc'],
              '--end', '2026-06-01T12:02:40Z', '--center', '52,5', '--temp', '6'])
    out = capsys.readouterr().out
    assert 'warning: the recording is 60 s shorter' in out and 'ground_temp_c' not in out
    assert case.Case(tmp_path / 'c').s['audio_end_utc'] == '2026-06-01T12:02:40Z'


def test_one_aircraft_or_a_temperature_far_from_the_prior_is_weak():
    r = _result()
    r['events'] = [{'explained': True, 'icao24': 'aaaaaa'}] * 3 + [{'explained': False}]
    a = locate.assess(r)
    assert a['verdict'] == 'weak' and 'one aircraft' in a['flags'][0]['message']
    r['events'][3] = {'explained': True, 'icao24': 'bbbbbb'}
    assert locate.assess(r)['verdict'] == 'ok'
    r['best']['ground_temp_c'], r['settings']['ground_temp_prior_c'] = -15.0, 5.5       # at the lower bound
    assert locate.assess(r)['verdict'] == 'weak'
    r['best']['ground_temp_c'] = 16.0                                                   # 10.5 C from the prior
    assert locate.assess(r)['verdict'] == 'weak'
    r['best']['ground_temp_c'] = 9.0
    assert locate.assess(r)['verdict'] == 'ok'
