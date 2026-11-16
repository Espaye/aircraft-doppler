"""Broadband screening and geometric identifiability; synthetic consistency only."""
import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from adoppler import adsb, case, cli, geo, jets, physics


ALPHA = float(physics.absorption_db_per_m(jets.ABSORPTION_HZ, 15))   # case default ground temperature


def level(R, peak_db=30.0):
    """Synthetic activity in dB above the noise floor: spreading and air absorption, power-added to the floor."""
    v = -20 * np.log10(R) - ALPHA * R
    v = v - v.max() + 10 * np.log10(10 ** (peak_db / 10) - 1)
    return 10 * np.log10(1 + 10 ** (v / 10))


def make_case(tmp_path, seconds=600, radius=15):
    fs = 4000
    sf.write(str(tmp_path / 'a.wav'), np.zeros(seconds * fs, np.float32), fs)
    return case.create(tmp_path, {'audio': 'a.wav', 'audio_start_utc': '2026-06-01T12:00:00Z',
                                  'search_center': [52, 5], 'search_radius_km': radius,
                                  'geoid_undulation_m': 0})


def independent_range(t, cpa, origin, velocity, observer, c):
    # Solve the light/sound-cone quadratic directly, independently of the Newton solver.
    D = np.asarray(origin) + (np.asarray(t) - cpa)[..., None] * velocity - observer
    vv = np.dot(velocity, velocity)
    dv = D @ velocity
    delay = (-dv + np.sqrt(dv ** 2 + (c * c - vv) * np.sum(D * D, axis=-1))) / (c * c - vv)
    return c * delay


def flight(enu, cpa, origin, velocity, ident='abc123'):
    t = np.arange(cpa - 500, cpa + 501, 2.)
    P = np.asarray(origin) + (t - cpa)[:, None] * velocity
    lat, lon, alt = enu.to_geodetic(P)
    return adsb.Track(ident, ident, 'A320', None, 'synthetic', t, lat, lon, alt,
                      np.full(len(t), 'geom'))


def test_broad_noise_survives_while_modulated_single_tone_does_not(tmp_path):
    fs = 4000
    t = np.arange(300 * fs) / fs
    rng = np.random.default_rng(9)
    # A loud changing narrow tone at 50 s, broadband roar at 190 s, flat background.
    x = .2 * np.exp(-.5 * ((t - 50) / 15) ** 2) * np.sin(2 * np.pi * 440 * t)
    x += (1e-5 + .02 * np.exp(-.5 * ((t - 190) / 25) ** 2)) * rng.normal(size=len(t))
    sf.write(str(tmp_path / 'noise.flac'), x, fs, subtype='PCM_24')
    env = jets.envelope(tmp_path / 'noise.flac')
    ev = jets.find_events(env)
    assert any(abs(e['peak_s'] - 190) < 10 for e in ev)
    assert not any(abs(e['peak_s'] - 50) < 20 for e in ev)
    assert all(e['label'] == 'unclassified broadband event' for e in ev)


def test_flat_audio_and_window_refusals():
    env = {'t_s': np.arange(.5, 120).tolist(), 'activity_db': np.zeros(120).tolist(),
           'smoothed_band_db': np.zeros((120, 4)).tolist(), 'floor_db': [0] * 4}
    assert jets.find_events(env) == []
    for windows in ([(-1, 30)], [(0, 10)], [(100, 140)], [(10, 50), (40, 80)], [(0, float('nan'))]):
        with pytest.raises(jets.JetError):
            jets.find_events(env, windows)
    assert jets.find_events(env, [(100, 120)])[0]['flags'] == ['clipped_by_recording', 'limited_frequency_support', 'abrupt_onset']


def test_range_envelope_uses_sound_travel_time():
    enu = geo.ENU(52, 5)
    c = 325.
    tr = flight(enu, 1000, [0, 0, 10000], np.array([230, 0, 0]))
    traj = physics.Trajectory(tr, enu, 500, 1500, dt=.5, smooth_s=.01)
    rx = np.arange(950., 1100., .25)
    predicted = jets.range_level(traj, [[0, 0, 0]], rx, c)[0]
    exact = -20 * np.log10(independent_range(rx, 1000, [0, 0, 10000], np.array([230, 0, 0]), [0, 0, 0], c))
    assert np.allclose(predicted, exact, atol=.002)
    assert rx[np.argmax(predicted)] == pytest.approx(1000 + 10000 / c, abs=.3)


def test_gaps_anticorrelation_and_flat_envelopes_cannot_win():
    y = np.array([0, 3, 6, 3, 0.])
    assert jets.shape_cost(y[None] + 70, y)[0] < 1e-10
    assert jets.shape_cost((-y)[None], y)[0] == pytest.approx(1)
    assert jets.shape_cost(np.array([[np.nan, 3, 6, 3, np.nan]]), y)[0] == 1
    assert jets.shape_cost(y[None], np.ones(5))[0] == 1
    # An almost flat far-away model cannot be amplified without limit.
    assert jets.shape_cost((y / 1000)[None], y)[0] > .99


def test_air_absorption_matches_iso_9613_table():
    # ISO 9613-2 table 2, dB/km at 70 % relative humidity
    for t_c, table in ((20, {500: 2.8, 1000: 5.0, 2000: 9.0}), (10, {500: 1.9, 1000: 3.7, 4000: 32.8})):
        for f, db_km in table.items():
            assert physics.absorption_db_per_m(f, t_c) * 1000 == pytest.approx(db_km, rel=.03)


def test_level_fit_uses_bounded_gain_and_directivity():
    k = np.arange(60.)
    x = -20 * np.log10(np.hypot(9000, 230 * (k - 30) * 4))
    u = (k - 30) / np.hypot(9000 / 230 / 4, k - 30)          # cosine: towards, then away
    cost, gain, b = jets.fit_level(x, 1.2 * x + 3 * u + 50, u)
    assert cost < 1e-9 and gain == pytest.approx(1.2) and b == pytest.approx(3)
    assert jets.shape_cost(x, 1.2 * x + 3 * u) > 1e-3          # without the term the asymmetry stays
    cost, _, b = jets.fit_level(x, x + 12 * u, u)
    assert b == pytest.approx(jets.DIRECTIVITY_DB[1]) and cost > 1e-3   # b is bounded
    cost, gain, _ = jets.fit_level(x / 20, x, u)
    assert gain == pytest.approx(jets.GAIN_LIMITS[1]) and cost > .5    # an almost flat model is not stretched
    cost, gain, _ = jets.fit_level(x, -x, u)
    assert gain == pytest.approx(jets.GAIN_LIMITS[0]) and cost == 1    # nor inverted


def test_sounds_share_one_clock_offset():
    # Each sound fits at its own offset, but the two offsets differ: the shared offset cannot serve both.
    costs = np.ones((2, 1, 2, 2), np.float32)       # sound x cell x offset x flight
    costs[0, 0, :, 0] = .1, .9
    costs[1, 0, :, 1] = .9, .1
    per, assoc = jets._assign_lags(costs)
    assert (assoc[:, 0] == [[0, 0], [1, 1]]).all()
    score, _ = jets._best_offsets(per)
    assert score[0] == pytest.approx(.5)              # not .9, as per-sound timing shifts would give


def test_one_aircraft_cannot_be_counted_as_two_independent_passes():
    costs = np.array([[[.01]], [[.02]]])
    per, assoc = jets._assign(costs)
    assert (assoc[:, 0] >= 0).sum() == 1
    assert per[:, 0].sum() == pytest.approx(1.01)


def test_single_pass_retains_mirror_and_corridor_ambiguity(tmp_path):
    c = make_case(tmp_path)
    enu = geo.ENU(*c.center)
    origin, velocity, obs = np.array([0, 0, 9000]), np.array([230, 0, 0]), np.array([2500, 5000, 2])
    tr = flight(enu, c.t0 + 280, origin, velocity)
    adsb.save_tracks([tr, tr], c.tracks_path)  # Duplicate archive records still mean one aircraft.
    t = np.arange(.5, 600)
    cs = physics.path_sound_speed(15, 2, 9000)
    y = level(independent_range(c.t0 + t, c.t0 + 280, origin, velocity, obs, cs))
    env = {'t_s': t.tolist(), 'activity_db': y.tolist()}
    events = [{'id': 'J01', 't_start': 180, 't_end': 420, 'flags': []}]
    r, g = jets.match(c, env, events, step_m=2500, log=lambda *a: None)
    assert r['status'] == 'ambiguous' and r['n_supported'] == 1
    assert r['location_estimate'] is None
    for north in (5000, -5000):
        i = np.flatnonzero((g['E_m'] == 2500) & (g['N_m'] == north))[0]
        assert g['near_best'][i]
    assert np.ptp(g['E_m'][g['near_best']]) >= 5000
    assert r['stability']['bootstrap_grid_90_m'] is None


def test_three_crossing_passes_and_saved_data_are_not_claimed_as_real_accuracy(tmp_path):
    c = make_case(tmp_path, seconds=1300)
    enu = geo.ENU(*c.center)
    observer = np.array([2500, -2500, 2])
    delta = 4.0     # recording clock ahead of acoustic time, shared by every sound
    t = np.arange(.5, 1300)
    y = np.zeros(len(t))
    tracks, events = [], []
    cs = physics.path_sound_speed(15, 2, 9000)
    for k, (cpa, vel) in enumerate([(170, [230, 0, 0]), (590, [0, 230, 0]), (1050, [180, 150, 0])]):
        origin, velocity = np.array([0, 0, 9000]), np.array(vel)
        tracks.append(flight(enu, c.t0 + cpa, origin, velocity, f'abc12{k}'))
        sel = abs(t - cpa) < 190
        y[sel] = level(independent_range(c.t0 + t[sel] - delta, c.t0 + cpa, origin, velocity, observer, cs))
        events.append({'id': f'J{k + 1:02d}', 't_start': cpa - 90, 't_end': cpa + 100, 'flags': []})
    adsb.save_tracks(tracks, c.tracks_path)
    r, g = jets.match(c, {'t_s': t.tolist(), 'activity_db': y.tolist()}, events, step_m=2500,
                      timing_slack_s=0, log=lambda *a: None)
    assert r['n_supported'] == 3
    assert len({a['icao24'] for a in r['assignments']}) == 3
    assert abs(r['clock_offset_s'] - delta) <= 1.5
    assert {a['timing_shift_s'] for a in r['assignments']} == {r['clock_offset_s']}
    assert r['settings']['absorption_db_per_km'] == pytest.approx(ALPHA * 1000, abs=1e-3)
    true_cell = np.flatnonzero((g['E_m'] == observer[0]) & (g['N_m'] == observer[1]))[0]
    assert g['near_best'][true_cell]
    assert r['stability']['bootstrap_grid_90_m'] is not None
    assert len(r['stability']['leave_one_out']) == 3
    assert len(r['null_tests']) == 3
    assert r['location_estimate'] is None   # internal consistency cannot calibrate real accuracy


def test_cli_pending_report_and_reload_do_not_touch_tonal_prediction(tmp_path):
    c = make_case(tmp_path, seconds=120)
    fs = 4000
    rng = np.random.default_rng(3)
    t = np.arange(120 * fs) / fs
    sf.write(str(c.audio_path), (.005 + .1 * np.exp(-.5 * ((t - 60) / 15) ** 2)) * rng.normal(size=len(t)), fs)
    c.match_dir.mkdir()
    c.result_path.write_text('frozen tonal result')
    r = jets.run(c, log=lambda *a: None)
    assert r['status'] == 'awaiting_adsb' and r['location_estimate'] is None
    out = c.dir / 'jets'
    assert (out / 'result.sha256').read_text().strip() == hashlib.sha256((out / 'result.json').read_bytes()).hexdigest()
    assert json.loads((out / 'events.json').read_text())[0]['label'] == 'unclassified broadband event'
    assert 'plotly' in (out / 'report.html').read_text(encoding='utf-8').lower()
    # Rerun using persisted files/settings and exercise the actual command parser.
    cli.main(['jets', str(c.dir), '--analyse-only', '--window', '30', '90'])
    assert json.loads((out / 'result.json').read_text())['status'] == 'analysis_only'
    assert c.result_path.read_text() == 'frozen tonal result'

def test_matched_report_grid_stamp_and_cli_rerun(tmp_path):
    c = make_case(tmp_path)
    fs = 4000
    enu = geo.ENU(*c.center)
    origin, velocity, observer = np.array([0, 0, 9000]), np.array([230, 0, 0]), np.array([2500, 5000, 2])
    cpa = c.t0 + 280
    tr = flight(enu, cpa, origin, velocity)
    adsb.save_tracks([tr], c.tracks_path)
    t = np.arange(600 * fs) / fs
    cs = physics.path_sound_speed(15, 2, 9000)
    R = independent_range(c.t0 + t, cpa, origin, velocity, observer, cs)
    rng = np.random.default_rng(7)
    sf.write(str(c.audio_path), (.001 + 3000 / R * 10 ** (-ALPHA * R / 20)) * rng.normal(size=len(t)), fs)
    r = jets.run(c, windows=[(180, 420)], step_m=2500, timing_slack_s=0, log=lambda *a: None)
    assert r['status'] == 'ambiguous' and r['n_supported'] == 1
    out = c.dir / 'jets'
    assert (out / 'grid.npz').exists()
    with np.load(out / 'grid.npz') as grid:
        assert grid['near_best'].sum() > 1
    assert all(np.isfinite(a['predicted_db']).all() for a in r['assignments'] if 'predicted_db' in a)
    cli.main(['jets', str(c.dir), '--window', '180', '420', '--step', '2500', '--timing-slack', '0'])
    saved = json.loads((out / 'result.json').read_text())
    assert saved['status'] == 'ambiguous' and saved['n_supported'] == 1
    assert saved['inputs_sha256']['tracks'] == hashlib.sha256(c.tracks_path.read_bytes()).hexdigest()
    assert (out / 'result.sha256').read_text().strip() == hashlib.sha256((out / 'result.json').read_bytes()).hexdigest()
    assert 'fitted range envelope' in (out / 'report.html').read_text(encoding='utf-8')


def test_insufficient_flight_coverage_replaces_stale_spatial_output(tmp_path):
    c = make_case(tmp_path, seconds=120)
    rng = np.random.default_rng(2)
    t = np.arange(120 * 4000) / 4000
    sf.write(str(c.audio_path), (.001 + .1 * np.exp(-.5 * ((t - 60) / 15) ** 2)) * rng.normal(size=len(t)), 4000)
    adsb.save_tracks([], c.tracks_path)
    (c.adsb_dir / 'source.json').write_text(json.dumps({'window_utc': [c.t0 - 600, c.t0 + 720],
                                                       'center': list(c.center), 'radius_m': 1000}))
    out = c.dir / 'jets'
    out.mkdir()
    (out / 'grid.npz').write_bytes(b'old grid')
    (out / 'result.json').write_text('{"status":"old hypothesis"}')
    r = jets.run(c, log=lambda *a: None)
    assert r['status'] == 'incomplete_adsb' and r['location_estimate'] is None
    assert json.loads((out / 'result.json').read_text())['status'] == 'incomplete_adsb'
    assert not (out / 'grid.npz').exists()


def test_local_impacts_are_gain_invariant_and_do_not_discard_a_loud_pass(tmp_path):
    fs = 11025
    t = np.arange(180 * fs) / fs
    rng = np.random.default_rng(21)
    background = rng.normal(size=len(t))
    level = .001 + .03 * np.exp(-.5 * ((t - 95) / 18) ** 2)
    smooth = level * background
    mixed = smooth.copy()
    mixed[(t >= 93) & (t < 93.1)] *= 12
    paths = [tmp_path / name for name in ('smooth.flac', 'mixed.flac', 'quiet.flac')]
    for path, x in zip(paths, (smooth, mixed, mixed / 8)):
        sf.write(str(path), x, fs, subtype='PCM_24')
    clean, impacted, quiet = [jets.envelope(path) for path in paths]
    assert not any(clean['fast']['impulse'])  # loud, sustained roar is allowed
    assert any(impacted['fast']['impulse'])
    assert impacted['fast']['impulse'] == quiet['fast']['impulse']
    assert np.allclose(impacted['clean_activity_db'], quiet['clean_activity_db'], atol=.02)
    assert any(a <= 93.05 <= b for a, b in impacted['transient_intervals_s'])
    events = jets.find_events(impacted)
    event = next(e for e in events if abs(e['peak_s'] - 95) < 12)
    assert 'fast_transient_contamination' in event['flags']
    assert event['screening']['impulse_frames'] >= 1
    assert np.max(impacted['clean_activity_db']) > 15
    rel, y = jets._observations(impacted, event)
    assert not any(a - 1 <= x <= b + 1 for x in rel for a, b in impacted['transient_intervals_s'])
    assert len(rel) >= 5 and np.isfinite(y).all()
    assert np.allclose(clean['clean_activity_db'], impacted['clean_activity_db'], atol=1)


def review_export(c, events):
    duration = c.audio_info()['duration_s']
    return {'recording': c.dir.name, 'audio_start_utc': c.s['audio_start_utc'],
            'candidates': [{'id': e['id'], 'candidate_clip_start_s': max(0, e['t_start'] - 20),
                            'candidate_clip_end_s': min(duration, e['t_end'] + 20),
                            'label': 'likely_jet' if i == 0 else 'unreviewed',
                            'notes': '<script>handling?</script>' if i else 'Sustained roar'}
                           for i, e in enumerate(events)]}


def test_review_selection_preserves_rejects_and_checks_recording_and_windows(tmp_path):
    c = make_case(tmp_path, seconds=180)
    events = [{'id': 'J01', 't_start': 10, 't_end': 50},
              {'id': 'J02', 't_start': 100, 't_end': 140}]
    data = review_export(c, events)
    path = tmp_path / 'labels.json'
    path.write_text(json.dumps(data))
    assert [e['id'] for e in jets.apply_review(c, events, path)] == ['J01']
    assert events[1]['review']['notes'] == '<script>handling?</script>'
    assert not events[1]['review']['selected']
    for mutate in (lambda d: d.update(recording='different-recording'),
                   lambda d: d.update(audio_start_utc='2026-06-02T12:00:00Z'),
                   lambda d: d['candidates'][0].update(candidate_clip_start_s=999),
                   lambda d: d['candidates'].append(d['candidates'][0]),
                   lambda d: d['candidates'][0].update(label='verified_airliner'),
                   lambda d: d['candidates'][0].update(candidate_clip_end_s=float('nan'))):
        changed = json.loads(json.dumps(data))
        mutate(changed)
        path.write_text(json.dumps(changed))
        with pytest.raises(jets.JetError):
            jets.apply_review(c, events, path)


def test_cli_review_selection_persists_with_hash_and_escaped_notes(tmp_path):
    c = make_case(tmp_path, seconds=300)
    fs = 4000
    t = np.arange(300 * fs) / fs
    rng = np.random.default_rng(51)
    level = .001 + .04 * np.exp(-.5 * ((t - 60) / 12) ** 2) + .02 * np.exp(-.5 * ((t - 220) / 14) ** 2)
    sf.write(str(c.audio_path), level * rng.normal(size=len(t)), fs)
    events = jets.find_events(jets.envelope(c.audio_path))
    assert len(events) == 2
    path = tmp_path / 'labels.json'
    path.write_text(json.dumps(review_export(c, events)))
    cli.main(['jets', str(c.dir), '--review-labels', str(path)])
    out = c.dir / 'jets'
    r = json.loads((out / 'result.json').read_text())
    assert r['n_events'] == 1 and r['n_candidates'] == 2
    assert r['review_selected_ids'] == ['J01']
    assert r['inputs_sha256']['listening_review'] == hashlib.sha256(path.read_bytes()).hexdigest()
    saved = json.loads((out / 'events.json').read_text())
    assert [e['id'] for e in saved if e['selected_for_matching']] == ['J01']
    assert len(saved) == 2
    html = (out / 'report.html').read_text(encoding='utf-8')
    assert '&lt;script&gt;handling?&lt;/script&gt;' in html
    assert '<script>handling?</script>' not in html
    assert (out / 'result.sha256').read_text().strip() == hashlib.sha256((out / 'result.json').read_bytes()).hexdigest()
    # Reload through the CLI; combining explicit IDs intersects the reviewed positives.
    cli.main(['jets', str(c.dir), '--review-labels', str(path), '--events', 'J02'])
    assert json.loads((out / 'result.json').read_text())['status'] == 'no_events'
    with pytest.raises(jets.JetError):
        jets.run(c, review_labels=path, windows=[(30, 90)], log=lambda *a: None)
