"""Source screening on synthetic sounds: internal consistency only, not accuracy on real recordings."""
import numpy as np
import soundfile as sf
from scipy import signal

from adoppler import classify

FS = 8000


def _noise(n, rng, lo=None, hi=None):
    x = rng.standard_normal(n)
    if lo or hi:
        sos = signal.butter(4, [lo, hi] if lo and hi else (hi or lo), 'bandpass' if lo and hi else
                            ('lowpass' if hi else 'highpass'), fs=FS, output='sos')
        x = signal.sosfilt(sos, x)
    return x / x.std()


def _write(tmp_path, x, name):
    p = tmp_path / name
    sf.write(str(p), (0.3 * x / np.abs(x).max()).astype(np.float32), FS)
    return p


def _background(n, rng):
    return 0.05 * _noise(n, rng, 50, 3000)


def test_smooth_lowpassed_pass_screens_as_jet(tmp_path):
    rng = np.random.default_rng(1)
    t = np.arange(300 * FS) / FS
    gain_db = 26 - 10 * np.log10(1 + ((t - 150) / 20) ** 2)   # distant source passing, no tones
    x = _background(len(t), rng) + 0.05 * 10 ** (gain_db / 20) * _noise(len(t), rng, 80, 400)
    bl = classify.band_levels(_write(tmp_path, x, 'jet.wav'))
    f = classify.features(bl, 90, 210)
    assert abs(f['peak_s'] - 150) < 5
    assert f['burst_db'] < 2.2 and f['hf_minus_lf_db'] < -10 and f['pass_s'] > 40
    assert classify.screen(f)['class'] == 'jet'


def test_bursty_traffic_with_engine_tones_is_not_aircraft(tmp_path):
    rng = np.random.default_rng(2)
    t = np.arange(300 * FS) / FS
    x = _background(len(t), rng)
    for t0, f0 in ((120, 37.0), (135, 44.0), (150, 31.0), (162, 40.0)):   # a short queue of vehicles
        env = 10 ** ((22 - 10 * np.log10(1 + ((t - t0) / 3) ** 2)) / 20)
        tone = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 12))
        x += 0.05 * env * (0.6 * _noise(len(t), rng, 80, 2500) + 0.5 * tone)
    bl = classify.band_levels(_write(tmp_path, x, 'road.wav'))
    f = classify.features(bl, 100, 180)
    s = classify.screen(f)
    assert s['class'] == 'not_aircraft' and s['jet_score'] < 0.5 and s['reasons']


def test_prop_needs_long_harmonic_doppler_series():
    base = {'burst_db': 2.0, 'tonal_excess_db': 4.0, 'hf_minus_lf_db': 0.0, 'pass_s': 45.0, 'peak_excess_db': 30.0,
            'fall_s': 20.0, 'impulse_frames': 0}
    long_series = dict(base, doppler_event={'id': 'E01', 'n_tracks': 27, 'duration_s': 127.0})
    short_series = dict(base, doppler_event={'id': 'E04', 'n_tracks': 11, 'duration_s': 36.0})
    assert classify.screen(long_series)['class'] == 'prop'
    assert classify.screen(long_series)['tonal_event'] == 'E01'
    assert classify.screen(short_series)['class'] == 'not_aircraft'
    assert classify.screen(dict(base, doppler_event=None))['class'] == 'not_aircraft'


def test_impulsive_handling_noise():
    f = {'burst_db': 5.0, 'tonal_excess_db': -1.0, 'hf_minus_lf_db': -12.0, 'pass_s': 10.0,
         'peak_excess_db': 40.0, 'fall_s': 0.7, 'impulse_frames': 18, 'doppler_event': None}
    assert classify.screen(f)['class'] == 'handling'


def test_harmonic_spacing():
    assert abs(classify.harmonic_spacing([36.9 * k for k in (2, 3, 4, 6, 7, 9)]) - 36.9) < 0.3
    assert classify.harmonic_spacing([100.0, 130.0]) is None
