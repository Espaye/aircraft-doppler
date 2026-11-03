"""Synthetic test cases: aircraft passes rendered to audio, plus the ADS-B tracks a receiver would have logged.

The audio is rendered with physics that the matcher does not model, so the inversion is not simply run on its
own forward model: a uniform wind, a ground reflection, engine speed drift, ADS-B position noise and gaps, a
recording latency, a ground temperature that differs from the case setting, an audible aircraft without ADS-B,
a road vehicle, a mains hum, bird-like chirps and wind noise. Results on these cases show that the pipeline is
internally consistent; they say nothing about accuracy on real recordings.
"""
import dataclasses, datetime as dt, json
from pathlib import Path

import numpy as np
import soundfile as sf

from . import adsb, config, geo, physics


@dataclasses.dataclass
class Flight:
    callsign: str
    icao24: str
    icao_type: str
    desc: str
    cpa_s: float            # emission time (s from audio start, acoustic clock) at the closest approach
    heading_deg: float
    speed_mps: float
    alt_agl_m: float        # height above the microphone at the closest approach
    offset_m: float         # horizontal closest-approach distance (signed: + = track passes east/right)
    bpf_hz: float = 0.0     # blade-pass (or engine firing) frequency; 0 = silent
    harmonics: tuple = (1.0, 0.6, 0.35, 0.2)
    level: float = 1.0      # relative source amplitude
    climb_mps: float = 0.0
    turn_dps: float = 0.0
    adsb: bool = True       # False: audible but not in the ADS-B data (e.g. no transponder)
    ground: bool = False    # road vehicle (height fixed near the ground)


# A 20-minute scene. Positions are relative to the hidden microphone; nothing here refers to a real recording.
EXAMPLE = {
    'name': 'Synthetic example (20 min)',
    'search_center': [52.1000, 5.8000],     # prior: a nominal centre ~5 km from the hidden microphone
    'mic': [52.0712, 5.8521],               # hidden truth (open heathland, chosen arbitrarily)
    'ground_elevation_m': 40.0, 'geoid_undulation_m': 43.5,
    'start_utc': '2026-06-14T10:00:00Z', 'duration_s': 1200.0, 'samplerate': 11025,
    'delay_s': 1.6, 'temp_true_c': 21.0, 'temp_case_c': 18.0, 'wind_mps': [4.0, 1.0],
    'search_radius_km': 15.0,
    'flights': [
        Flight('SYN101', 'f00101', 'C172', 'CESSNA 172', 130, 75, 56, 650, 420, 82.0, level=1.0),
        Flight('SYN202', 'f00202', 'SR22', 'CIRRUS SR22', 300, 210, 78, 1100, -1500, 131.0, level=1.6, turn_dps=0.4),
        Flight('SYN303', 'f00303', 'TBM9', 'SOCATA TBM-900', 480, 340, 135, 2600, 2200, 166.0, level=3.5,
               climb_mps=-3.0),
        Flight('NOXPDR', 'f00404', 'C150', 'CESSNA 150 (no ADS-B)', 610, 120, 48, 500, 700, 78.0, level=0.8,
               adsb=False),
        Flight('SYN505', 'f00505', 'P28A', 'PIPER PA-28', 760, 10, 52, 850, -900, 79.0, level=1.1, climb_mps=2.5),
        Flight('CAR', 'f00606', 'CAR', 'road vehicle', 900, 160, 24, 0.5, 55, 98.0, (1.0, 0.5, 0.3), level=0.05,
               adsb=False, ground=True),
        Flight('SYN707', 'f00707', 'BE20', 'BEECH 200 KING AIR', 1010, 285, 112, 2300, -3000, 113.0, level=3.0),
        # in the ADS-B data but not audible (silent): candidates that must not be matched
        Flight('SYN808', 'f00808', 'A20N', 'AIRBUS A320neo', 470, 300, 230, 10500, 1500, 0.0),
        Flight('SYN909', 'f00909', 'B738', 'BOEING 737-800', 1000, 120, 220, 9000, -2500, 0.0),
        Flight('SYN010', 'f00010', 'C182', 'CESSNA 182', 640, 30, 62, 900, 14000, 0.0),
    ],
}

# A 6-minute scene for automated tests (fast to render and to invert).
MINI = {
    'name': 'Synthetic mini (6 min)',
    'search_center': [52.1000, 5.8000], 'mic': [52.0890, 5.8300],
    'ground_elevation_m': 40.0, 'geoid_undulation_m': 43.5,
    'start_utc': '2026-06-14T12:00:00Z', 'duration_s': 360.0, 'samplerate': 11025,
    'delay_s': 1.2, 'temp_true_c': 16.0, 'temp_case_c': 15.0, 'wind_mps': [3.0, 0.0],
    'search_radius_km': 6.0,
    'flights': [
        Flight('MIN101', 'f10101', 'C172', 'CESSNA 172', 70, 80, 56, 600, 350, 84.0, level=1.0),
        Flight('MIN202', 'f10202', 'SR22', 'CIRRUS SR22', 180, 200, 78, 1000, -1200, 129.0, level=1.6),
        Flight('MIN303', 'f10303', 'BE20', 'BEECH 200 KING AIR', 290, 320, 110, 1800, 1900, 113.0, level=3.0),
        Flight('MIN404', 'f10404', 'A20N', 'AIRBUS A320neo', 200, 300, 230, 10500, 800, 0.0),
    ],
}

SCENES = {'example': EXAMPLE, 'mini': MINI}


def _path(fl, mic, t0, t1, dt_s=0.1, rng=None):
    """True trajectory (ENU, metres) on a uniform emission-time grid."""
    tau = np.arange(t0, t1 + dt_s / 2, dt_s)
    s = tau - fl.cpa_s
    hdg = np.radians(fl.heading_deg + fl.turn_dps * s)
    vx, vy = fl.speed_mps * np.sin(hdg), fl.speed_mps * np.cos(hdg)
    i0 = int(np.argmin(np.abs(s)))
    x = np.cumsum(vx) * dt_s; y = np.cumsum(vy) * dt_s
    x -= x[i0]; y -= y[i0]
    h0 = np.radians(fl.heading_deg)
    x += mic[0] + fl.offset_m * np.cos(h0); y += mic[1] - fl.offset_m * np.sin(h0)   # offset perpendicular to track
    z = mic[2] + fl.alt_agl_m + fl.climb_mps * s
    if fl.ground:
        z = np.full_like(s, mic[2] - 1.0)
    return tau, np.stack([x, y, z], 1)


def _emission_time(tau_g, P, obs, t_rx, c, wind, n_iter=6):
    """Solve |obs - p(tau) - u (t - tau)| = c (t - tau) for tau (sound carried by a uniform wind u)."""
    def at(t):
        return np.stack([np.interp(t, tau_g, P[:, k]) for k in range(3)], 1)
    V = np.gradient(P, tau_g, axis=0)
    tau = t_rx - np.linalg.norm(at(t_rx) - obs, axis=1) / c
    for _ in range(n_iter):
        p = at(tau); v = np.stack([np.interp(tau, tau_g, V[:, k]) for k in range(3)], 1)
        T = t_rx - tau
        D = obs - p - wind * T[:, None]
        R = np.linalg.norm(D, axis=1)
        F = R - c * T
        dF = (-(D * (v - wind)).sum(1) / R) + c           # dF/dtau
        tau = tau - F / dF
    p = at(tau)
    return tau, np.linalg.norm(obs - p, axis=1)


def _render(fl, tau_g, P, obs, image, t_rec, delay, c, wind, fs, rng, refl=0.6):
    """Pressure at the microphone from one source (direct ray + ground reflection), recorded clock t_rec."""
    out = np.zeros(len(t_rec))
    if fl.bpf_hz <= 0:
        return out
    # engine speed drift: slow random walk of +/- ~0.3 %, integrated to phase on the emission grid
    k = max(4, int((tau_g[-1] - tau_g[0]) / 20))
    knots = np.cumsum(rng.normal(0, 0.0012, k)); knots -= knots.mean()
    f_src = fl.bpf_hz * (1 + np.interp(tau_g, np.linspace(tau_g[0], tau_g[-1], k), knots))
    phase = np.concatenate([[0.0], np.cumsum(0.5 * (f_src[1:] + f_src[:-1]) * np.diff(tau_g))])
    t_ac = t_rec - delay
    # coarse reception grid where the source is plausibly audible, then interpolate to audio samples
    tc = np.arange(max(t_ac[0], tau_g[0] + 1), min(t_ac[-1], tau_g[-1]), 0.02)
    if len(tc) < 10:
        return out
    rays = [_emission_time(tau_g, P, np.asarray(o, float)[None, :], tc, c, wind) for o in (obs, image)]
    i0, i1 = np.searchsorted(t_ac, [tc[0], tc[-1]])
    for a in range(i0, i1, 60 * fs):                       # one minute of samples at a time (memory)
        b = min(i1, a + 60 * fs)
        ta = t_ac[a:b]
        fade = np.clip(np.minimum(ta - tc[0], tc[-1] - ta) / 20.0, 0, 1)
        for (tau, R), gain in zip(rays, (1.0, refl)):
            R_a = np.interp(ta, tc, R)
            ph = np.interp(np.interp(ta, tc, tau), tau_g, phase)
            amp = gain * fl.level * 4.0 / np.maximum(R_a, 5.0) * fade
            for h, w in enumerate(fl.harmonics, 1):
                absorb = np.exp(-R_a * 5e-7 * (h * fl.bpf_hz))   # crude air absorption, stronger at high f
                out[a:b] += w * amp * absorb * np.sin(2 * np.pi * h * ph + h * 0.7)
    return out


def _background(n, fs, rng, duration):
    """Pink-ish noise with wind gusts, a weak mains hum and bird-like chirps."""
    X = np.fft.rfft(rng.normal(0, 1, n))
    f = np.fft.rfftfreq(n, 1 / fs)
    X *= 1 / np.sqrt(np.maximum(f, 20.0) / 20.0)
    x = np.fft.irfft(X, n)
    x *= 0.005 / x.std()
    t = np.arange(n) / fs
    gust = np.interp(t, np.linspace(0, duration, int(duration / 7) + 2), rng.uniform(0.3, 1.0, int(duration / 7) + 2))
    W = np.fft.rfft(rng.normal(0, 1, n)); W[f > 40] = 0
    wind = np.fft.irfft(W, n); wind *= 0.03 / wind.std()
    x += wind * gust
    x += 0.0015 * np.sin(2 * np.pi * 50 * t) + 0.0010 * np.sin(2 * np.pi * 150 * t + 1)
    for _ in range(int(duration / 4)):                    # chirps: 0.1-0.4 s sweeps, 900-1700 Hz
        t0 = rng.uniform(0, duration - 1); d = rng.uniform(0.1, 0.4)
        f0, f1 = rng.uniform(900, 1700, 2)
        a, b = int(t0 * fs), int((t0 + d) * fs)
        tt = np.arange(b - a) / fs
        x[a:b] += rng.uniform(0.002, 0.02) * np.sin(2 * np.pi * (f0 * tt + 0.5 * (f1 - f0) / d * tt * tt)) * \
            np.hanning(b - a)
    return x


def _adsb_track(fl, tau_g, P, enu, geoid_n, t_utc0, rng, t_lo, t_hi):
    """What a receiver logs: irregular 0.5-3 s reports with gaps, ~8 m position noise, 25 ft altitude steps."""
    ts = [t_lo]
    while ts[-1] < t_hi:
        ts.append(ts[-1] + (rng.uniform(8, 25) if rng.random() < 0.02 else rng.uniform(0.5, 3.0)))
    ts = np.array(ts[1:-1])
    ts = ts[(ts >= tau_g[0]) & (ts <= tau_g[-1])]
    p = np.stack([np.interp(ts, tau_g, P[:, k]) for k in range(3)], 1)
    p[:, :2] += rng.normal(0, 8.0, (len(ts), 2))
    lat, lon, h = enu.to_geodetic(p)
    alt = np.round((h - geoid_n) / config.FT / 25) * 25 * config.FT     # geometric altitude, 25 ft steps
    gs = np.full(len(ts), fl.speed_mps / config.KT) + rng.normal(0, 1, len(ts))
    return adsb.Track(fl.icao24, fl.callsign, fl.icao_type, None, 'synthetic', ts + t_utc0, lat, lon, alt,
                      np.array(['geom'] * len(ts)), meta={'registration': fl.callsign, 'desc': fl.desc,
                                                          'gs_kt': gs.round(1).tolist()})


def make_case(out_dir, scene='example', seed=1, log=print):
    """Render a synthetic case directory (audio, ADS-B file, case.json, truth.json). Returns the directory."""
    sc = SCENES[scene] if isinstance(scene, str) else scene
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    lat0, lon0 = sc['search_center']
    N = sc['geoid_undulation_m']; zg = sc['ground_elevation_m']
    enu = geo.ENU(lat0, lon0, zg + N)
    mic_h = 2.0
    m = enu.from_geodetic(sc['mic'][0], sc['mic'][1], zg + N + mic_h)
    image = m - np.array([0, 0, 2 * mic_h])
    fs, D = sc['samplerate'], sc['duration_s']
    n = int(D * fs)
    t_rec = np.arange(n) / fs
    wind = np.array([*sc['wind_mps'], 0.0])
    t_utc0 = dt.datetime.fromisoformat(sc['start_utc'].replace('Z', '+00:00')).timestamp()
    log(f"rendering {D / 60:.0f} min at {fs} Hz, {len(sc['flights'])} sources")
    x = _background(n, fs, rng, D)
    tracks, truth_fl = [], []
    for fl in sc['flights']:
        t_lo, t_hi = -600.0, D + 600.0
        tau_g, P = _path(fl, m, max(t_lo, fl.cpa_s - 1500), min(t_hi, fl.cpa_s + 1500))
        c = float(physics.path_sound_speed(sc['temp_true_c'], m[2], np.median(P[:, 2])))
        x += _render(fl, tau_g, P, m, image, t_rec, sc['delay_s'], c, wind, fs, rng)
        if fl.adsb:
            tracks.append(_adsb_track(fl, tau_g, P, enu, N, t_utc0, rng, t_lo, t_hi))
        truth_fl.append({'callsign': fl.callsign, 'icao24': fl.icao24, 'type': fl.icao_type, 'adsb': fl.adsb,
                         'audible': fl.bpf_hz > 0, 'cpa_utc': t_utc0 + fl.cpa_s, 'bpf_hz': fl.bpf_hz,
                         'closest_m': float(np.hypot(fl.offset_m, fl.alt_agl_m))})
    peak = np.abs(x).max()
    x = (x / peak * 0.8).astype(np.float32)
    sf.write(str(out / 'audio.flac'), x, fs, subtype='PCM_16')
    adsb.save_tracks(tracks, out / 'flights.json')
    case = {'name': sc['name'], 'audio': 'audio.flac', 'audio_start_utc': sc['start_utc'],
            'search_center': sc['search_center'], 'search_radius_km': sc['search_radius_km'],
            'ground_temp_c': sc['temp_case_c'], 'terrain': 'flat', 'ground_elevation_m': zg,
            'geoid_undulation_m': N, 'adsb_source': 'file', 'adsb_file': 'flights.json',
            'synthetic': True}
    if scene == 'mini' or sc is MINI:
        case['match'] = {'coarse_step_m': 500.0, 'null_tests': 1, 'modes': 2}
    (out / 'case.json').write_text(json.dumps(case, indent=1), encoding='utf-8')
    truth = {'note': 'Hidden truth of a synthetic case. Compare only after `adoppler match`.',
             'mic_lat': sc['mic'][0], 'mic_lon': sc['mic'][1], 'mic_E_m': float(m[0]), 'mic_N_m': float(m[1]),
             'delay_s': sc['delay_s'], 'ground_temp_c': sc['temp_true_c'], 'wind_mps': sc['wind_mps'],
             'seed': seed, 'flights': truth_fl}
    (out / 'truth.json').write_text(json.dumps(truth, indent=1), encoding='utf-8')
    log(f'wrote {out}')
    return out
