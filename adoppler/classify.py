"""Source screening for sound candidates: aircraft (jet / propeller) versus road traffic and other noise.

Features are computed from the audio around each candidate and describe what a listener uses: how slowly the
sound builds up and dies away, how smooth the level is, where its energy sits in the spectrum, and whether it
carries tones with a gentle Doppler shift. See docs/method.md ("Source screening").
"""
import json

import numpy as np
import soundfile as sf
from scipy import ndimage as ndi
from scipy import signal

HOP_S = 0.1
NFFT = 2048
THIRD_OCT = np.array([63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000],
                     float)


def band_levels(audio, hop_s=HOP_S, nfft=NFFT, chunk_s=60.0):
    """Third-octave band levels (dB, uncalibrated) and a per-frame tonality measure, streamed in chunks."""
    with sf.SoundFile(str(audio)) as fh:
        fs = fh.samplerate
        hop = int(round(hop_s * fs))
        n_total = len(fh)
        f = np.fft.rfftfreq(nfft, 1 / fs)
        fc = THIRD_OCT[THIRD_OCT * 2 ** (1 / 6) < fs / 2]
        idx = [np.flatnonzero((f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))) for c in fc]
        tband = (f >= 60) & (f <= 1000)
        win = signal.get_window('hann', nfft).astype(np.float32)
        L, T, tonal = [], [], []
        centre = nfft // 2
        chunk = int(chunk_s * fs) // hop * hop
        while centre < n_total - nfft // 2:
            a = centre - nfft // 2
            n_fr = min(chunk // hop, (n_total - nfft // 2 - centre) // hop + 1)
            fh.seek(a)
            x = fh.read((n_fr - 1) * hop + nfft, dtype='float32', always_2d=True).mean(1)
            if len(x) < nfft:
                break
            n_fr = min(n_fr, (len(x) - nfft) // hop + 1)
            fr = np.lib.stride_tricks.sliding_window_view(x, nfft)[::hop][:n_fr] * win
            P = np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-20
            L.append(np.stack([10 * np.log10(P[:, i].mean(1)) for i in idx], 1))
            Pt = P[:, tband]
            ratio = Pt / ndi.median_filter(Pt, size=(1, 21), mode='nearest')
            tonal.append(10 * np.log10(np.percentile(ratio, 98, axis=1)))
            T.append((centre + np.arange(n_fr) * hop) / fs)
            centre += n_fr * hop
    return {'t': np.concatenate(T), 'fc': fc, 'L': np.concatenate(L), 'tonal_db': np.concatenate(tonal),
            'fs': fs, 'hop_s': hop_s}


def _power_sum(L, cols):
    return 10 * np.log10(np.sum(10 ** (L[:, cols] / 10), axis=1))


def _gauss(x, sigma_s, hop_s):
    return ndi.gaussian_filter1d(x, sigma_s / hop_s, mode='nearest')


def _tonal_overlap(start_s, end_s, tonal_events, tonal_tracks, peak_s):
    """Tonal detector evidence inside the candidate: Doppler events and harmonic tracks."""
    out = {'tonal_tracks': 0, 'doppler_event': None}
    for tr in tonal_tracks or []:
        t = np.asarray(tr['t'], float)
        if len(t) and np.mean((t >= start_s) & (t <= end_s)) > 0.5 and abs(np.median(t) - peak_s) <= 60:
            out['tonal_tracks'] += 1
    best = None
    for e in tonal_events or []:
        fit = e.get('fit')
        if not fit or e['label'] != 'aircraft':
            continue
        ov = min(end_s, e['t_end']) - max(start_s, e['t_start'])
        if ov < 0.5 * min(e['t_end'] - e['t_start'], end_s - start_s):
            continue
        if best is None or e['score'] > best['score']:
            best = {'id': e['id'], 'n_tracks': len(e['tracks']), 'duration_s': float(e['t_end'] - e['t_start']), 'v': fit['v'], 'd': fit['d'],
                    'rms_log': fit['rms_log'], 'score': e['score'], 't_cpa_sigma': fit.get('t_cpa_sigma'),
                    'harmonic_spacing_hz': harmonic_spacing(fit.get('f0') or [])}
    out['doppler_event'] = best
    return out


def harmonic_spacing(f0, lo=20.0, hi=200.0):
    """Common fundamental (Hz) of fitted emitted frequencies if most form one harmonic series, else None."""
    f = np.asarray(f0, float)
    f = f[(f > 30) & (f < 1500)]
    if len(f) < 5:
        return None
    cand = np.arange(lo, hi, 0.1)
    k = np.round(f[None, :] / cand[:, None])
    ok = k >= 1
    err = np.abs(f[None, :] - k * cand[:, None]) / cand[:, None]
    frac = np.where(ok, err < 0.04, False).mean(1)
    # prefer the largest fundamental that explains most tracks (sub-harmonics trivially explain everything)
    good = np.flatnonzero(frac >= 0.6)
    return float(round(cand[good[-1]], 1)) if len(good) else None


def pass_fit(t, y, bg):
    """Least-squares fit of one passing point source over a background: level, closest-approach time, width.

    L(t) = bg (+) L0 - 10 log10(1 + ((t - tc) / tau)^2), with separate widths before and after tc (directivity,
    acceleration). Returns parameters and the RMS residual in dB where the model is at least 3 dB above bg.
    """
    from scipy.optimize import least_squares
    ip = int(np.argmax(y))

    def model(x):
        L0, tc, la, lb = x
        tau = np.where(t < tc, np.exp(la), np.exp(lb))
        s = L0 - 10 * np.log10(1 + ((t - tc) / tau) ** 2)
        return 10 * np.log10(10 ** (bg / 10) + 10 ** (s / 10))

    x0 = [float(y[ip]), float(t[ip]), np.log(10.0), np.log(10.0)]
    best = None
    for w0 in (3.0, 10.0, 30.0):
        x0[2] = x0[3] = np.log(w0)
        r = least_squares(lambda x: model(x) - y, x0, loss='soft_l1', f_scale=2.0,
                          bounds=([bg, t[0], np.log(0.3), np.log(0.3)], [y.max() + 10, t[-1], np.log(600), np.log(600)]))
        if best is None or r.cost < best.cost:
            best = r
    L0, tc, la, lb = best.x
    m = model(best.x)
    act = m >= bg + 3
    rms = float(np.sqrt(np.mean((m[act] - y[act]) ** 2))) if act.sum() > 5 else float('nan')
    return {'tc_s': float(tc), 'tau_before_s': float(np.exp(la)), 'tau_after_s': float(np.exp(lb)),
            'level_db': float(L0 - bg), 'fit_rms_db': rms}


def features(bl, start_s, end_s, peak_s=None, tonal_events=None, tonal_tracks=None, impulse_frames=0):
    """Feature dictionary for one candidate window (seconds from recording start)."""
    t, L, fc, hop = bl['t'], bl['L'], bl['fc'], bl['hop_s']
    bb = np.flatnonzero((fc >= 100) & (fc <= 2000))
    B = _power_sum(L, bb)
    bg = float(np.percentile(_gauss(B, 1.0, hop), 10))
    S3, S05, S8 = _gauss(B, 3.0, hop), _gauss(B, 0.5, hop), _gauss(B, 8.0, hop)
    w = np.flatnonzero((t >= start_s) & (t <= end_s))
    if len(w) < 20:
        raise ValueError('candidate window too short for screening')
    ip = int(w[np.argmax(S3[w])]) if peak_s is None else int(np.argmin(np.abs(t - peak_s)))
    if not (w[0] <= ip <= w[-1]):
        ip = int(w[np.argmax(S3[w])])
    peak = float(S3[ip]); excess = peak - bg
    drop = max(3.0, min(10.0, 0.7 * excess))
    i = ip
    while i > 0 and S3[i] > peak - drop:
        i -= 1
    j = ip
    while j < len(t) - 1 and S3[j] > peak - drop:
        j += 1
    rise_s, fall_s = (ip - i) * hop, (j - ip) * hop
    above = S3 > bg + 3
    a = ip
    while a > 0 and above[a - 1]:
        a -= 1
    b = ip
    while b < len(t) - 1 and above[b + 1]:
        b += 1
    audible_s = (b - a) * hop
    clipped = i == 0 or j == len(t) - 1
    core = np.arange(i, j + 1)
    fluct = S05[core] - S8[core]
    burst_db = float(np.std(fluct))
    S15 = _gauss(B, 1.5, hop)
    pk, _ = signal.find_peaks(S15[core], prominence=3.0)
    # spectral excess around the peak relative to the recording's quiet level in each band
    near = np.flatnonzero(np.abs(t - t[ip]) <= 5)
    Pb = 10 ** (np.percentile(L, 10, axis=0) / 10)
    Pp = (10 ** (L[near] / 10)).mean(0)
    ex = 10 * np.log10(np.maximum(Pp - Pb, 1e-3 * Pb) / Pb)
    lo = (fc >= 100) & (fc <= 400); hi = (fc >= 1000) & (fc <= 2000)
    exp = np.maximum(Pp - Pb, 0)
    sel = (fc >= 63) & (fc <= 2000)
    low_share = float(exp[sel & (fc <= 500)].sum() / max(exp[sel].sum(), 1e-30))
    tonal = bl['tonal_db']
    tonal_core = float(np.median(tonal[core]) - np.median(tonal))
    feats = {
        'peak_s': float(t[ip]), 'peak_excess_db': excess, 'rise_s': rise_s, 'fall_s': fall_s,
        'pass_s': rise_s + fall_s, 'drop_db': drop, 'clipped': bool(clipped), 'audible_s': audible_s,
        'burst_db': burst_db, 'core_peaks': int(len(pk)),
        'hf_minus_lf_db': float(np.mean(ex[hi]) - np.mean(ex[lo])) if hi.any() else float('nan'),
        'low_share': low_share, 'tonal_excess_db': tonal_core, 'impulse_frames': int(impulse_frames),
    }
    # one-pass fit on 1 s smoothed level, local background, window padded by 20 s
    fw = np.flatnonzero((t >= start_s - 20) & (t <= end_s + 20))[::5]
    S1 = _gauss(B, 1.0, hop)
    lbg = float(np.percentile(S1[np.flatnonzero(np.abs(t - t[ip]) <= max(90, end_s - start_s))], 10))
    feats.update(pass_fit(t[fw], S1[fw], lbg))
    feats.update(_tonal_overlap(start_s, end_s, tonal_events, tonal_tracks, float(t[ip])))
    return feats


def case_features(case_dir, events=None, bl=None):
    """Features for every broadband candidate of a case (jets/events.json) using the case audio."""
    from pathlib import Path
    case_dir = Path(case_dir)
    cj = json.loads((case_dir / 'case.json').read_text())
    if bl is None:
        bl = band_levels(case_dir / cj['audio'])
    if events is None:
        events = json.loads((case_dir / 'jets' / 'events.json').read_text())
    an = case_dir / 'analysis'
    tev = json.loads((an / 'events.json').read_text())['events'] if (an / 'events.json').exists() else []
    ttr = json.loads((an / 'tracks.json').read_text()) if (an / 'tracks.json').exists() else []
    return {e['id']: features(bl, e['t_start'], e['t_end'], e.get('peak_s'), tev, ttr,
                              (e.get('screening') or {}).get('impulse_frames', 0)) for e in events}, bl


# ---------------------------------------------------------------- screening rule
# Soft tests, each 0..1 (linear between the two limits). Limits were chosen from one owner-verified recording
# (16 sounds: 2 jets, 2 propeller aircraft, 10 road, 1 handling, 1 unsure) and are physical rather than fitted:
#   jets at cruise height arrive smooth, without engine tones and strongly low-passed by air absorption, and
#   take tens of seconds to build up and die away; propeller aircraft carry a long, complete harmonic series
#   with a gentle Doppler shift; road vehicles are irregular, short, or carry engine tones without a long series.
JET_TESTS = (
    ('smooth level', 'burst_db', 3.0, 2.2, 'level fluctuates like traffic ({:.1f} dB short-term)'),
    ('no engine tones', 'tonal_excess_db', 1.0, 0.3, 'engine tones present ({:+.1f} dB)'),
    ('low-passed by distance', 'hf_minus_lf_db', -5.0, -10.0, 'too much high-frequency sound for a distant jet ({:+.1f} dB)'),
    ('slow build-up and decay', 'pass_s', 20.0, 40.0, 'passes too quickly ({:.0f} s within 10 dB of the peak)'),
    ('clearly above background', 'peak_excess_db', 6.0, 10.0, 'barely above background ({:.1f} dB)'),
)
PROP_TESTS = (
    ('long harmonic series', 'n_tracks', 10.0, 20.0, 'few harmonics with a Doppler shift ({:.0f})'),
    ('long steady Doppler curve', 'duration_s', 45.0, 90.0, 'Doppler curve too short ({:.0f} s)'),
    ('tones stand out', 'tonal_excess_db', 0.5, 1.5, 'weak tones ({:+.1f} dB)'),
    ('not traffic-like', 'burst_db', 4.0, 3.0, 'level fluctuates like traffic ({:.1f} dB short-term)'),
)


def _ramp(x, a, b):
    if x is None or not np.isfinite(x):
        return 0.0
    return float(np.clip((x - a) / (b - a), 0, 1))


def _score(f, tests):
    parts, why = {}, []
    for name, key, a, b, msg in tests:
        v = f.get(key)
        parts[name] = _ramp(v, a, b)
        if parts[name] < 0.5:
            why.append(msg.format(v if v is not None else float('nan')))
    return float(np.prod(list(parts.values())) ** (1 / len(parts))), parts, why


def screen(f):
    """Source class for one candidate's features: jet, propeller, handling (phone/impact) or not_aircraft."""
    de = f.get('doppler_event') or {}
    pf = dict(f, n_tracks=de.get('n_tracks', 0), duration_s=de.get('duration_s', 0))
    jet, jet_parts, jet_why = _score(f, JET_TESTS)
    prop, prop_parts, prop_why = _score(pf, PROP_TESTS) if de else (0.0, {}, ['no aircraft-like Doppler tones'])
    handling = f['burst_db'] >= 4.5 and (f.get('impulse_frames', 0) >= 8 or f['fall_s'] < 1.5)
    if handling:
        cls, why = 'handling', ['impulsive and very irregular: handling, impacts or wind on the microphone']
    elif jet >= 0.5 and jet >= prop:
        cls, why = 'jet', []
    elif prop >= 0.5:
        cls, why = 'prop', []
    else:
        cls, why = 'not_aircraft', jet_why[:2] + [w for w in prop_why if w not in jet_why][:2]
    return {'class': cls, 'jet_score': round(jet, 3), 'prop_score': round(prop, 3), 'jet_tests': jet_parts,
            'prop_tests': prop_parts, 'reasons': why,
            'tonal_event': de.get('id') if cls == 'prop' else None}


# ---------------------------------------------------------------- case command

def run(case, log=print):
    """Screen every broadband candidate of a case; writes classify/sources.json and returns its content."""
    from . import jets
    jd = case.dir / 'jets'
    if not (jd / 'events.json').exists():
        jd.mkdir(exist_ok=True)
        env = jets.envelope(case.audio_path)
        (jd / 'envelope.json').write_text(json.dumps(env), encoding='utf-8')
        (jd / 'events.json').write_text(json.dumps(jets.find_events(env), indent=1), encoding='utf-8')
    events = json.loads((jd / 'events.json').read_text(encoding='utf-8'))
    tonal = (case.dir / 'analysis' / 'events.json').exists()
    feats, _ = case_features(case.dir, events)
    rows = []
    for e in events:
        f = feats[e['id']]
        s = screen(f)
        rows.append({'id': e['id'], 't_start': e['t_start'], 't_end': e['t_end'], 'peak_s': e['peak_s'], **s,
                     'features': {k: v for k, v in f.items() if k != 'doppler_event'},
                     'doppler_event': f['doppler_event']})
    out = {'audio_start_utc': case.s['audio_start_utc'], 'tonal_analysis_used': tonal, 'sources': rows,
           'jets': [r['id'] for r in rows if r['class'] == 'jet'],
           'props': [r['id'] for r in rows if r['class'] == 'prop'],
           'prop_tonal_events': sorted({r['tonal_event'] for r in rows if r['class'] == 'prop' and r['tonal_event']}),
           'note': 'Rule-based screening from envelope smoothness, spectrum and tones; set on one owner-verified '
                   'recording. Listen before trusting a class.'}
    d = case.dir / 'classify'
    d.mkdir(exist_ok=True)
    (d / 'sources.json').write_text(json.dumps(out, indent=1, default=float), encoding='utf-8')
    for r in rows:
        why = '; '.join(r['reasons'])
        log(f"  {r['id']} {r['t_start'] / 60:5.1f}-{r['t_end'] / 60:5.1f} min  {r['class']:12s} "
            f"jet {r['jet_score']:.2f} prop {r['prop_score']:.2f}{'  ' + why if why else ''}")
    if not tonal:
        log('  (no tonal analysis yet: run `adoppler analyse` so propeller aircraft can be recognised)')
    return out


def load_sources(case, kind):
    """IDs screened as `kind` ('jet' or 'prop'); checks the screening belongs to the current candidates."""
    p = case.dir / 'classify' / 'sources.json'
    if not p.exists():
        raise ValueError('no screening yet: run `adoppler classify` first')
    d = json.loads(p.read_text(encoding='utf-8'))
    if d['audio_start_utc'] != case.s['audio_start_utc']:
        raise ValueError('screening belongs to a different recording start; run `adoppler classify` again')
    ev = case.dir / 'jets' / 'events.json'
    cur = {e['id']: (round(e['t_start'], 1), round(e['t_end'], 1)) for e in json.loads(ev.read_text(encoding='utf-8'))}
    if any(cur.get(r['id']) != (round(r['t_start'], 1), round(r['t_end'], 1)) for r in d['sources']):
        raise ValueError('broadband candidates changed since screening; run `adoppler classify` again')
    return d['jets'] if kind == 'jet' else d['prop_tonal_events']
