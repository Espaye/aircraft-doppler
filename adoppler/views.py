"""Read-only data for the browser UI and the static export: spectrogram images, events, flights, results."""
import io, json, struct, zlib

import numpy as np

from . import adsb, geo, spectro

# perceptual dark-to-light map (approximates 'magma'), 0..1 -> RGB
_CMAP = np.array([[0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122], [229, 80, 100],
                  [251, 135, 97], [254, 194, 135], [252, 253, 191]], float)


def _colour(x):
    x = np.clip(x, 0, 1) * (len(_CMAP) - 1)
    i = np.minimum(x.astype(int), len(_CMAP) - 2); u = (x - i)[..., None]
    return (_CMAP[i] * (1 - u) + _CMAP[i + 1] * u).astype(np.uint8)


def png(rgb):
    """Minimal PNG encoder for an (H, W, 3) uint8 array."""
    h, w, _ = rgb.shape
    raw = b''.join(b'\x00' + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def spectrogram(case, t0=None, t1=None, f0=0.0, f1=None, width=1600, height=600, kind='tonal'):
    """PNG of the tonal-enhanced (dB above local noise) or power spectrogram, max-pooled to width x height.
    Returns (png bytes, extent dict with t0, t1 (s from audio start), f0, f1 (Hz))."""
    P, t, f, m = spectro.load(case.analysis_dir)
    if kind == 'tonal':
        A = np.load(case.analysis_dir / 'tonal_T.npy', mmap_mode='r'); lo, hi = 5.0, 20.0
    else:
        A = P; lo = hi = None
    t0 = t[0] if t0 is None else max(float(t0), t[0]); t1 = t[-1] if t1 is None else min(float(t1), t[-1])
    f1 = min(f[-1], 1000.0 if f1 is None else float(f1)); f0 = max(0.0, float(f0))
    i0, i1 = np.searchsorted(t, [t0, t1]); j0, j1 = np.searchsorted(f, [f0, f1])
    i1 = max(i1, i0 + 2); j1 = max(j1, j0 + 2)
    nt, nf = i1 - i0, j1 - j0
    bt, bf = max(1, -(-nt // width)), max(1, -(-nf // height))
    rows = []
    for a in range(i0, i1, bt * 512):                     # chunks: memmaps of long recordings stay on disk
        b = min(i1, a + bt * 512)
        X = np.asarray(A[a:b, j0:j1], np.float32)
        if kind != 'tonal':
            X = 10 * np.log10(X + 1e-20)
        n = (len(X) // bt) * bt
        if n:
            X = X[:n].reshape(-1, bt, X.shape[1]).max(1)
            rows.append(X)
    X = np.concatenate(rows)
    n = (X.shape[1] // bf) * bf
    X = X[:, :n].reshape(len(X), -1, bf).max(2)
    if lo is None:
        lo, hi = np.percentile(X, [20, 99.7]); lo = hi - max(30.0, hi - lo)
    img = _colour((X.T[::-1] - lo) / (hi - lo))           # rows: high frequency first
    ext = {'t0': float(t[i0]), 't1': float(t[i0 + len(X) * bt - 1]), 'f0': float(f[j0]),
           'f1': float(f[j0 + n - 1]), 'kind': kind, 'db_range': [float(lo), float(hi)]}
    return png(img), ext


def events(case, max_pts=400):
    """Detected events, thinned, plus which ones the match will use (by the case settings)."""
    d = json.loads(case.events_path.read_text())
    ms = case.s['match']
    out = []
    for e in d['events']:
        trs = []
        for tr in e['tracks']:
            k = max(1, len(tr['t']) // max_pts)
            trs.append({'t': tr['t'][::k], 'f': tr['f'][::k], 'snr': tr['snr'][::k]})
        fit = e.get('fit') or {}
        usable = e['label'] == 'aircraft' and e['score'] >= ms['min_score'] and fit.get('t_cpa_sigma', 99) < 30
        sel = ms.get('events')
        out.append({'id': e['id'], 'label': e['label'], 'score': e['score'], 'reasons': e.get('reasons', []),
                    't_start': e['t_start'], 't_end': e['t_end'], 'fit': fit, 'tracks': trs,
                    'used': (e['id'] in sel) if sel is not None else usable,
                    'manual': (case.manual_dir / f"{e['id']}.json").exists()})
    return {'t0_utc': case.t0, 'counts': d.get('counts'), 'events': out}


def flights(case, max_pts=300):
    """ADS-B paths (lat, lon, MSL altitude) within the recording window +/- 10 min, thinned for display."""
    tracks = adsb.load_tracks(case.tracks_path)
    a, b = case.window
    a -= 600; b += 600
    out = []
    for tr in tracks:
        m = (tr.t >= a) & (tr.t <= b) & np.isfinite(tr.lat)
        if m.sum() < 2:
            continue
        idx = np.flatnonzero(m)
        idx = idx[::max(1, len(idx) // max_pts)]
        out.append({'icao24': tr.icao24, 'label': tr.label, 'type': tr.icao_type, 'source': tr.source,
                    't': np.round(tr.t[idx], 1).tolist(), 'lat': np.round(tr.lat[idx], 5).tolist(),
                    'lon': np.round(tr.lon[idx], 5).tolist(),
                    'alt_m': [None if not np.isfinite(x) else round(float(x)) for x in tr.alt_msl_m[idx]]})
    src = case.adsb_dir / 'source.json'
    return {'flights': out, 'source': json.loads(src.read_text()) if src.exists() else None}


def result(case):
    """result.json plus grids in lat/lon for maps; validation.json when present."""
    r = json.loads(case.result_path.read_text())
    enu = geo.ENU(*r['enu_origin'])

    def ll(E, N):
        lat, lon, _ = enu.to_geodetic(np.stack([E, N, np.zeros_like(E)], -1))
        return np.round(lat, 5).tolist(), np.round(lon, 5).tolist()
    out = {'result': r}
    gc = np.load(case.match_dir / 'grid_coarse.npz')
    tot = gc['cost'].sum(0).min(1)
    lat, lon = ll(gc['E'], gc['N'])
    out['coarse'] = {'lat': lat, 'lon': lon, 'total': np.round(tot, 2).tolist(), 'step_m': float(gc['step'])}
    gl = np.load(case.match_dir / 'grid_local.npz')
    keep = gl['p'] > gl['p'].max() * 1e-3
    lat, lon = ll(gl['E'][keep], gl['N'][keep])
    out['posterior'] = {'lat': lat, 'lon': lon, 'p': np.round(gl['p'][keep] / gl['p'].max(), 4).tolist(),
                        'step_m': float(gl['step'])}
    if case.validation_path.exists():
        out['validation'] = json.loads(case.validation_path.read_text())
    return out


def audio_clip(case, t0, t1, max_s=180.0, rate=11025):
    """Mono 16-bit WAV of [t0, t1] seconds (for listening to an event in the browser)."""
    import soundfile as sf
    from scipy.signal import resample_poly
    info = case.audio_info()
    t0 = max(0.0, float(t0)); t1 = min(info['duration_s'], float(t1), t0 + max_s)
    if t1 <= t0:
        raise ValueError('empty audio range')
    fs = info['samplerate']
    x, _ = sf.read(str(case.audio_path), start=int(t0 * fs), stop=int(t1 * fs), dtype='float32', always_2d=True)
    y = x.mean(1)
    if fs != rate:
        g = np.gcd(int(fs), rate)
        y = resample_poly(y, rate // g, int(fs) // g)
    y = y / max(1e-6, np.abs(y).max()) * 0.9
    buf = io.BytesIO()
    sf.write(buf, y, rate, format='WAV', subtype='PCM_16')
    return buf.getvalue()
