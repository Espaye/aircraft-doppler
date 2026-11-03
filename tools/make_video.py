"""Render a 28-second promo video (1920x1080, 30 fps, H.264 and AAC) of the synthetic example: a flyover you can
hear, its Doppler curve, then the pipeline finding the hidden microphone. On-screen numbers come from the case
files; the real-recording errors are the published ones from README.md (Status).

Needs matplotlib and imageio-ffmpeg on top of adoppler (pip install matplotlib imageio-ffmpeg). Builds the example
case like tools/build_site.py when it is missing (about two minutes), then renders the frames in parallel.

    python tools/make_video.py [--cases DIR] [--out build/video/adoppler-promo.mp4] [--stills 2,6,20] [--workers N]
"""
import argparse, io, json, os, subprocess, sys, tempfile, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adoppler import case, geo, physics, pipeline, synth, views  # noqa: E402

CASE_ID = 'synth-example'
W, H, FPS, SR, DUR = 1920, 1080, 30, 48000, 28.5
A1, B1, C1, D1 = 8.0, 12.6, 19.2, 24.0      # scene ends: flyover, recording, map search, result; then end card
PT = 72 / 120                                # points per pixel (figure is 16 x 9 in at 120 dpi)
BG, WHITE, SOFT, MUTED, AMBER, TEAL, NAVY = '#070d14', '#ffffff', '#c9d6e2', '#7d8fa3', '#f2b56b', '#7fd1c7', '#17324d'
SITE, REPO = 'aircraft-doppler.vercel.app', 'github.com/Espaye/aircraft-doppler'
REAL = (('day 1', '17 m', '35 m'), ('day 2', '114 m', '214 m'))  # README.md, Status: error, 68% radius

# A low, close pass rendered with synth's physics (not part of the example case): 70 m/s, closest approach 125 m
HERO = synth.Flight('HERO', 'f0f0f0', 'C172', 'CESSNA 172', 45.0, 80, 70.0, 110, 60, 95.0,
                    harmonics=(1, .75, .55, .4, .3, .22, .15, .1))
HERO_T, HERO_LEN = 3.9, 10.0                 # video time of the closest approach, clip length (s)
SPEC_Y0, SPEC_Y1, F_MAX = 230, 1040, 1000.0  # spectrogram band on screen (px) and its top frequency
MAP_CX, MAP_CY = 960, 640
SCAN0, SCAN1, Z0, Z1 = 15.3, 16.4, 16.5, 18.7   # grid scan, zoom into the answer
MARK, TRUTH, REAL_T = 18.4, 19.4, 21.6      # estimate appears, hidden truth revealed, real results

CAPTIONS = [  # (from, to, text, kind): 'h' headline, 's' subline
    (0.25, 3.6, 'A plane flies over a microphone.', 'h'),
    (0.8, 3.6, 'sound on', 's'),
    (3.75, A1, 'Hear the pitch drop? That\u2019s Doppler.', 'h'),
    (4.9, A1, 'Its shape gives away the speed ({v}) and how close it passed ({d}).', 's'),
    (A1 + 0.2, B1, 'adoppler finds every pass in a 20-minute recording\u2026', 'h'),
    (A1 + 0.5, B1, '{n_events} tonal events, {n_used} clean aircraft passes kept', 's'),
    (B1 + 0.25, 15.15, '\u2026lines them up with ADS-B flight tracks\u2026', 'h'),
    (B1 + 0.6, 15.15, 'the positions aircraft broadcast, kept in public archives', 's'),
    (15.2, TRUTH - 0.1, '\u2026and tests every spot the mic could be.', 'h'),
    (15.5, TRUTH - 0.1, '{cells} spots \u00d7 {deltas} clock offsets, all passes at once', 's'),
    (TRUTH, REAL_T - 0.1, 'Found it: {err} from the hidden mic.', 'h'),
    (TRUTH + 0.3, REAL_T - 0.1, 'synthetic test recording, 90% radius {r90}', 's'),
    (REAL_T, D1, 'Real livestream, analysed blind: {real} off.', 'h'),
    (REAL_T + 0.3, D1, 'each estimate stamped before the true position was looked up', 's'),
]

FONT_FILES = {'b': (['/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf', 'C:/Windows/Fonts/segoeuib.ttf'], 700),
              'm': (['/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf', 'C:/Windows/Fonts/seguisb.ttf'], 500),
              'r': (['/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf', 'C:/Windows/Fonts/segoeui.ttf'], 400),
              'mono': (['/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf', 'C:/Windows/Fonts/consola.ttf'], 400)}
_fonts = {}


def font_files(tmp):
    """First font found per kind. matplotlib draws only the default instance of a variable font (newer Ubuntu
    ships Ubuntu-B.ttf as a link to one), so cut a static instance at the wanted weight into tmp."""
    from fontTools.ttLib import TTFont
    out = {}
    for kind, (paths, wght) in FONT_FILES.items():
        path = next((p for p in paths if os.path.exists(p)), None)
        if path and 'fvar' in TTFont(path, lazy=True):
            from fontTools.varLib.instancer import instantiateVariableFont
            f = TTFont(path)
            axes = {a.axisTag: a.defaultValue for a in f['fvar'].axes}
            axes['wght'] = wght
            path = str(Path(tmp) / f'{kind}.ttf')
            instantiateVariableFont(f, axes).save(path)
        out[kind] = path
    return out


def font(kind, px):
    from matplotlib.font_manager import FontProperties
    if (kind, px) not in _fonts:
        path = _D['fonts'][kind]
        fp = FontProperties(fname=path) if path else FontProperties(weight='bold' if kind == 'b' else 'normal')
        fp.set_size(px * PT)
        _fonts[kind, px] = fp
    return _fonts[kind, px]


def ramp(t, a, b):
    return float(np.clip((t - a) / (b - a), 0.0, 1.0))


def ease(u):
    return u * u * (3 - 2 * u)


def shown(t, a, b, fi=0.35, fo=0.3):
    """Opacity of something on screen from a to b, with fades."""
    return ease(ramp(t, a, a + fi)) * (1 - ease(ramp(t, b - fo, b)))


# ---------------------------------------------------------------------------------------------------- data


def hero():
    """The flyover: stereo audio with the closest approach heard at HERO_T, its spectrogram and model curves."""
    rng = np.random.default_rng(7)
    mic = np.array([0.0, 0.0, 2.0]); image = mic * np.array([1, 1, -1])
    tau_g, P = synth._path(HERO, mic, 0.0, 90.0)
    c = float(physics.sound_speed(15.0)); still = np.zeros(3)
    t_rec = np.arange(int(15 * SR), int(75 * SR)) / SR          # synth fades sources in over 20 s: stay inside
    tone = synth._render(HERO, tau_g, P, mic, image, t_rec, 0.0, c, still, SR, rng)
    tc = np.arange(20.0, 70.0, 0.005)
    tau, R = synth._emission_time(tau_g, P, mic[None, :], tc, c, still)
    shift = tc[np.argmin(R)] - HERO_T                            # video time = reception time - shift
    a, n = int(round((shift - t_rec[0]) * SR)), int(HERO_LEN * SR)
    tone, tv = tone[a:a + n], tc - shift
    ts = np.arange(n) / SR
    Rn = np.interp(ts, tv, R)
    f = np.fft.rfftfreq(n, 1 / SR)
    X = np.fft.rfft(rng.normal(0, 1, n)) * ((f > 150) & (f < 4000)) / np.sqrt(np.maximum(f, 150) / 150)
    hiss = np.fft.irfft(X, n); hiss *= 0.35 * 4.0 / np.maximum(Rn, 5.0) / hiss.std()   # propeller noise
    bg = synth._background(n, SR, rng, HERO_LEN)                 # wind, hum and birds of the synthetic scenes
    Ex = np.interp(tau, tau_g, P[:, 0])
    th = (np.interp(ts, tv, np.clip(Ex / R, -1, 1)) * 0.8 + 1) * np.pi / 4   # pan with the aircraft, west to east
    plane = tone + hiss
    stereo = np.stack([plane * np.cos(th), plane * np.sin(th)], 1) * np.sqrt(2) + bg[:, None]
    stereo *= np.clip(ts / 0.3, 0, 1)[:, None]
    stereo *= 0.9 / np.abs(stereo).max()
    V = np.gradient(P, tau_g, axis=0)
    p = np.stack([np.interp(tau, tau_g, P[:, k]) for k in range(3)], 1)
    v = np.stack([np.interp(tau, tau_g, V[:, k]) for k in range(3)], 1)
    vr = ((p - mic) * v).sum(1) / R
    keep = (tv >= 0) & (tv <= HERO_LEN)
    curves = [(tv[keep], h * HERO.bpf_hz / (1 + vr[keep] / c)) for h in (1, 2, 3, 4)]
    img, t_cols, f_top = hero_spectrogram(tone + hiss + bg)
    return stereo, {'img': img, 't': t_cols, 'f_top': f_top, 'curves': curves}


def hero_spectrogram(x, nper=8192, zpad=4, cols_per_s=150):
    """Tonal-enhanced spectrogram (as in adoppler.spectro) mixed with a little plain power, magma colours."""
    from scipy.ndimage import median_filter
    hop, nfft = SR // cols_per_s, nper * zpad
    nb = int(F_MAX / (SR / nfft)) + 1
    xp = np.concatenate([np.zeros(nper // 2), x, np.zeros(nper // 2)])
    nfr = (len(xp) - nper) // hop + 1
    win = np.hanning(nper)
    L = np.empty((nfr, nb), np.float32)
    for a in range(0, nfr, 128):
        idx = np.arange(a, min(nfr, a + 128))[:, None] * hop + np.arange(nper)
        L[a:a + len(idx)] = 10 * np.log10(np.abs(np.fft.rfft(xp[idx] * win, n=nfft, axis=1)[:, :nb]) ** 2 + 1e-14)
    Z = L - np.median(L, axis=0)
    T = Z - median_filter(Z, size=(1, int(40 / (SR / nfft)) | 1), mode='nearest')
    lo, hi = np.percentile(L, [30, 99.8])
    val = 0.8 * np.clip((T - 4) / 14, 0, 1) + 0.25 * np.clip((L - lo) / (hi - lo), 0, 1)
    return views._colour(val.T[::-1]), np.arange(nfr) * hop / SR, (nb - 1) * SR / nfft


def _grid(E, N, step):
    i = np.round((E - E.min()) / step).astype(int); j = np.round((N - N.min()) / step).astype(int)
    return i, j.max() - j, (j.max() + 1, i.max() + 1), (E.min() - step / 2, E.max() + step / 2,
                                                        N.min() - step / 2, N.max() + step / 2)


def _overlay(rgb, alpha):
    return np.concatenate([rgb / 255.0, alpha[..., None]], -1)


def prepare(c):
    """Everything the frames need, as plain arrays (sent to each worker once), plus the soundtrack."""
    from PIL import Image
    r = json.loads(c.result_path.read_text())
    val = json.loads(c.validation_path.read_text())
    png, ext = views.spectrogram(c, width=W, height=SPEC_Y1 - SPEC_Y0, f1=F_MAX, kind='tonal')
    ev = views.events(c)
    events = sorted(({'used': e['used'], 't': e['t_start'],
                      'tracks': [(np.array(tr['t']), np.array(tr['f'])) for tr in e['tracks']]}
                     for e in ev['events']), key=lambda e: e['t'])
    for k, e in enumerate(events):
        e['show'] = A1 + 0.6 + k * 2.6 / len(events)
    enu = geo.ENU(*r['enu_origin'])
    flights = []
    for f in views.flights(c)['flights']:
        p = enu.from_geodetic(f['lat'], f['lon'], np.zeros(len(f['lat'])))
        flights.append({'E': p[:, 0], 'N': p[:, 1], 't': np.array(f['t']) - ev['t0_utc'], 'type': f['type']})
    gc = np.load(c.match_dir / 'grid_coarse.npz')
    tot = gc['cost'].sum(0).min(1)
    q = np.clip((np.median(tot) - tot) / (np.median(tot) - tot.min()), 0, 1) ** 0.5
    i, j, shape, box = _grid(gc['E'], gc['N'], float(gc['step']))
    A = np.zeros(shape + (4,)); A[j, i] = _overlay(views._colour(0.2 + 0.8 * q), 0.16 + 0.8 * q)
    rows_N = box[3] - (np.arange(shape[0]) + 0.5) * float(gc['step'])
    gl = np.load(c.match_dir / 'grid_local.npz')
    pl = (gl['p'] / gl['p'].max()) ** 0.35
    i, j, shape, lbox = _grid(gl['E'], gl['N'], float(gl['step']))
    L = np.zeros(shape + (4,)); L[j, i] = _overlay(views._colour(0.3 + 0.7 * pl), 0.95 * pl)
    stereo, hd = hero()
    yy, xx = np.mgrid[-1:1:108j, -1:1:192j]
    end_bg = np.array([[0x17, 0x32, 0x4d], [0x1f, 0x4a, 0x6b], [0x27, 0x5c, 0x73]], float)
    u = np.clip((xx * 0.35 + yy * 0.94 + 1.3) / 2.6, 0, 1)[..., None]            # ~160 degrees, like the site
    grad = np.where(u < 0.6, end_bg[0] + (end_bg[1] - end_bg[0]) * u / 0.6,
                    end_bg[1] + (end_bg[2] - end_bg[1]) * (u - 0.6) / 0.4).astype(np.uint8)
    n_used = sum(e['used'] for e in events)
    facts = {'v': f'{HERO.speed_mps:.0f} m/s', 'd': f'{np.hypot(HERO.offset_m, HERO.alt_agl_m):.0f} m',
             'n_events': ev['counts']['events'], 'n_used': n_used, 'cells': f'{len(tot):,}',
             'deltas': gc['cost'].shape[2], 'err': f"{val['error_m']:.0f} m",
             'r90': f"{r['uncertainty']['bootstrap_90_m']:.0f} m", 'real': ' and '.join(e for _, e, _ in REAL)}
    D = {'hero': hd, 'rec': np.asarray(Image.open(io.BytesIO(png)).convert('RGB')), 'rec_ext': ext,
         'events': events, 'flights': flights, 'coarse': A, 'coarse_box': box, 'coarse_rows': rows_N,
         'local': L, 'local_box': lbox, 'best': (r['best']['E_m'], r['best']['N_m']),
         'truth': (val['ref_E_m'], val['ref_N_m']), 'r90': r['uncertainty']['bootstrap_90_m'],
         'radius_m': r['settings']['search_radius_m'], 'facts': facts, 'end_bg': grad,
         'vignette': np.concatenate([np.zeros(yy.shape + (3,)),
                                     (np.clip(np.hypot(xx, yy) - 0.55, 0, None) ** 2 * 0.9)[..., None]], -1)}
    return D, soundtrack(stereo, events)


def soundtrack(hero_st, events):
    """Flyover, then a quiet synthesised pad, ticks for detected passes, a riser and chimes at the answer."""
    n = int(DUR * SR); t = np.arange(n) / SR
    y = np.zeros((n, 2))
    m = len(hero_st)
    y[:m] += hero_st * np.clip((A1 + 2.0 - t[:m]) / 2.2, 0, 1)[:, None]
    rng = np.random.default_rng(3)

    def span(a, b):
        return slice(int(a * SR), min(n, int(b * SR)))

    def pad(freqs, a, b, gain):
        s = span(a, b); tt = t[s]
        env = np.clip((tt - a) / 1.6, 0, 1) * np.clip((b - tt) / 1.4, 0, 1) * gain
        for ch, det in ((0, (-0.0016, 0.0004)), (1, (-0.0004, 0.0016))):
            sig = sum(np.sin(2 * np.pi * f * (1 + d) * tt + f) + 0.2 * np.sin(4 * np.pi * f * (1 + d) * tt)
                      for f in freqs for d in det)
            y[s, ch] += sig * env * (1 + 0.15 * np.sin(2 * np.pi * 0.25 * tt + ch))

    def ping(at, f0, gain):
        s = span(at, at + 2.5); tt = t[s] - at
        sig = sum(w * np.sin(2 * np.pi * f0 * k * tt) * np.exp(-tt / d)
                  for k, w, d in ((1, 1, 1.1), (2.0, .35, .6), (2.76, .25, .4), (5.4, .08, .2)))
        y[s] += (sig * gain * np.clip(tt / 0.004, 0, 1))[:, None]

    def tick(at, f0, gain):
        s = span(at, at + 0.08); tt = t[s] - at
        y[s] += (np.sin(2 * np.pi * f0 * tt) * np.exp(-tt / 0.018) * gain)[:, None]

    def noise(a, b, gain, shape):
        s = span(a, b); k = s.stop - s.start
        f = np.fft.rfftfreq(k, 1 / SR)
        x = np.fft.irfft(np.fft.rfft(rng.normal(0, 1, k)) * ((f > 400) & (f < 9000)), k)
        y[s] += (x / x.std() * gain * shape((t[s] - a) / (b - a)))[:, None]

    pad([73.42, 110.0, 146.83, 164.81, 220.0], A1 - 1.0, TRUTH + 0.3, 0.012)
    pad([73.42, 110.0, 185.0, 220.0, 277.18, 329.63], TRUTH - 0.1, DUR, 0.011)
    for e in events:
        tick(e['show'], 1760.0 if e['used'] else 1046.5, 0.07 if e['used'] else 0.035)
    noise(SCAN0, SCAN1 + 0.2, 0.012, lambda u: np.sin(np.pi * np.clip(u, 0, 1)))
    noise(Z0 - 0.3, TRUTH, 0.05, lambda u: u ** 3)
    s = span(Z0 - 0.3, TRUTH); tt = t[s] - (Z0 - 0.3)
    y[s] += (0.03 * np.sin(2 * np.pi * 220 * (np.exp(tt * 0.5) - 1) / 0.5) * (tt / tt[-1]) ** 2)[:, None]
    ping(TRUTH, 1318.5, 0.16)
    s = span(TRUTH, TRUTH + 1.0); tt = t[s] - TRUTH
    y[s] += (0.35 * np.sin(2 * np.pi * 52 * tt) * np.exp(-tt / 0.3) * np.clip(tt / 0.01, 0, 1))[:, None]
    ping(REAL_T, 987.77, 0.11)
    ping(D1 + 0.2, 1174.66, 0.10); ping(D1 + 0.25, 1760.0, 0.05)
    y *= np.clip((DUR - t) / 1.8, 0, 1)[:, None]
    return (y * 0.95 / np.abs(y).max()).astype(np.float32)


# ---------------------------------------------------------------------------------------------------- frames

_D = _FIG = _CANVAS = None


def _init(d):
    global _D, _FIG, _CANVAS
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    _D = d
    _FIG = Figure(figsize=(16, 9), dpi=120)
    _CANVAS = FigureCanvasAgg(_FIG)


def glow(ax, x, y, color, lw, alpha, **kw):
    for w, a in ((lw * 5, 0.07), (lw * 2.4, 0.18), (lw, 1.0)):
        ax.plot(x, y, color=color, lw=w * PT, alpha=alpha * a, solid_capstyle='round', **kw)


def text(ax, x, y, s, kind, px, color=WHITE, alpha=1.0, ha='center', **kw):
    if alpha > 0.002:
        ax.text(x, y, s, fontproperties=font(kind, px), color=color, alpha=alpha, ha=ha, va='center', **kw)


def freq_axis(ax, f_top, k):
    for f in (250, 500, 750):
        y = SPEC_Y1 - f / f_top * (SPEC_Y1 - SPEC_Y0)
        ax.plot([0, W], [y, y], color=WHITE, lw=1 * PT, alpha=0.07 * k)
        text(ax, 22, y - 14, f'{f} Hz', 'r', 18, MUTED, 0.9 * k, ha='left')


def draw_flyover(ax, t, k):
    d = _D['hero']
    x_of = lambda s: np.asarray(s) / A1 * W                      # noqa: E731
    ax.imshow(d['img'], extent=(x_of(d['t'][0]), x_of(d['t'][-1]), SPEC_Y1, SPEC_Y0), aspect='auto',
              interpolation='bilinear', alpha=k)
    xp = float(x_of(min(t, A1 + 1)))
    if xp < W:
        ax.add_patch(_rect(xp, SPEC_Y0 - 2, W - xp + 4, SPEC_Y1 - SPEC_Y0 + 4, BG, k))
        glow(ax, [xp, xp], [SPEC_Y0, SPEC_Y1], WHITE, 2, 0.8 * k)
    freq_axis(ax, d['f_top'], k)
    yf = lambda f: SPEC_Y1 - np.asarray(f) / d['f_top'] * (SPEC_Y1 - SPEC_Y0)   # noqa: E731
    text(ax, x_of(1.4), SPEC_Y0 + 36, 'coming closer: higher pitch', 'm', 24, SOFT, shown(t, 2.2, A1 + 0.6) * k)
    text(ax, x_of(6.1), SPEC_Y0 + 36, 'moving away: lower pitch', 'm', 24, SOFT, shown(t, 5.4, A1 + 0.6) * k)
    u = ease(ramp(t, 5.0, 6.1))
    if u > 0:
        for tt, f in d['curves']:
            m = tt <= min(u * A1, t)
            ax.plot(x_of(tt[m]), yf(f[m]), color=WHITE, lw=2.2 * PT, alpha=0.9 * k, ls=(0, (3, 2.5)))
        tt, f = d['curves'][3]
        text(ax, x_of(0.5), float(yf(f[0])) - 26, 'dashed: Doppler model', 'm', 22, WHITE,
             shown(t, 5.2, A1 + 0.6) * k, ha='left')
        a = shown(t, 5.6, A1 + 0.6) * k
        xc = float(x_of(HERO_T))
        ax.plot([xc, xc], [SPEC_Y0 + 70, SPEC_Y1 - 50], color=AMBER, lw=1.6 * PT, alpha=0.8 * a, ls=(0, (4, 4)))
        text(ax, xc, SPEC_Y1 - 28, f"closest approach: {_D['facts']['d']}", 'm', 22, AMBER, a)


def _rect(x, y, w, h, color, alpha):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, facecolor=color, edgecolor='none', alpha=alpha)


def draw_recording(ax, t, k):
    ext = _D['rec_ext']
    x_of = lambda s: (np.asarray(s) - ext['t0']) / (ext['t1'] - ext['t0']) * W        # noqa: E731
    y_of = lambda f: SPEC_Y1 - (np.asarray(f) - ext['f0']) / (ext['f1'] - ext['f0']) * (SPEC_Y1 - SPEC_Y0)  # noqa
    ax.imshow(_D['rec'], extent=(0, W, SPEC_Y1, SPEC_Y0), aspect='auto', interpolation='bilinear', alpha=0.9 * k)
    freq_axis(ax, ext['f1'], k)
    for mnt in range(5, 20, 5):
        x = float(x_of(mnt * 60))
        ax.plot([x, x], [SPEC_Y1 - 12, SPEC_Y1], color=SOFT, lw=1.5 * PT, alpha=0.8 * k)
        text(ax, x, SPEC_Y1 - 30, f'{mnt} min', 'r', 18, SOFT, 0.9 * k)
    for e in _D['events']:
        a = ease(ramp(t, e['show'], e['show'] + 0.3)) * k
        if a <= 0:
            continue
        col = AMBER if e['used'] else '#8796a8'
        for tt, f in e['tracks']:
            glow(ax, x_of(tt), y_of(f), col, 2.2 if e['used'] else 1.4, a * (1 if e['used'] else 0.7))


def view(t):
    """Map camera: (east, north) at the map centre and pixels per metre; zooms into the estimate."""
    s0, s1 = 400 / _D['radius_m'], 7.0
    u = ease(ramp(t, Z0, Z1)); s = s0 * (s1 / s0) ** u
    bE, bN = _D['best']
    x0, y0 = MAP_CX + bE * s0, MAP_CY - bN * s0
    w = ease(ramp(t, Z0, Z0 + 0.65 * (Z1 - Z0)))
    px, py = x0 + (MAP_CX - x0) * w, y0 + (MAP_CY - y0) * w
    return bE - (px - MAP_CX) / s, bN + (py - MAP_CY) / s, s


def draw_map(ax, t, k):
    from matplotlib.patches import Circle
    cE, cN, s = view(t)
    X = lambda E: MAP_CX + (np.asarray(E) - cE) * s             # noqa: E731
    Y = lambda N: MAP_CY - (np.asarray(N) - cN) * s             # noqa: E731
    ax.add_patch(_rect(0, 0, W, H, '#0a1622', k))
    Elo, Ehi = cE - MAP_CX / s, cE + (W - MAP_CX) / s
    Nlo, Nhi = cN - (H - MAP_CY) / s, cN + MAP_CY / s
    for L in (5000, 1000, 250, 50, 10):
        a = 0.13 * ramp(L * s, 30, 110) * (1 - ramp(L * s, 1500, 3000)) * k
        if a > 0.004:
            ax.vlines(np.arange(np.ceil(Elo / L), Ehi / L + 1) * L * s + MAP_CX - cE * s, 0, H, color=TEAL,
                      lw=1 * PT, alpha=a)
            ax.hlines(MAP_CY + cN * s - np.arange(np.ceil(Nlo / L), Nhi / L + 1) * L * s, 0, W, color=TEAL,
                      lw=1 * PT, alpha=a)
    r = _D['radius_m'] * s
    if r < 8000:
        ax.add_patch(Circle((X(0), Y(0)), r, fill=True, fc=TEAL, ec='none', alpha=0.03 * k))
        ax.add_patch(Circle((X(0), Y(0)), r, fill=False, ec=TEAL, lw=1.6 * PT, ls=(0, (5, 5)), alpha=0.55 * k))
        text(ax, X(0) + r * 0.72 + 24, Y(0) + r * 0.72 + 10,
             f"search area: {_D['radius_m'] / 1000:.0f} km around a rough guess", 'm', 22, TEAL,
             shown(t, B1 + 0.5, Z0 + 0.2) * k, ha='left')
    u = ramp(t, B1 + 0.4, B1 + 3.2); tf = -600 + u * 2400
    lab = (1 - ramp(t, B1 + 3.2, B1 + 3.7)) * k
    for f in _D['flights']:
        m = f['t'] <= tf
        if m.sum() < 2:
            continue
        x, y = X(f['E'][m]), Y(f['N'][m])
        glow(ax, x, y, TEAL, 1.5, 0.6 * k)
        if lab > 0:
            ax.scatter([x[-1]], [y[-1]], s=(9 * PT) ** 2, color=WHITE, alpha=lab, zorder=5, lw=0)
            text(ax, x[-1] + 12, y[-1] - 16, f['type'], 'mono', 20, SOFT, lab, ha='left')
    kc = (1 - ramp(s, 0.12, 0.5)) * k
    if t > SCAN0 and kc > 0:
        A = _D['coarse'].copy()
        E0, E1, N0, N1 = _D['coarse_box']
        Ns = N1 - ramp(t, SCAN0, SCAN1) * (N1 - N0)
        A[_D['coarse_rows'] < Ns, :, 3] = 0
        A[..., 3] *= kc
        ax.imshow(A, extent=(X(E0), X(E1), Y(N0), Y(N1)), interpolation='nearest')
        if t < SCAN1:
            half = np.sqrt(max(_D['radius_m'] ** 2 - Ns ** 2, 0))
            glow(ax, [X(-half), X(half)], [Y(Ns), Y(Ns)], AMBER, 2, kc)
    kp = ramp(s, 0.15, 0.6) * (1 - 0.6 * ease(ramp(t, MARK, MARK + 0.6))) * k   # dim it under the marks
    if kp > 0:
        Lc = _D['local'].copy(); Lc[..., 3] *= kp
        E0, E1, N0, N1 = _D['local_box']
        ax.imshow(Lc, extent=(X(E0), X(E1), Y(N0), Y(N1)), interpolation='bilinear')
    (bE, bN), (tE, tN) = _D['best'], _D['truth']
    a = ease(ramp(t, MARK, MARK + 0.4)) * k
    if a > 0:
        bx, by = float(X(bE)), float(Y(bN))
        ax.add_patch(Circle((bx, by), _D['r90'] * s, fill=False, ec=AMBER, lw=2 * PT, ls=(0, (4, 3)), alpha=0.9 * a))
        text(ax, bx + _D['r90'] * s * 0.72 + 30, by + _D['r90'] * s * 0.72 + 18, '90%', 'm', 22, AMBER, a)
        g = 9 + 20 * (1 - a)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            seg = [bx + dx * g, bx + dx * (g + 22)], [by + dy * g, by + dy * (g + 22)]
            ax.plot(*seg, color=BG, lw=8 * PT, alpha=0.6 * a, solid_capstyle='round')
            glow(ax, *seg, AMBER, 3, a)
        text(ax, bx - 40, by + 52, 'estimate', 'm', 24, AMBER, a, ha='right')
    a = ease(ramp(t, TRUTH, TRUTH + 0.35)) * k
    if a > 0:
        tx, ty = float(X(tE)), float(Y(tN))
        ax.add_patch(Circle((tx, ty), 15 + 30 * (1 - a), fill=False, ec=WHITE, lw=3 * PT, alpha=a))
        ax.plot([float(X(bE)), tx], [float(Y(bN)), ty], color=WHITE, lw=1.4 * PT, alpha=0.7 * a)
        xl = float(X(bE)) + _D['r90'] * s + 40
        ax.plot([tx + 20, xl - 12], [ty, ty], color=WHITE, lw=1.2 * PT, alpha=0.5 * a)
        text(ax, xl, ty, 'hidden mic', 'm', 24, WHITE, a, ha='left')
    for L in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)[::-1]:
        if L * s <= 260:
            break
    x1, y1 = W - 90, H - 80
    ax.plot([x1 - L * s, x1], [y1, y1], color=SOFT, lw=2 * PT, alpha=0.8 * k)
    text(ax, x1 - L * s / 2, y1 - 20, f'{L // 1000} km' if L >= 1000 else f'{L} m', 'r', 20, SOFT, 0.9 * k)


def draw_real(ax, t, k):
    """The published real-recording errors as tiles over the dimmed synthetic map (no map: location is private)."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(_rect(0, 0, W, H, BG, 0.88 * ease(ramp(t, REAL_T - 0.1, REAL_T + 0.4)) * k))
    for n, (day, err, r68) in enumerate(REAL):
        a = ease(ramp(t, REAL_T + 0.2 + 0.25 * n, REAL_T + 0.6 + 0.25 * n)) * k
        if a <= 0:
            continue
        x0, y0 = W / 2 - 470 + n * 500 + 20 * (1 - a), 390
        ax.add_patch(FancyBboxPatch((x0, y0), 440, 300, boxstyle='round,pad=0,rounding_size=24', fc='#10263b',
                                    ec=AMBER, lw=1.5 * PT, alpha=0.95 * a))
        text(ax, x0 + 220, y0 + 50, day, 'm', 28, SOFT, a)
        text(ax, x0 + 220, y0 + 145, err, 'b', 110, AMBER, a)
        text(ax, x0 + 220, y0 + 245, f'error  ·  68% radius {r68}', 'r', 24, SOFT, a)
    text(ax, W / 2, 780, 'one site, two recordings  ·  the location stays private', 'r', 24, MUTED,
         ease(ramp(t, REAL_T + 0.8, REAL_T + 1.2)) * k)


def logo(ax, cx, cy, size, alpha, fc=NAVY):
    from matplotlib.patches import FancyBboxPatch, PathPatch
    from matplotlib.path import Path as MPath
    sc = size / 32; x0, y0 = cx - size / 2, cy - size / 2
    ax.add_patch(FancyBboxPatch((x0, y0), size, size, boxstyle=f'round,pad=0,rounding_size={7 * sc}', fc=fc,
                                ec='none', alpha=alpha))
    for pts, col, lw, op in (((4, 10), (10, 10), (12, 12), (16, 16), (20, 20), (22, 22), (28, 22)), TEAL, 2.6, 1), \
                            (((4, 17), (10, 17), (12, 18.5), (16, 21.5), (20, 24.5), (22, 26), (28, 26)), AMBER, 2, .85):
        path = MPath([(x0 + px * sc, y0 + py * sc) for px, py in pts], [MPath.MOVETO] + [MPath.CURVE4] * 6)
        ax.add_patch(PathPatch(path, fill=False, ec=col, lw=lw * sc * PT, capstyle='round', alpha=alpha * op))


def draw_end(ax, t, k):
    from matplotlib.patches import FancyBboxPatch
    ax.imshow(_D['end_bg'], extent=(0, W, H, 0), aspect='auto', interpolation='bilinear', alpha=k)
    a = ease(ramp(t, D1 + 0.3, D1 + 0.8))
    logo(ax, W / 2, 300 + 20 * (1 - a), 150, a, fc='#10263b')
    text(ax, W / 2, 452, 'adoppler', 'b', 92, WHITE, a)
    text(ax, W / 2, 575, 'Where was this recorded?', 'b', 58, WHITE, shown(t, D1 + 0.5, DUR + 1))
    text(ax, W / 2, 650, 'Ask the aircraft overhead.', 'b', 58, AMBER, shown(t, D1 + 0.9, DUR + 1))
    a = shown(t, D1 + 1.4, DUR + 1)
    if a > 0:
        ax.add_patch(FancyBboxPatch((W / 2 - 430, 760), 860, 80, boxstyle='round,pad=0,rounding_size=40', fc=AMBER,
                                    ec='none', alpha=a))
        text(ax, W / 2, 800, f'Try the demo in your browser: {SITE}', 'b', 32, NAVY, a)
    text(ax, W / 2, 900, f'Open source (MIT)  \u00b7  {REPO}', 'm', 30, SOFT, shown(t, D1 + 1.8, DUR + 1))


def frame(i):
    t = i / FPS
    fig = _FIG; fig.clf()
    ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_autoscale_on(False)
    ax.axis('off'); fig.patch.set_facecolor(BG)
    kA = 1 - ease(ramp(t, A1 - 0.1, A1 + 0.7))
    kB = ease(ramp(t, A1 - 0.1, A1 + 0.7)) * (1 - ease(ramp(t, B1 - 0.1, B1 + 0.6)))
    kC = ease(ramp(t, B1 - 0.1, B1 + 0.6)) * (1 - ease(ramp(t, D1 - 0.2, D1 + 0.3)))
    kE = ease(ramp(t, D1 - 0.2, D1 + 0.3))
    if kC > 0:
        draw_map(ax, t, kC)
        if t > REAL_T - 0.1:
            draw_real(ax, t, kC)
    if kB > 0:
        draw_recording(ax, t, kB)
    if kA > 0:
        draw_flyover(ax, t, kA)
    ax.imshow(_D['vignette'], extent=(0, W, H, 0), aspect='auto', interpolation='bilinear')
    if kE > 0:
        draw_end(ax, t, kE)
    top = np.zeros((64, 1, 4)); top[..., 3] = np.linspace(0.92, 0, 64)[:, None] * (1 - kE)
    ax.imshow(top, extent=(0, W, 300, 0), aspect='auto', interpolation='bilinear')
    f = _D['facts']
    for a, b, s, kind in CAPTIONS:
        al = shown(t, a, b)
        if al > 0:
            rise = (1 - ease(ramp(t, a, a + 0.5))) * 14
            if kind == 'h':
                text(ax, W / 2, 105 + rise, s.format(**f), 'b', 60, WHITE, al)
            else:
                text(ax, W / 2, 172 + rise, s.format(**f), 'm', 30, SOFT, al)
    corner = 1 - kE
    if corner > 0:
        tags = ((0, A1, "synthetic flyover, rendered with adoppler's physics"),
                (A1, REAL_T, f'synthetic demo case: {SITE}/demo'), (REAL_T, DUR, 'real-recording results: see the README'))
        for a, b, tag in tags:
            text(ax, 36, H - 30, tag, 'r', 20, MUTED, corner * shown(t, a, b, 0.4, 0.4), ha='left')
        logo(ax, W - 196, H - 34, 34, 0.9 * corner)
        text(ax, W - 170, H - 34, 'adoppler', 'b', 28, WHITE, 0.9 * corner, ha='left')
    for z, artist in enumerate(ax.get_children()):   # images default to below patches and lines: keep draw order
        if artist is not ax.patch:
            artist.set_zorder(z)
    _CANVAS.draw()
    return np.asarray(_CANVAS.buffer_rgba())[:, :, :3].tobytes()


# ---------------------------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--cases', default=str(ROOT / 'cases'), help='where to keep the example case (default: cases/)')
    ap.add_argument('--out', default=str(ROOT / 'build' / 'video' / 'adoppler-promo.mp4'),
                    help='output video (default: build/video/adoppler-promo.mp4, git-ignored)')
    ap.add_argument('--stills', help='comma-separated times in seconds: write PNG frames instead of the video')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    t0 = time.time()
    d = Path(a.cases) / CASE_ID
    if not (d / 'case.json').exists():
        synth.make_case(d, 'example', seed=1)
    c = case.Case(d)
    if not c.result_path.exists():
        pipeline.run_all(c)
    if not c.validation_path.exists():
        pipeline.validate_synthetic(c)
    D, audio = prepare(c)
    tmp = tempfile.TemporaryDirectory()
    D['fonts'] = font_files(tmp.name)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"prepared in {time.time() - t0:.0f} s: {D['facts']}")
    if a.stills:
        from PIL import Image
        _init(D)
        for s in (float(x) for x in a.stills.split(',')):
            p = out.with_name(f'{out.stem}-{s:05.2f}s.png')
            Image.frombytes('RGB', (W, H), frame(int(round(s * FPS)))).save(p)
            print(p)
        return
    import imageio_ffmpeg
    import soundfile as sf
    wav = out.with_suffix('.wav')
    sf.write(str(wav), audio, SR, subtype='PCM_16')
    n = int(DUR * FPS)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-i', str(wav), '-c:v', 'libx264', '-preset', 'slow',
           '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart',
           '-shortest', str(out)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(a.workers, _init, (D,)) as pool:
        for k, buf in enumerate(pool.imap(frame, range(n), chunksize=4)):
            ff.stdin.write(buf)
            if k % 150 == 0:
                print(f'frame {k}/{n}', flush=True)
    ff.stdin.close()
    if ff.wait():
        raise SystemExit('ffmpeg failed')
    wav.unlink()
    print(f'{out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
