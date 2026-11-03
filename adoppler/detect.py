"""Automatic detection of aircraft-like Doppler tracks in a tonal-enhanced spectrogram.

Pipeline: spectral peaks per frame -> frame-to-frame tracking -> gap merging -> harmonic grouping into events
-> constant-velocity Doppler fit per event -> classification and confidence ranking.
"""
import json
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import least_squares

C_FIT = 340.0  # sound speed used only for the per-event descriptive fit (matching uses its own model)


# ---------------------------------------------------------------- peaks

def find_peaks(T, Z, L_db, df, fmin=25.0, fmax=1800.0, thr_db=6.0, z_min=2.0, max_per_frame=40, chunk=2000):
    """Local maxima along frequency of T above thr_db (and Z above z_min). Returns array (frame, freq_hz, T_db)."""
    out = []
    j0, j1 = int(fmin / df), int(fmax / df)
    for a in range(0, T.shape[0], chunk):
        b = min(T.shape[0], a + chunk)
        Tc = np.asarray(T[a:b, j0 - 1:j1 + 1]); Zc = np.asarray(Z[a:b, j0 - 1:j1 + 1]); Lc = L_db(a, b, j0 - 1, j1 + 1)
        c = Tc[:, 1:-1]
        m = (c > Tc[:, :-2]) & (c >= Tc[:, 2:]) & (c > thr_db) & (Zc[:, 1:-1] > z_min)
        fi, bi = np.nonzero(m)
        if not len(fi):
            continue
        v = c[fi, bi]
        # parabolic interpolation on log power for sub-bin frequency
        y0, y1, y2 = Lc[fi, bi], Lc[fi, bi + 1], Lc[fi, bi + 2]
        den = y0 - 2 * y1 + y2
        off = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0).clip(-0.5, 0.5)
        freq = (j0 + bi + off) * df
        frames = fi + a
        # keep strongest per frame
        o = np.lexsort((-v, frames))
        frames, freq, v = frames[o], freq[o], v[o]
        rank = np.arange(len(frames)) - np.searchsorted(frames, frames, side='left')
        k = rank < max_per_frame
        out.append(np.stack([frames[k], freq[k], v[k]], 1))
    return np.concatenate(out) if out else np.zeros((0, 3))


# ---------------------------------------------------------------- tracking

def _line(fr, fq):
    """Least-squares slope/intercept of a short series (fast closed form)."""
    fr = np.asarray(fr, float); fq = np.asarray(fq, float)
    if len(fr) < 4 or fr[-1] == fr[0]:
        return 0.0, float(fq[-1]), float(fr[-1])
    x = fr - fr.mean()
    sl = float((x * (fq - fq.mean())).sum() / (x * x).sum())
    return sl, float(fq.mean() - sl * fr.mean()), None


def track(peaks, hop_s, df, max_gap_s=2.0, min_len_s=5.0, min_points=16, tol_bins=1.2):
    """Greedy frame-to-frame association with linear frequency prediction (vectorised over active tracks)."""
    max_gap = int(round(max_gap_s / hop_s))
    if not len(peaks):
        return []
    frames = peaks[:, 0].astype(int)
    starts = np.searchsorted(frames, np.arange(frames.max() + 2))
    tracks = []                     # all tracks: dict(f, q, v lists)
    act = np.zeros(0, int)          # indices of active tracks
    a_last = np.zeros(0); a_sl = np.zeros(0); a_ic = np.zeros(0); a_lq = np.zeros(0)
    for k in range(int(frames.max()) + 1):
        if len(act):
            alive = k - a_last <= max_gap
            act, a_last, a_sl, a_ic, a_lq = act[alive], a_last[alive], a_sl[alive], a_ic[alive], a_lq[alive]
        P = peaks[starts[k]:starts[k + 1]]
        if len(P) == 0:
            continue
        used = np.zeros(len(P), bool)
        if len(act):
            pred = np.where(a_sl != 0, a_sl * k + a_ic, a_lq)
            tol = tol_bins * df + 0.0015 * pred + 0.5 * np.abs(a_sl) * (k - a_last)
            D = np.abs(P[:, 1][None, :] - pred[:, None]) / tol[:, None]
            pairs = np.argwhere(D < 1)
            if len(pairs):
                order = np.argsort(D[pairs[:, 0], pairs[:, 1]])
                ta = np.zeros(len(act), bool)
                for ti, pi in pairs[order]:
                    if ta[ti] or used[pi]:
                        continue
                    ta[ti] = used[pi] = True
                    tr = tracks[act[ti]]
                    tr['f'].append(k); tr['q'].append(P[pi, 1]); tr['v'].append(P[pi, 2])
                    sl, ic, _ = _line(tr['f'][-12:], tr['q'][-12:])
                    a_last[ti], a_sl[ti], a_ic[ti], a_lq[ti] = k, sl, ic, P[pi, 1]
        new = np.nonzero(~used)[0]
        if len(new):
            base = len(tracks)
            tracks += [{'f': [k], 'q': [P[pi, 1]], 'v': [P[pi, 2]]} for pi in new]
            act = np.concatenate([act, base + np.arange(len(new))])
            a_last = np.concatenate([a_last, np.full(len(new), k)]); a_sl = np.concatenate([a_sl, np.zeros(len(new))])
            a_ic = np.concatenate([a_ic, np.zeros(len(new))]); a_lq = np.concatenate([a_lq, P[new, 1]])
    keep = [t for t in tracks if len(t['f']) >= min_points and (t['f'][-1] - t['f'][0]) * hop_s >= min_len_s]
    return [{k: np.array(v) for k, v in t.items()} for t in keep]


def merge_tracks(tracks, hop_s, df, max_gap_s=12.0):
    """Join collinear pieces of one curve separated by short dropouts (chains of best continuations)."""
    tracks = sorted(tracks, key=lambda t: t['f'][0])
    n = len(tracks)
    ends = [np.polyfit(t['f'][-40:], t['q'][-40:], 1) for t in tracks]
    begs = [np.polyfit(t['f'][:40], t['q'][:40], 1) for t in tracks]
    st = np.array([t['f'][0] for t in tracks]); en = np.array([t['f'][-1] for t in tracks])
    nxt, has_prev = [-1] * n, [False] * n
    for i in np.argsort(en):
        cand = np.nonzero((st > en[i]) & (st <= en[i] + max_gap_s / hop_s))[0]
        best = None
        for j in cand:
            if has_prev[j]:
                continue
            mid = (en[i] + st[j]) / 2
            pa, pb = np.polyval(ends[i], mid), np.polyval(begs[j], mid)
            if abs(pa - pb) < 2 * df + 0.006 * pa and abs(ends[i][0] - begs[j][0]) / hop_s < 0.6 + 0.01 * pa:
                if best is None or st[j] - en[i] < best[1]:
                    best = (j, st[j] - en[i])
        if best:
            nxt[i] = best[0]; has_prev[best[0]] = True
    out = []
    for i in range(n):
        if has_prev[i]:
            continue
        chain = [i]
        while nxt[chain[-1]] >= 0:
            chain.append(nxt[chain[-1]])
        out.append({k: np.concatenate([tracks[c][k] for c in chain]) for k in ('f', 'q', 'v')})
    return out


def track_quality(tr, hop_s, df, win_s=4.0):
    """Fill fraction over the span and RMS deviation from a local quadratic fit (Hz), plus median SNR."""
    f, q = tr['f'].astype(float), tr['q'].astype(float)
    span = f[-1] - f[0] + 1
    fill = len(f) / span
    half = win_s / hop_s / 2
    res = []
    for i in range(0, len(f), max(1, len(f) // 60)):
        m = np.abs(f - f[i]) <= half
        if m.sum() >= 6:
            c = np.polyfit(f[m] - f[i], q[m], 2)
            res.append(q[i] - c[2])
    rough = float(np.sqrt(np.mean(np.square(res)))) if res else np.inf
    return {'fill': float(fill), 'rough_hz': rough, 'snr': float(np.median(tr['v'])),
            'dur_s': float(span * hop_s)}


def filter_tracks(tracks, hop_s, df, min_fill=0.55, max_rough_bins=0.45, min_snr=7.0, min_dur_s=6.0):
    out = []
    for tr in tracks:
        qa = track_quality(tr, hop_s, df)
        if (qa['fill'] >= min_fill and qa['rough_hz'] <= max_rough_bins * df + 0.0006 * np.median(tr['q'])
                and qa['snr'] >= min_snr and qa['dur_s'] >= min_dur_s):
            tr['qa'] = qa
            out.append(tr)
    return out


def group_harmonics(tracks, max_rel_std=0.004, min_overlap=0.5):
    """Union tracks whose frequency ratio stays constant while both are present (same source)."""
    n = len(tracks); parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    st = np.array([t['f'][0] for t in tracks]); en = np.array([t['f'][-1] for t in tracks])
    for i in range(n):
        for j in np.nonzero((st < en[i]) & (en > st[i]))[0]:
            if j <= i:
                continue
            a, b = tracks[i], tracks[j]
            common, ia, ib = np.intersect1d(a['f'], b['f'], return_indices=True)
            if len(common) < min_overlap * min(len(a['f']), len(b['f'])) or len(common) < 12:
                continue
            r = a['q'][ia] / b['q'][ib]
            spread = np.ptp(np.log(a['q'][ia])) + np.ptp(np.log(b['q'][ib]))
            if spread < 0.01:  # both flat: a ratio test says nothing about a common source
                continue
            if np.std(np.log(r)) < max_rel_std:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# ---------------------------------------------------------------- Doppler model (straight line, constant speed)

def doppler_curve(t, tau_c, v, d, c=C_FIT):
    """Observed/emitted frequency ratio at reception times t for a source on a straight line at speed v,
    closest distance d at emission time tau_c. Closed-form emission-time solution."""
    T = t - tau_c
    s = (c * c * T - np.sqrt(c * c * v * v * T * T + (c * c - v * v) * d * d)) / (c * c - v * v)
    R = np.sqrt(d * d + v * v * s * s)
    vr = v * v * s / R
    return 1.0 / (1.0 + vr / c)


def fit_event(tlist, flist, wlist=None):
    """Fit shared (tau_c, v, d) and one emitted frequency per track. Returns dict with estimates and 1-sigma."""
    t_ref = float(np.mean(np.concatenate(tlist)))
    tlist = [np.asarray(t, float) - t_ref for t in tlist]
    allt = np.concatenate(tlist)
    nt = len(tlist)
    wl = wlist or [np.ones_like(t) for t in tlist]

    def resid(p):
        tau_c, lv, ld = p[:3]
        v, d = np.exp(lv), np.exp(ld)
        r = [(np.log(f) - p[3 + k] - np.log(doppler_curve(t, tau_c, v, d))) * np.sqrt(w)
             for k, (t, f, w) in enumerate(zip(tlist, flist, wl))]
        return np.concatenate(r)
    # data-driven starting point from the longest track: steepest descent time, asymptote ratio, slope
    k_ref = int(np.argmax([len(t) for t in tlist]))
    tr_, lf = tlist[k_ref], np.log(flist[k_ref])
    n_s = max(3, len(tr_) // 25) | 1
    lfs = np.convolve(lf, np.ones(n_s) / n_s, mode='same')
    core = slice(n_s, max(n_s + 1, len(lfs) - n_s))
    der = np.gradient(lfs, tr_)
    i_c = core.start + int(np.argmin(der[core])) if len(der[core]) else len(der) // 2
    slope = max(1e-5, -float(der[i_c]))
    r = float(np.exp(np.ptp(lfs[core]))) if len(lfs[core]) else 1.01
    v_lo = float(np.clip(C_FIT * (r - 1) / (r + 1), 8, 300))
    best = None
    for v0 in {min(300.0, v_lo * m) for m in (1.0, 1.4, 2.0)}:
        d_c = float(np.clip(v0 * v0 / (C_FIT * slope), 20, 30000))
        for d0 in (d_c / 2.5, d_c, d_c * 2.5):
            d0 = float(np.clip(d0, 10, 35000))
            p0 = [tr_[i_c] - d0 / C_FIT, np.log(v0), np.log(d0)]
            p0 += [np.median(np.log(f) - np.log(doppler_curve(t, p0[0], v0, d0))) for t, f in zip(tlist, flist)]
            try:
                r_ = least_squares(resid, p0, loss='soft_l1', f_scale=0.004, max_nfev=200, x_scale='jac',
                                   bounds=([allt.min() - 600, np.log(5), np.log(5)] + [-np.inf] * nt,
                                           [allt.max() + 600, np.log(330), np.log(40000)] + [np.inf] * nt))
            except Exception:
                continue
            if best is None or r_.cost < best.cost:
                best = r_
    p = best.x
    res = best.fun
    dof = max(1, len(res) - len(p))
    s2 = float(np.sum(res ** 2) / dof)
    try:
        J = best.jac
        cov = np.linalg.pinv(J.T @ J) * s2
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        sig = np.full(len(p), np.nan)
    tau_c, v, d = p[0] + t_ref, float(np.exp(p[1])), float(np.exp(p[2]))
    t_cpa = tau_c + d / C_FIT
    # uncertainty of reception CPA time ~ tau_c and d terms
    sig_tcpa = float(np.hypot(sig[0], d * sig[2] / C_FIT)) if np.all(np.isfinite(sig[:3])) else np.nan
    return {'tau_c': float(tau_c), 't_cpa': float(t_cpa), 't_cpa_sigma': sig_tcpa,
            'v': v, 'v_sigma_rel': float(sig[1]), 'd': d, 'd_sigma_rel': float(sig[2]),
            'f0': [float(np.exp(x)) for x in p[3:]], 'rms_log': float(np.sqrt(np.mean(best.fun ** 2))),
            'mach': v / C_FIT}


# ---------------------------------------------------------------- events

@dataclass
class Event:
    id: str
    tracks: list            # list of dicts: t (s from WAV start), f (Hz), snr (dB)
    fit: dict
    t_start: float
    t_end: float
    label: str
    score: float
    reasons: list = field(default_factory=list)

    def to_json(self):
        return {'id': self.id, 't_start': self.t_start, 't_end': self.t_end, 'label': self.label,
                'score': self.score, 'reasons': self.reasons, 'fit': self.fit,
                'tracks': [{'t': tr['t'].round(4).tolist(), 'f': tr['f'].round(4).tolist(),
                            'snr': tr['snr'].round(2).tolist(), **({'manual': True} if tr.get('manual') else {})}
                           for tr in self.tracks]}

    @staticmethod
    def from_json(d):
        trs = [{'t': np.array(x['t']), 'f': np.array(x['f']), 'snr': np.array(x['snr']),
                'manual': x.get('manual', False)} for x in d['tracks']]
        return Event(d['id'], trs, d['fit'], d['t_start'], d['t_end'], d['label'], d['score'], d.get('reasons', []))


def classify(ev_tracks, fit):
    """Heuristic label + confidence in [0, 1] with the reasons that drove it."""
    reasons = []
    t = np.concatenate([tr['t'] for tr in ev_tracks])
    dur = np.ptp(t)
    snr = float(np.median(np.concatenate([tr['snr'] for tr in ev_tracks])))
    # fractional Doppler swing actually observed (fitted curve over the observed span)
    tt = np.linspace(t.min(), t.max(), 200)
    g = doppler_curve(tt, fit['tau_c'], fit['v'], fit['d'])
    swing = float(np.log(g.max() / g.min()))
    slopes = []
    for tr in ev_tracks:
        if len(tr['t']) > 8:
            k = max(1, len(tr['t']) // 20)
            ft = np.convolve(tr['f'], np.ones(k) / k, mode='valid')
            slopes.append(np.mean(np.diff(ft) > 0.5))
    rising = float(np.mean(slopes)) if slopes else 0.0
    label = 'aircraft'
    if swing < 0.01:
        label = 'stationary/steady tone'; reasons.append(f'Doppler swing only {100 * swing:.1f}%')
    elif rising > 0.35:
        label = 'non-Doppler (rising)'; reasons.append(f'{100 * rising:.0f}% rising steps')
    elif fit['v'] < 40 and fit['d'] < 120:
        label = 'possible road vehicle'; reasons.append(f"v={fit['v']:.0f} m/s, d={fit['d']:.0f} m")
    elif dur < 8:
        label = 'too short'; reasons.append(f'{dur:.1f} s')
    s_snr = min(1.0, max(0.0, (snr - 5) / 10))
    s_dur = min(1.0, dur / 40)
    s_fit = float(np.exp(-0.5 * (fit['rms_log'] / 0.006) ** 2))
    s_swing = min(1.0, swing / 0.08)
    s_harm = min(1.0, 0.6 + 0.2 * (len(ev_tracks) - 1))
    score = s_snr ** 0.5 * s_dur * s_fit ** 0.5 * s_swing * s_harm
    if label != 'aircraft':
        score *= 0.2
    reasons += [f'snr {snr:.1f} dB', f'duration {dur:.0f} s', f'swing {100 * swing:.1f}%',
                f'fit rms {100 * fit["rms_log"]:.2f}%', f'{len(ev_tracks)} harmonic track(s)']
    return label, float(score), reasons, {'snr_db': snr, 'duration_s': float(dur), 'swing': swing}


def _to_obs(tr, hop_s, t0):
    return {'t': t0 + tr['f'] * hop_s, 'f': tr['q'].astype(float), 'snr': tr['v'].astype(float)}


def _swing(x):
    n_s = max(3, len(x['f']) // 15)
    lf = np.convolve(np.log(x['f']), np.ones(n_s) / n_s, mode='valid')
    return lf[0] - lf[-1] if len(lf) > 1 else 0.0


def _model_rms(fit, x, pad_s=0.0):
    """RMS (log) of a track against an event model after eliminating its own emitted frequency."""
    g = doppler_curve(x['t'], fit['tau_c'], fit['v'], fit['d'])
    r = np.log(x['f']) - np.log(g)
    r -= np.median(r)
    return float(np.sqrt(np.mean(r ** 2)))


def _fit(trs):
    return fit_event([x['t'] for x in trs], [x['f'] for x in trs], [np.clip(x['snr'] / 10, 0.3, 3) for x in trs])


def build_events(tracks, hop_s, t0, groups=None, min_dur_s=8.0, min_swing=0.006, accept_rms=0.0025,
                 max_rounds=4):
    """Model-based grouping: seed on the strongest descending track, fit a Doppler curve, absorb every
    time-overlapping track that follows the same curve (any emitted frequency), refit, repeat."""
    obs = [_to_obs(tr, hop_s, t0) for tr in tracks]
    free = set(range(len(obs)))
    strength = {i: len(o['t']) * np.median(o['snr']) for i, o in enumerate(obs)}
    events = []
    while True:
        seeds = [i for i in free if _swing(obs[i]) >= min_swing and np.ptp(obs[i]['t']) >= min_dur_s]
        if not seeds:
            break
        i0 = max(seeds, key=lambda i: strength[i])
        members = [i0]
        try:
            fit = _fit([obs[i0]])
        except Exception:
            free.discard(i0); continue
        for _ in range(max_rounds):
            ts, te = min(obs[i]['t'][0] for i in members), max(obs[i]['t'][-1] for i in members)
            added = False
            for j in sorted(free - set(members), key=lambda i: -strength[i]):
                x = obs[j]
                ov = min(te, x['t'][-1]) - max(ts, x['t'][0])
                if ov < 0.6 * np.ptp(x['t']) and ov < 15:
                    continue
                tol = max(accept_rms, 1.5 * fit['rms_log'])
                if _model_rms(fit, x) < tol:
                    members.append(j); added = True
            if not added:
                break
            try:
                fit = _fit([obs[i] for i in members])
            except Exception:
                break
            # drop members that no longer follow the refined model
            keep = [i for i in members if i == i0 or _model_rms(fit, obs[i]) < max(accept_rms, 2.0 * fit['rms_log'])]
            if len(keep) < len(members):
                members = keep
                fit = _fit([obs[i] for i in members])
        for i in members:
            free.discard(i)
        trs = [obs[i] for i in members]
        label, score, reasons, extra = classify(trs, fit)
        fit.update(extra)
        tt = np.concatenate([x['t'] for x in trs])
        events.append(Event('', trs, fit, float(tt.min()), float(tt.max()), label, score, reasons))
    events.sort(key=lambda e: -e.score)
    for k, e in enumerate(events):
        e.id = f'E{k + 1:02d}'
    return events


# ---------------------------------------------------------------- driver

DEFAULT_PARAMS = {'thr_db': 6.0, 'tol_bins': 1.2, 'max_gap_s': 2.0, 'merge_gap_s': 12.0, 'min_fill': 0.55,
                  'max_rough_bins': 0.45, 'min_snr': 7.0, 'min_dur_s': 6.0, 'fmin': 25.0, 'fmax': 1800.0}


def run(wav, out_dir, t0_utc, force=False, params=None, log=print):
    """Full automatic pass over one recording. t0_utc = UTC (unix s) of the first sample.
    Writes out_dir/{spec.*, tonal_*.npy, peaks.npy, tracks.json, events.json}; returns the events.json content."""
    from pathlib import Path
    import soundfile as sf
    from . import spectro
    an = Path(out_dir); an.mkdir(parents=True, exist_ok=True)
    ev_p = an / 'events.json'
    if ev_p.exists() and not force:
        return json.loads(ev_p.read_text())
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    decim, nperseg = spectro.analysis_params(sf.info(str(wav)).samplerate)
    log(f'spectrogram (decimation {decim}, window {nperseg} samples)')
    P, t, f, m = spectro.compute(wav, an, decim=decim, nperseg=nperseg)
    log('tonal enhancement')
    T, Z = spectro.tonal(P, m['df'], an)
    hop, df = m['hop_s'], m['df']
    L = lambda a, b, j0, j1: 10 * np.log10(np.asarray(P[a:b, j0:j1], np.float64) + 1e-20)
    pk = find_peaks(T, Z, L, df, fmin=p['fmin'], fmax=min(p['fmax'], f[-1] - 2 * df), thr_db=p['thr_db'])
    np.save(an / 'peaks.npy', pk.astype(np.float32))
    log(f'{len(pk)} spectral peaks; tracking')
    tr = track(pk, hop, df, max_gap_s=p['max_gap_s'], tol_bins=p['tol_bins'])
    n_raw = len(tr)
    tr = merge_tracks(tr, hop, df, max_gap_s=p['merge_gap_s']) if tr else []
    n_merged = len(tr)
    tr = filter_tracks(tr, hop, df, p['min_fill'], p['max_rough_bins'], p['min_snr'], p['min_dur_s'])
    log(f'{n_raw} raw tracks, {n_merged} after merging, {len(tr)} kept; grouping into events')
    events = build_events(tr, hop, m['t0'])
    out = {'audio': str(wav), 'wav_t0_utc_unix': float(t0_utc), 'params': p, 'spec': m,
           'counts': {'peaks': int(len(pk)), 'tracks_raw': n_raw, 'tracks_merged': n_merged,
                      'tracks_kept': len(tr), 'events': len(events)},
           'time_note': 'event times are seconds from the first audio sample; recording-clock UTC = '
                        'wav_t0_utc_unix + t. Acoustic time = that UTC - delta (latency, fitted in matching).',
           'events': [e.to_json() for e in events]}
    (an / 'tracks.json').write_text(json.dumps([{'t': (m['t0'] + x['f'] * hop).round(3).tolist(),
                                                 'f': x['q'].round(3).tolist(), 'qa': x['qa']} for x in tr]))
    ev_p.write_text(json.dumps(out))
    n = sum(1 for e in events if e.label == 'aircraft' and e.score >= 0.4)
    log(f'{len(events)} events, {n} aircraft-like with score >= 0.4')
    return out
