"""Experimental broadband flyover screening; never used by the tonal matcher.

Broad sound is not an aircraft classifier. The level model below (spherical spreading,
air absorption, the noise floor, a bounded directivity term, one clock offset shared by all sounds) is a
hypothesis, not a calibrated jet noise model: variable thrust, wind, ground effect and
microphone AGC can all change the envelope. Compatible grid cells and resampling are
sensitivity diagnostics, not location confidence bounds.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import scipy.signal as signal
from scipy.optimize import linear_sum_assignment
import soundfile as sf

from . import adsb, geo, physics

BANDS = ((100, 250), (250, 500), (500, 1000), (1000, 2000))
ABSORPTION_HZ = 354.0            # air absorption is evaluated at the centre of the 250-500 Hz band
GAIN_LIMITS = (0.3, 2.0)         # modest envelope compression or extra attenuation, never a flat or inverted model
DIRECTIVITY_DB = (-8.0, 4.0)     # level change per unit cos(angle between flight direction and observer)
MAX_RANGE_M = 25000              # audibility cutoff: about 40 dB of 354 Hz absorption alone at 15 C
LAG_STEP_S = 1.0                 # spacing of the shared clock offsets that are searched
MIN_ALTITUDE_M = 4000            # candidate jets: median MSL altitude while within reach of the search area
SPEED_MPS = (140, 340)           # and median horizontal speed over that same part of the track
LEVEL_CHUNK = 1e6                # cells x clock offsets x samples evaluated at once (memory bound)
GAP_FINALISE_S = 8.0             # audio_end_utc may trail the last sample by this much (file finalisation)
GAP_STEP_S = 2.0                 # spacing of the recording-gap sizes that are searched
LIMITATIONS = [
    'Experimental envelope matching, not validated on real high-altitude jets.',
    'Impact screening is a relative heuristic; listening labels are human impressions, not verified identities.',
    'Broadband candidates can be wind, road traffic, speech or microphone handling.',
    'The level model (spreading, ISO 9613-1 absorption at 354 Hz, a bounded directivity term) omits variable '
    'thrust, wind, ground effect and phone AGC.',
    'One clock offset is shared by all sounds; its slack beyond the clock prior is an assumption, not a measurement.',
    'Compatible cells and bootstrap distances are sensitivity diagnostics, not confidence intervals.',
    'One flight normally leaves a corridor or mirror ambiguity; a ranked cell is not a location estimate.',
]


class JetError(ValueError):
    pass


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def envelope(audio):
    """One-second median spectral levels: reject narrow tones, retain slow broad rises.

    Reading one second at a time bounds memory independently of recording length.
    Levels are uncalibrated PSD dB, not dB SPL. Channel averaging matches analyse.
    """
    levels, times, fast_times, fast_db = [], [], [], []
    with sf.SoundFile(str(audio)) as fh:
        fs = fh.samplerate
        if fs < 4000 or len(fh) / fs < 60:
            raise JetError('broadband inspection needs at least 60 s of audio at >= 4000 Hz')
        start = 0
        while True:
            x = fh.read(fs, dtype='float32', always_2d=True)
            if len(x) < fs / 2:
                break
            mono = x.mean(1)
            # Rounded edges support 11025 Hz and the final partial second.
            edges = np.rint(np.arange(0, len(mono), fs / 10)).astype(int)
            offset = 0
            for block in np.split(mono, edges[1:]):
                fast_times.append((start + offset + len(block) / 2) / fs)
                fast_db.append(float(20 * np.log10(np.sqrt(np.mean(block ** 2)) + 1e-12)))
                offset += len(block)
            f, P = signal.welch(mono, fs=fs, nperseg=min(len(x), fs // 4))
            L = 10 * np.log10(P + 1e-20)
            levels.append([float(np.median(L[(f >= a) & (f < b)])) for a, b in BANDS])
            times.append((start + len(x) / 2) / fs)
            start += len(x)
    t, raw = np.array(times), np.array(levels)
    smooth = ndi.gaussian_filter1d(ndi.median_filter(raw, size=(5, 1), mode='nearest'), 4, axis=0)
    floor = np.percentile(smooth, 20, axis=0)
    relative = smooth - floor
    ft, fd = np.array(fast_times), np.array(fast_db)
    steps = np.abs(np.diff(fd))
    adjacent_jump = np.maximum(np.r_[0, steps], np.r_[steps, 0])
    local_excess = fd - ndi.median_filter(fd, size=51, mode='nearest')
    impulse = (local_excess > 8) & (adjacent_jump > 8)
    # Mask locally, never discard a whole flyover because of a superposed bump.
    padded = ndi.maximum_filter1d(impulse.astype(int), size=21, mode='nearest') > 0
    edges = np.diff(np.r_[False, padded, False].astype(int))
    intervals = [[float(max(0, ft[a] - .05)), float(ft[b - 1] + .05)]
                 for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
    mask = np.zeros(len(t), bool)
    for a, b in intervals:
        mask |= (t + .5 >= a) & (t - .5 <= b)
    clean = raw.copy()
    if (~mask).any():
        for k in range(len(BANDS)):
            clean[:, k] = np.interp(t, t[~mask], raw[~mask, k])
    cleaned = ndi.gaussian_filter1d(ndi.median_filter(clean, size=(5, 1), mode='nearest'), 4, axis=0)
    clean_activity = np.median(cleaned - np.percentile(cleaned, 20, axis=0), axis=1)
    return {'t_s': t.tolist(), 'bands_hz': BANDS, 'band_db': raw.tolist(),
            'smoothed_band_db': smooth.tolist(), 'floor_db': floor.tolist(),
            'activity_db': np.median(relative, axis=1).tolist(),
            'clean_activity_db': clean_activity.tolist(),
            'transient_intervals_s': intervals,
            'fast': {'t_s': ft.tolist(), 'rms_dbfs': fd.tolist(),
                     'impulse': impulse.tolist()},
            'transient_rule': {'frame_s': .1, 'local_baseline_s': 5.1,
                               'min_excess_db': 8, 'min_adjacent_jump_db': 8, 'padding_s': 1},
            'note': 'Fast impacts are a heuristic, not a source classifier. Median PSD in each band; recording-wide 20th percentile baseline; no aircraft classification.'}


def find_events(env, windows=None):
    """Find slow multiband increases. Manual windows are seconds, not aircraft labels."""
    t, y = np.array(env['t_s']), np.array(env['activity_db'])
    if windows is not None:
        spans = []
        for a, b in sorted(windows):
            if not np.isfinite([a, b]).all() or a < 0 or b > t[-1] + 0.5 or b - a < 20:
                raise JetError('each --window needs 0 <= START < END <= audio duration and at least 20 s')
            if spans and a < spans[-1][1]:
                raise JetError('manual windows must not overlap (they would count one sound twice)')
            spans.append((a, b))
    else:
        base = float(np.percentile(y, 10))
        padded = np.r_[base, y, base]
        peaks, _ = signal.find_peaks(padded, prominence=3, distance=40)
        widths, _, left, right = signal.peak_widths(padded, peaks, rel_height=0.8)
        spans = []
        for p, w, a, b in zip(peaks, widths, left, right):
            if w < 20:
                continue
            a, b = max(t[0], a - 1 + t[0]), min(t[-1], b - 1 + t[0])
            if spans and a <= spans[-1][1] + 10:
                spans[-1] = (spans[-1][0], max(spans[-1][1], b))
            else:
                spans.append((float(a), float(b)))
    bands = np.array(env['smoothed_band_db']) - np.array(env['floor_db'])
    out = []
    if 'fast' in env:
        fast = env['fast']
        ft, fd = np.array(fast['t_s']), np.array(fast['rms_dbfs'])
        raw_bands = np.array(env['band_db'])
    for a, b in spans:
        ix = np.flatnonzero((t >= a) & (t <= b))
        if not len(ix):
            continue
        peak = int(ix[np.argmax(y[ix])])
        flags = []
        if a < 10 or b > t[-1] - 10:
            flags.append('clipped_by_recording')
        if b - a > 300:
            flags.append('long_or_overlapping_sound')
        if (bands[peak] >= 3).sum() < 2:
            flags.append('limited_frequency_support')
        # A fast edge is a useful warning, not grounds for claiming it is not an aircraft.
        rise = t[peak] - a
        if rise < 10:
            flags.append('abrupt_onset')
        out.append({'id': f'J{len(out) + 1:02d}', 't_start': a, 't_end': b,
                    'peak_s': float(t[peak]), 'peak_excess_db': float(y[peak]),
                    'bands_above_3db': int((bands[peak] >= 3).sum()),
                    'origin': 'manual_window' if windows is not None else 'automatic',
                    'label': 'unclassified broadband event', 'flags': flags})
        if 'fast' in env:
            sel = (ft >= a) & (ft <= b)
            impulses = np.array(fast['impulse'], bool) & sel
            intervals = [[max(a, lo), min(b, hi)] for lo, hi in env['transient_intervals_s']
                         if hi >= a and lo <= b]
            out[-1]['screening'] = {
                'impulse_frames': int(impulses.sum()), 'transient_intervals_s': intervals,
                'fast_step_p99_db': float(np.percentile(np.abs(np.diff(fd[sel])), 99)) if sel.sum() > 1 else 0,
                'low_to_high_psd_db': float(np.median(raw_bands[ix, :2].mean(1) - raw_bands[ix, 3])),
                'note': 'Relative shape diagnostics; no aircraft probability or absolute loudness cutoff.'}
            if intervals:
                flags.append('fast_transient_contamination')
    return out


def apply_review(case, events, path):
    """Use only explicit likely_jet labels; preserve notes without interpreting free text.

    IDs alone are unsafe after re-detection: require the same case, start time and
    padded clip windows. A changed selection must be reviewed again.
    """
    from .case import _parse_utc, CaseError
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if data['recording'] != case.dir.name or _parse_utc(data['audio_start_utc']) != case.t0:
            raise JetError('listening review belongs to a different recording or start time')
        rows = data['candidates']
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            raise JetError('listening review candidates must be a list of objects')
        by_id = {r['id']: r for r in rows}
        if len(by_id) != len(rows) or set(by_id) != {e['id'] for e in events}:
            raise JetError('listening review IDs differ from current candidates; review them again')
        duration = case.audio_info()['duration_s']
        labels = {'likely_jet', 'likely_road', 'mixed', 'other', 'unsure', 'unreviewed'}
        for e in events:
            r = by_id[e['id']]
            bounds = [float(r['candidate_clip_start_s']), float(r['candidate_clip_end_s'])]
            expected = [max(0, e['t_start'] - 20), min(duration, e['t_end'] + 20)]
            if not np.isfinite(bounds).all() or not np.allclose(bounds, expected, rtol=0, atol=.1):
                raise JetError(f"listening review clip for {e['id']} differs from current detection; review it again")
            if r['label'] not in labels or not isinstance(r.get('notes', ''), str):
                raise JetError('invalid listening review label or notes')
            e['review'] = {'label': r['label'], 'notes': r.get('notes', ''),
                           'selected': r['label'] == 'likely_jet',
                           'note': 'Human listening impression, not verified aircraft identity.'}
    except (OSError, ValueError, TypeError, KeyError, CaseError) as exc:
        raise JetError(f'cannot use listening review: {exc}') from exc
    return [e for e in events if e['review']['selected']]


def _observations(env, event):
    """Fixed observed support for both fitting and saved plots; omit interpolated impacts."""
    et = np.array(env['t_s'])
    ey = np.array(env.get('clean_activity_db', env['activity_db']))
    a, b = max(et[0], event['t_start'] - 25), min(et[-1], event['t_end'] + 25)
    rel = np.arange(a, b + .1, 4.)
    keep = np.ones(len(rel), bool)
    for lo, hi in env.get('transient_intervals_s', []):
        # Any one-second PSD block touching a masked frame was interpolated.
        keep &= ~((rel >= lo - 1) & (rel <= hi + 1))
    return rel[keep], np.interp(rel[keep], et, ey)


def level_terms(traj, observers, t_rx, c, alpha=0.0):
    """Model level in dB (spherical spreading, air absorption alpha in dB/m) and the directivity cosine.

    Both use the retarded emission time; R is slant range then. cos is +1 when the aircraft flies straight
    towards the observer and -1 when it flies away. The additive source level is unknown. This deliberately
    has no Doppler-derived pitch, speed or absolute acoustic range.
    """
    _, R, tau = physics.doppler(traj, observers, t_rx, c)
    P, V = traj.at(tau)
    D = np.asarray(observers, float).reshape(-1, 1, 3) - P
    cos = np.einsum('mkj,mkj->mk', V, D) / (np.linalg.norm(V, axis=-1) * R + 1e-9)
    return -20 * np.log10(np.maximum(R, 1)) - alpha * R, cos, R


def above_floor(x, u, peak_db):
    """Model level above the recording's noise floor, as the observed activity is.

    The source level is set so the model peak equals the observed peak (peak_db above the floor); the floor
    adds in power, so a distant aircraft fades into the floor instead of falling without limit. The
    directivity term is weighted by the aircraft's share of the power, so it vanishes at the floor too.
    """
    s = 10 * np.log10(10 ** (max(peak_db, .5) / 10) - 1)
    v = s + x - np.max(np.where(np.isfinite(x), x, -np.inf), axis=-1, keepdims=True)
    z = 10 * np.log10(1 + 10 ** (v / 10))
    return z, u * (1 - 10 ** (-z / 10))


def jet_candidate(traj, enu, reach_m):
    """High, fast traffic while within reach_m (horizontal) of the search centre during the recording window.

    Judged only over that part of the track: the median of a whole downloaded track mostly reflects how much
    of its climb or descent elsewhere happened to be downloaded."""
    h = np.hypot(traj.P[:, 0], traj.P[:, 1])
    near = np.isfinite(h) & (h <= reach_m) & np.isfinite(traj.P[:, 2]) & np.isfinite(traj.V[:, 0])
    if not near.any():
        return False
    altitude = np.median(enu.to_geodetic(traj.P[near])[2])
    speed = np.median(np.linalg.norm(traj.V[near, :2], axis=1))
    return bool(altitude >= MIN_ALTITUDE_M and SPEED_MPS[0] <= speed <= SPEED_MPS[1])


def range_level(traj, observers, t_rx, c):
    """Fixed-source inverse-square intensity in dB, including retarded emission time."""
    return level_terms(traj, observers, t_rx, c)[0]


def fit_level(predicted, observed, directivity=None):
    """Fit observed ~ offset + gain * predicted (+ b * directivity) with bounded gain and b.

    Returns (residual fraction, gain, b). The gain stays in GAIN_LIMITS: modest compression or extra
    attenuation, never an inverted or unlimited stretch of an almost flat distant source. b stays in
    DIRECTIVITY_DB, so it cannot replace the range term. Fixed support is mandatory: sparse tracks cannot win
    by fitting only a peak. Constant or near-flat audio has no information.
    """
    x, y = np.asarray(predicted, float), np.asarray(observed, float)
    shape = x.shape[:-1]
    y = y - y.mean()
    energy = float(np.sum(y ** 2))
    if np.ptp(y) < 2 or energy < 1:
        return np.ones(shape), np.zeros(shape), np.zeros(shape)
    u = np.zeros_like(x) if directivity is None else np.asarray(directivity, float)
    (g0, g1), (b0, b1) = GAIN_LIMITS, (DIRECTIVITY_DB if directivity is not None else (0.0, 0.0))
    valid = np.isfinite(x).all(axis=-1) & np.isfinite(u).all(axis=-1)
    x, u = np.where(np.isfinite(x), x, 0), np.where(np.isfinite(u), u, 0)
    x, u = x - x.mean(axis=-1, keepdims=True), u - u.mean(axis=-1, keepdims=True)
    xx, xu, uu = (x * x).sum(-1), (x * u).sum(-1), (u * u).sum(-1)
    xy, uy = (x * y).sum(-1), (u * y).sum(-1)
    # A convex quadratic on a box has its minimum inside or on an edge: try all of them.
    cands = []
    for g in (g0, g1):
        b = np.divide(uy - g * xu, uu, out=np.zeros(shape), where=uu > 1e-12)
        cands.append((np.full(shape, g), np.clip(b, b0, b1)))
    for b in (b0, b1):
        g = np.divide(xy - b * xu, xx, out=np.full(shape, g0), where=xx > 1e-12)
        cands.append((np.clip(g, g0, g1), np.full(shape, b)))
    det = xx * uu - xu * xu
    ok = det > 1e-9 * np.maximum(xx * uu, 1e-30)
    gi = np.divide(uu * xy - xu * uy, det, out=np.full(shape, g0), where=ok)
    bi = np.divide(xx * uy - xu * xy, det, out=np.full(shape, b0), where=ok)
    inside = ok & (gi >= g0) & (gi <= g1) & (bi >= b0) & (bi <= b1)
    cands.append((np.where(inside, gi, g0), np.where(inside, bi, b0)))
    q = np.stack([g * g * xx + 2 * g * b * xu + b * b * uu - 2 * g * xy - 2 * b * uy + energy for g, b in cands])
    k = np.argmin(q, axis=0)
    gain, b = np.choose(k, [g for g, _ in cands]), np.choose(k, [b for _, b in cands])
    cost = np.clip(np.min(q, axis=0) / energy, 0, 1)
    return np.where(valid, cost, 1.0), gain, b


def shape_cost(predicted, observed, directivity=None):
    """Residual fraction of the bounded level fit; 1 for gaps, anticorrelation or flat audio."""
    return fit_level(predicted, observed, directivity)[0]


def _assign(costs, weights=None):
    """One flight per sound and one sound per flight; unmatched costs one.

    costs: event x position x aircraft. Return per-event costs and associations.
    Repeating a bootstrap unit changes its weight, not its number of assignments.
    """
    per, assoc = _assign_clocks(costs[:, :, None, :], np.zeros((1, len(costs)), int), weights)
    return per[:, :, 0], assoc[:, :, 0]


def _assign_lags(costs, weights=None):
    """Assignments for every (cell, shared clock offset); costs are sound x cell x offset x flight."""
    K, M, L, A = costs.shape
    return _assign_clocks(costs, np.repeat(np.arange(L)[:, None], K, axis=1), weights)


def _best_flights(costs):
    """Each sound's own best flight and its cost per cell and offset (sound x cell x offset)."""
    arg = np.argmin(costs, axis=3)
    return arg, np.take_along_axis(costs, arg[..., None], axis=3)[..., 0].astype(float)


def _assign_clocks(costs, idx, weights=None, best=None):
    """_assign for every (cell, clock hypothesis), without gathering the costs per hypothesis.

    costs: sound x cell x offset x flight; hypothesis h reads sound k at offset idx[h, k]. best: the result of
    _best_flights(costs), reused across bootstrap calls. Returns per-sound costs and flights (sound x cell x h).
    """
    K, M, L, A = costs.shape
    H = len(idx)
    weights = np.ones(K) if weights is None else np.asarray(weights, float)
    if A == 0:
        return np.ones((K, M, H)), np.full((K, M, H), -1, int)
    arg, low = _best_flights(costs) if best is None else best
    ix = np.broadcast_to(np.asarray(idx).T[:, None, :], (K, M, H))
    arg, low = np.take_along_axis(arg, ix, axis=2), np.take_along_axis(low, ix, axis=2)
    per, assoc = np.where(low < 1, low, 1.0), np.where(low < 1, arg, -1)
    # Each sound's own best flight is optimal unless two counted sounds want the same one.
    active = np.flatnonzero(weights > 0)
    clash = np.zeros((M, H), bool)
    for i, a in enumerate(active):
        for b in active[i + 1:]:
            clash |= (assoc[a] == assoc[b]) & (assoc[a] >= 0)
    n = len(active)
    for m, h in zip(*np.nonzero(clash)):
        C = np.concatenate([costs[active, m, idx[h, active]], np.ones((n, n))], axis=1)
        rows, cols = linear_sum_assignment(C * weights[active, None])
        for r, a in zip(rows, cols):
            k = active[r]
            per[k, m, h], assoc[k, m, h] = (C[r, a], a) if a < A and C[r, a] < 1 else (1.0, -1)
    return per, assoc


def clock_hypotheses(events, lags, gap_s=0.0):
    """Clock offsets to search for each sound.

    Without a recording gap every sound shares one offset. With one (a pause or dropout: audio_end_utc is
    later than the start plus the audio duration), sounds from an unknown one on arrive later by the gap:
    their offset is the shared offset minus the gap. The measured gap includes up to GAP_FINALISE_S of file
    finalisation, so smaller gaps are tried too (whole clock steps, so the offsets after the gap stay on the
    same grid). Returns the union of offsets, an index table (hypothesis x sound) into it, and per hypothesis
    (offset before the gap, gap, first sound after it or None).
    """
    lags = np.asarray(lags, float)
    K = len(events)
    rows, desc = [np.repeat(lags[:, None], K, axis=1)], [(float(x), 0.0, None) for x in lags]
    if gap_s and gap_s > 0:
        order = np.argsort([e['t_start'] for e in events], kind='stable')
        gaps = np.arange(gap_s, max(gap_s - GAP_FINALISE_S, LAG_STEP_S) - 1e-9, -GAP_STEP_S)
        gaps = np.unique(np.maximum(np.round(gaps / LAG_STEP_S), 1) * LAG_STEP_S)[::-1]
        for j in range(K):
            after = np.zeros(K, bool)
            after[order[j:]] = True
            for g in gaps:
                rows.append(lags[:, None] - g * after)
                desc += [(float(x), float(g), events[order[j]]['id']) for x in lags]
    table = np.round(np.concatenate(rows), 6)
    union, inv = np.unique(table, return_inverse=True)
    return union, inv.reshape(table.shape), desc


def _best_offsets(per, weights=None):
    """Best mean score per cell over the shared clock offsets, and the index of that offset."""
    s = 1 - np.average(per, weights=weights, axis=0)
    li = np.argmax(s, axis=1)
    return s[np.arange(len(s)), li], li


def score_grid(events, env, tracks, enu, observers, case, lags, alpha=0.0, shift_s=0, log=None):
    """Compare entire padded event windows to each candidate at every shared clock offset.

    Returns costs (sound x cell x offset x flight); _assign_clocks turns them into one-to-one assignments.
    Every null uses the same candidates and selection procedure. Only the sound
    times move, wrapped as whole windows so no discontinuity is spliced into one.
    """
    et = np.array(env['t_s'])
    K, M, L, A = len(events), len(observers), len(lags), len(tracks)
    if K * M * L * A > 6e7:
        raise JetError(f'{K} sounds x {M} cells x {L} clock offsets x {A} flights is too large; increase --step, '
                       'reduce --timing-slack or select fewer sounds')
    costs = np.ones((K, M, L, A), np.float32)
    duration = et[-1] - et[0]
    for k, event in enumerate(events):
        a, b = max(et[0], event['t_start'] - 25), min(et[-1], event['t_end'] + 25)
        rel, y = _observations(env, event)
        if len(rel) < 5 or np.ptp(rel) < 20:
            continue
        if shift_s:
            a_new = et[0] + (a - et[0] + shift_s) % duration
            rel = rel + a_new - a
        rx = case.t0 + rel
        # The level terms of one cell at one reception time do not depend on the clock offset, and the
        # lagged times overlap heavily (4 s samples, 1-2 s offsets): evaluate each distinct time once and
        # gather it per offset. Exact: every element sees the same arithmetic as a per-offset evaluation.
        times, inv = np.unique(rx[None, :] - np.asarray(lags, float)[:, None], return_inverse=True)
        inv = inv.reshape(L, len(rx))
        chunk = max(1, int(LEVEL_CHUNK // (L * len(rx))))
        compared = 0
        for ai, (track, traj) in enumerate(tracks):
            # Emission precedes reception; reject no-support hypotheses without extrapolation.
            lo, hi = rx[0] - max(lags) - 180, rx[-1] - min(lags)
            if traj.t[-1] < lo or traj.t[0] > hi:
                continue
            P = traj.P[(traj.t >= lo) & (traj.t <= hi)]
            P = P[np.isfinite(P).all(axis=1)]
            # Skip flights that never come within the audibility cutoff of the search disc (exact speed-up).
            if not len(P) or np.min(np.hypot(P[:, 0], P[:, 1])) > case.radius_m + MAX_RANGE_M:
                continue
            compared += 1
            c = physics.path_sound_speed(case.ground_temp_c, float(np.nanmedian(observers[:, 2])),
                                         float(np.nanmedian(traj.P[:, 2])))
            for start in range(0, M, chunk):
                stop = min(M, start + chunk)
                x, u, R = (a[:, inv] for a in level_terms(traj, observers[start:stop], times, c, alpha))
                z, ue = above_floor(x, u, y.max())          # cell x offset x sample
                cc = shape_cost(z, y, ue)
                # This is an audibility search cutoff, not a measured sound range.
                near = np.min(np.where(np.isfinite(R), R, np.inf), axis=-1) <= MAX_RANGE_M
                costs[k, start:stop, :, ai] = np.where(near, cc, 1)
        if log:
            log(f"  {event['id']}: compared {compared} of {A} high/fast flight tracks")
    return costs


def _region(E, N, selected, enu, step):
    if not selected.any():
        return None
    lat, lon, _ = enu.to_geodetic(np.column_stack([E[selected], N[selected], np.zeros(selected.sum())]))
    return {'lat_min': float(lat.min()), 'lat_max': float(lat.max()),
            'lon_min': float(lon.min()), 'lon_max': float(lon.max()),
            'sampled_area_km2': float(selected.sum() * step ** 2 / 1e6),
            'n_cells': int(selected.sum()), 'note': 'Union of near-best sampled cells; bbox can span disconnected modes.'}


def match(case, env, events, step_m=2000, timing_slack_s=5, log=print):
    if not np.isfinite(step_m) or step_m < 250:
        raise JetError('--step must be finite and at least 250 m (this is a coarse experiment)')
    if not np.isfinite(timing_slack_s) or not 0 <= timing_slack_s <= 60:
        raise JetError('--timing-slack must be between 0 and 60 seconds')
    enu = geo.ENU(*case.center, 0)
    if step_m > case.radius_m:
        raise JetError('--step must not exceed the search radius')
    n = int(np.ceil(case.radius_m / step_m))
    g = np.arange(-n, n + 1) * step_m
    EE, NN = np.meshgrid(g, g)
    keep = EE ** 2 + NN ** 2 <= case.radius_m ** 2
    E, N = EE[keep], NN[keep]
    if len(E) > 50000:
        raise JetError('grid exceeds 50000 positions; increase --step')
    lat, lon, _ = enu.to_geodetic(np.column_stack([E, N, np.zeros(len(E))]))
    z = case.terrain().elevation(lat, lon) + case.mic_height
    # Broad kilometre-scale experiment: use the same regional height datum for
    # source and observer, avoiding an extra online geoid dependency.
    observers = enu.from_geodetic(lat, lon, z)
    # Archive days or imported files may repeat an ICAO address. Combine those
    # pieces before assignment so one aircraft is never counted independently twice.
    grouped = {}
    for track in adsb.load_tracks(case.tracks_path):
        grouped.setdefault(track.icao24.lower(), []).append(track)
    tracks = []
    for ident, pieces in grouped.items():
        t = np.concatenate([tr.t for tr in pieces])
        order = np.argsort(t)
        _, unique = np.unique(t[order], return_index=True)
        ix = order[unique]
        arrays = [np.concatenate([getattr(tr, key) for tr in pieces])[ix]
                  for key in ('t', 'lat', 'lon', 'alt_msl_m', 'alt_kind')]
        first = pieces[0]
        track = adsb.Track(ident, first.callsign, first.icao_type, first.category, first.source, *arrays, meta=first.meta)
        try:
            traj = physics.Trajectory(track, enu, case.t0 - 600, case.window[1] + 600, dt=2, geoid_n=0)
        except ValueError:
            continue
        if jet_candidate(traj, enu, case.radius_m + MAX_RANGE_M):
            tracks.append((track, traj))
    if not tracks:
        return {'status': 'no_candidate_tracks', 'summary': 'No high-altitude, fast flight tracks in the supplied data.', 'location_estimate': None}, None
    lo, hi = case.delay_prior
    lags = np.arange(lo - timing_slack_s, hi + timing_slack_s + LAG_STEP_S / 2, LAG_STEP_S)
    gap = case.clock_gap_s or 0.0
    union, idx, clock = clock_hypotheses(events, lags, gap)
    alpha = float(physics.absorption_db_per_m(ABSORPTION_HZ, case.ground_temp_c))
    log(f'experimental jet matching: {len(events)} sounds, {len(tracks)} tracks, {len(E)} cells, '
        f'{len(lags)} shared clock offsets' + (f' x {len(idx) // len(lags) - 1} recording-gap hypotheses '
                                               f'(gap {gap:.0f} s)' if gap else '') +
        f', absorption {alpha * 1000:.2f} dB/km')
    costs = score_grid(events, env, tracks, enu, observers, case, union, alpha, log=log)
    pre = _best_flights(costs)
    per_l, assoc_l = _assign_clocks(costs, idx, best=pre)
    scores, lag_i = _best_offsets(per_l)
    cells = np.arange(len(E))
    per, assoc = per_l[:, cells, lag_i], assoc_l[:, cells, lag_i]
    best = int(np.argmax(scores))
    best_lag, best_gap, first_after = clock[lag_i[best]]
    sound_lag = union[idx[lag_i[best]]]
    accepted = per[:, best] <= 0.5
    supported = int(accepted.sum())
    near = scores >= scores[best] - 0.05
    nulls = []
    for fraction in (0.25, 0.5, 0.75):
        log(f'  time-shift null {fraction:.0%}')
        cn = score_grid(events, env, tracks, enu, observers, case, union, alpha,
                        shift_s=fraction * (env['t_s'][-1] - env['t_s'][0]))
        ns, _ = _best_offsets(_assign_clocks(cn, idx)[0])
        nulls.append({'shift_s': fraction * (env['t_s'][-1] - env['t_s'][0]), 'best_score': float(ns.max())})
    # Reassign on each leave-one-out/whole-event bootstrap, retaining one-to-one associations.
    loo = []
    if len(events) >= 2:
        for k, e in enumerate(events):
            weights = np.ones(len(events)); weights[k] = 0
            pp, _ = _assign_clocks(costs, idx, weights, best=pre)
            ss, _ = _best_offsets(pp, weights)
            i = int(np.argmax(ss))
            loo.append({'left_out': e['id'], 'movement_m': float(np.hypot(E[i] - E[best], N[i] - N[best]))})
    boot = []
    if supported >= 2:
        active = np.flatnonzero(accepted)
        rng = np.random.default_rng(0)
        for _ in range(100):
            weights = np.bincount(rng.choice(active, size=len(active)), minlength=len(events))
            pp, _ = _assign_clocks(costs, idx, weights, best=pre)
            ss, _ = _best_offsets(pp, weights)
            i = int(np.argmax(ss))
            boot.append(float(np.hypot(E[i] - E[best], N[i] - N[best])))
    flags = []
    if supported < 3:
        flags.append('Fewer than three independently assigned sound events fit; corridor/mirror ambiguity remains.')
    if max(n['best_score'] for n in nulls) >= scores[best] - 0.05:
        flags.append('Wrong-time flight data fit about as well; matches can be coincidence.')
    if near.sum() / len(E) > 0.25:
        flags.append('Near-best cells cover over a quarter of the search area.')
    if np.any(np.hypot(E[near], N[near]) > case.radius_m - 1.5 * step_m):
        flags.append('Near-best region reaches the search boundary; the prior may truncate it.')
    if any(e['flags'] for e, ok in zip(events, accepted) if ok):
        flags.append('At least one fitted sound is clipped, abrupt, long or has limited spectral support.')
    if loo and max(x['movement_m'] for x in loo) > 5000:
        flags.append('Leaving out one sound moves the top grid cell by more than 5 km.')
    if supported and len(lags) > 1 and min(abs(best_lag - lags[0]), abs(best_lag - lags[-1])) < LAG_STEP_S / 2:
        flags.append('The fitted clock offset reaches its allowed limit; the chosen slack may truncate alternatives.')
    if gap:
        flags.append(f'The recording is {gap:.0f} s shorter than its wall-clock span (audio_end_utc): sounds after an '
                     'unknown point are tried with a later clock, which adds timing freedom.')
    verdict = 'failed' if not supported else ('ambiguous' if flags else 'exploratory')
    assignments = []
    for k, e in enumerate(events):
        ai = int(assoc[k, best])
        assignment = {'event': e['id'], 'icao24': tracks[ai][0].icao24 if ai >= 0 else None,
                            'aircraft': tracks[ai][0].label if ai >= 0 else None,
                            'shape_score': float(1 - per[k, best]), 'passes_shape_screen': bool(accepted[k]),
                            'timing_shift_s': float(sound_lag[k]) if ai >= 0 else None}
        if ai >= 0:
            rel, observed = _observations(env, e)
            traj = tracks[ai][1]
            c = physics.path_sound_speed(case.ground_temp_c, float(np.nanmedian(observers[:, 2])),
                                         float(np.nanmedian(traj.P[:, 2])))
            x, u, _ = level_terms(traj, observers[best:best + 1], case.t0 + rel - sound_lag[k], c, alpha)
            x, u = above_floor(x, u, observed.max())
            _, gain, b = fit_level(x[0], observed, u[0])
            x, u = x[0] - x[0].mean(), u[0] - u[0].mean()
            assignment.update({'t_s': rel.tolist(), 'observed_db': observed.tolist(),
                               'predicted_db': (observed.mean() + gain * x + b * u).tolist(),
                               'shape_gain': float(gain), 'directivity_db': float(b)})
        assignments.append(assignment)
    result = {'status': verdict, 'summary': f'{supported}/{len(events)} broadband events pass the experimental shape screen.',
              'location_estimate': None, 'best_score': float(scores[best]), 'n_supported': supported,
              'clock_offset_s': best_lag if supported else None,
              'clock_gap': ({'measured_s': gap, 'fitted_s': best_gap, 'first_sound_after': first_after}
                            if gap and supported else None),
              'best_grid_hypothesis': {'lat': float(lat[best]), 'lon': float(lon[best])} if supported else None,
              'compatible_region': _region(E, N, near, enu, step_m) if supported else None,
              'flags': flags, 'assignments': assignments, 'null_tests': nulls,
              'stability': {'leave_one_out': loo,
                            'bootstrap_grid_68_m': float(np.percentile(boot, 68)) if boot else None,
                            'bootstrap_grid_90_m': float(np.percentile(boot, 90)) if boot else None,
                            'note': 'Uncalibrated grid stability only, conditional on the event/model assumptions.'},
              'settings': {'step_m': step_m, 'timing_slack_s': timing_slack_s, 'lag_range_s': [float(lags.min()), float(lags.max())],
                           'lag_step_s': LAG_STEP_S,
                           'clock_offset': 'one offset shared by all sounds' + (
                               f'; later by a {max(gap - GAP_FINALISE_S, LAG_STEP_S):.0f}-{gap:.0f} s recording gap '
                               'from an unknown sound on' if gap else ''),
                           'absorption_db_per_km': round(alpha * 1000, 3), 'absorption_frequency_hz': ABSORPTION_HZ,
                           'gain_limits': list(GAIN_LIMITS), 'directivity_db_limits': list(DIRECTIVITY_DB),
                           'min_altitude_m': MIN_ALTITUDE_M, 'ground_speed_mps': list(SPEED_MPS),
                           'candidate_filter': 'medians while within the search radius plus the slant range cutoff',
                           'max_slant_range_m': MAX_RANGE_M,
                           'near_best_score_tolerance': 0.05, 'bootstrap_resamples': len(boot)}}
    grid = {'E_m': E, 'N_m': N, 'lat': lat, 'lon': lon, 'score': scores,
            'near_best': near, 'per_event_cost': per, 'flight_index': assoc, 'clock_offset_s': np.array([clock[h][0] for h in lag_i])}
    return result, grid


def report(path, env, events, result, grid=None):
    """Standalone private HTML, using installed Plotly; no map tiles or service calls."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2 if grid is not None else 1, cols=1,
                        subplot_titles=['Broadband activity: candidates are not aircraft identifications'] +
                        (['Experimental grid scores: not a probability map'] if grid is not None else []))
    t = np.array(env['t_s']) / 60
    bands = np.array(env['smoothed_band_db']) - np.array(env['floor_db'])
    for k, (a, b) in enumerate(BANDS):
        fig.add_trace(go.Scatter(x=t, y=bands[:, k], name=f'{a}-{b} Hz', opacity=0.45), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=env['activity_db'], name='Median across bands', line={'color': 'black', 'width': 2}), row=1, col=1)
    if 'clean_activity_db' in env:
        fig.add_trace(go.Scatter(x=t, y=env['clean_activity_db'], name='Impact-screened activity',
                                line={'color': 'green', 'width': 2}), row=1, col=1)
    for lo, hi in env.get('transient_intervals_s', []):
        fig.add_vrect(x0=lo / 60, x1=hi / 60, fillcolor='red', opacity=.25, line_width=0, row=1, col=1)
    for assignment in result.get('assignments', []):
        if 'predicted_db' in assignment:
            fig.add_trace(go.Scatter(x=np.array(assignment['t_s']) / 60, y=assignment['predicted_db'],
                                    name=assignment['event'] + ' fitted range envelope',
                                    line={'dash': 'dash', 'width': 2}), row=1, col=1)
    for e in events:
        fig.add_vrect(x0=e['t_start'] / 60, x1=e['t_end'] / 60, fillcolor='orange' if e.get('selected_for_matching', True) else 'grey', opacity=0.15,
                      line_width=0, annotation_text=e['id'], row=1, col=1)
    fig.update_xaxes(title_text='Minutes from recording start', row=1, col=1)
    fig.update_yaxes(title_text='Broadband increase (relative dB)', row=1, col=1)
    if grid is not None:
        fig.add_trace(go.Scatter(x=grid['lon'], y=grid['lat'], mode='markers', name='Grid score',
                                marker={'size': 6, 'color': grid['score'], 'colorscale': 'Viridis', 'showscale': True}), row=2, col=1)
        fig.update_xaxes(title_text='Longitude', row=2, col=1)
        fig.update_yaxes(title_text='Latitude', row=2, col=1)
    fig.update_layout(title='Experimental jet screening - ' + result['status'], height=850 if grid is not None else 520)
    from html import escape
    notes = '<h2>' + escape(result['summary']) + '</h2><ul>' + ''.join('<li>' + escape(x) + '</li>' for x in LIMITATIONS + result.get('flags', [])) + '</ul>'
    notes += '<h2>Candidate screening and listening review</h2><p>Red spans mark abrupt local impacts; green is the impact-screened envelope. Neither identifies a source. Audio clips remain unchanged.</p>'
    notes += '<table border="1" cellpadding="6"><tr><th>ID</th><th>Use</th><th>Review</th><th>Impact frames</th><th>Fast change p99 (dB)</th><th>Low/high PSD (dB)</th><th>Flags / notes</th></tr>'
    for e in events:
        screening, review = e.get('screening', {}), e.get('review', {})
        values = [e['id'], 'yes' if e.get('selected_for_matching', True) else 'no',
                  review.get('label', 'unreviewed'), str(screening.get('impulse_frames', '-')),
                  f"{screening['fast_step_p99_db']:.1f}" if 'fast_step_p99_db' in screening else '-',
                  f"{screening['low_to_high_psd_db']:.1f}" if 'low_to_high_psd_db' in screening else '-',
                  ', '.join(e['flags']) + ' ' + review.get('notes', '')]
        notes += '<tr>' + ''.join('<td>' + escape(v) + '</td>' for v in values) + '</tr>'
    notes += '</table>'
    html = fig.to_html(include_plotlyjs=True, full_html=True)
    Path(path).write_text(html.replace('</body>', notes + '</body>'), encoding='utf-8')


def run(case, analyse_only=False, windows=None, ids=None, step_m=2000, timing_slack_s=5, log=print, review_labels=None):
    case.audio_info()
    out = case.dir / 'jets'
    out.mkdir(exist_ok=True)
    env = envelope(case.audio_path)
    candidates = find_events(env, windows=windows)
    if review_labels and windows is not None:
        raise JetError('--review-labels cannot be combined with manual windows')
    events = apply_review(case, candidates, review_labels) if review_labels else candidates
    if ids:
        known = {e['id'] for e in candidates}
        if set(ids) - known:
            raise JetError('unknown broadband event IDs: ' + ', '.join(sorted(set(ids) - known)))
        events = [e for e in events if e['id'] in ids]
    selected = {e['id'] for e in events}
    for e in candidates:
        e['selected_for_matching'] = e['id'] in selected
    log(f'{len(events)} selected of {len(candidates)} unclassified broadband events (not confirmed aircraft)')
    for e in events:
        log(f"  {e['id']}: {e['t_start'] / 60:.2f}-{e['t_end'] / 60:.2f} min, peak {e['peak_s'] / 60:.2f} min, {e['peak_excess_db']:.1f} dB; {', '.join(e['flags']) or 'no screening flags'}")
    (out / 'envelope.json').write_text(json.dumps(env), encoding='utf-8')
    (out / 'events.json').write_text(json.dumps(candidates, indent=1), encoding='utf-8')
    grid = None
    required_radius = max(case.aircraft_radius_m, case.radius_m + 40000)
    if not events:
        r = {'status': 'no_events', 'summary': 'No broadband candidates selected for matching.', 'location_estimate': None}
    elif analyse_only:
        r = {'status': 'analysis_only', 'summary': 'Broadband candidates saved; flight matching was not requested.', 'location_estimate': None}
    elif not case.tracks_path.exists():
        r = {'status': 'awaiting_adsb', 'summary': 'Candidates saved. Fetch flight data, then rerun jets to compare them.' +
             (f' Set aircraft_radius_km to at least {required_radius / 1000:g} to cover the full search disc.'
              if case.aircraft_radius_m < required_radius else ''), 'location_estimate': None}
    else:
        # Refuse silently reusing flight caches for a changed recording window or area.
        from .pipeline import _covers
        a, b = case.window
        if not _covers(case.adsb_dir / 'source.json', a - 600, b + 600, *case.center, required_radius):
            r = {'status': 'incomplete_adsb', 'location_estimate': None,
                 'summary': f'Stored flight data do not cover the full jet search. Set aircraft_radius_km to at least {required_radius / 1000:g}, then rerun adsb --force.'}
        else:
            r, grid = match(case, env, events, step_m=step_m, timing_slack_s=timing_slack_s, log=log)
    r.update({'experimental': True, 'limitations': LIMITATIONS,
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'audio_start_utc': case.s['audio_start_utc'], 'n_events': len(events), 'n_candidates': len(candidates),
              'review_selected_ids': sorted(selected) if review_labels else None, 'required_aircraft_radius_km': required_radius / 1000, 'selected_windows_s': windows, 'selected_event_ids': ids,
              'inputs_sha256': {'audio': _sha(case.audio_path),
                                'case': _sha(case.dir / 'case.json'),
                                'events': _sha(out / 'events.json'),
                                'model_code': _sha(Path(__file__)),
                                'listening_review': _sha(review_labels) if review_labels else None,
                                'tracks': _sha(case.tracks_path) if case.tracks_path.exists() else None}})
    report(out / 'report.html', env, candidates, r, grid)
    if grid is not None:
        np.savez_compressed(out / 'grid.npz', **grid)
    elif (out / 'grid.npz').exists():
        (out / 'grid.npz').unlink()
    # This is separate from the tonal prediction and is never read by validate/export.
    p = out / 'result.json'
    p.write_text(json.dumps(r, indent=1, allow_nan=False), encoding='utf-8')
    p.with_suffix('.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest() + '\n', encoding='ascii')
    log(r['summary'])
    log(f'private report: {out / "report.html"}')
    return r
