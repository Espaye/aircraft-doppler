"""Locate the microphone by matching observed Doppler curves against curves simulated from ADS-B tracks.

Unknowns: microphone position (E, N; height = terrain + mic height), a common time offset delta
(acoustic time = recording-clock UTC - delta), one emitted frequency per curve (eliminated in closed form, in log space),
and optionally the ground temperature (sound speed). Each event may be explained by any candidate aircraft;
the best one is chosen per hypothesis, and a capped robust cost keeps an unexplained event from dominating.
"""
import json
import numpy as np
from scipy.optimize import least_squares

from . import config, geo, physics


# ---------------------------------------------------------------- observed curves

def load_events(events_json, ids=None, min_score=0.0, labels=('aircraft',), max_tracks=4, max_pts=80, manual_dir=None):
    """Observed curves as lists of (t_utc, f, w) per track, thinned for speed. Uses manual edits if present."""
    d = json.loads(open(events_json).read()) if isinstance(events_json, str) else events_json
    t0 = d['wav_t0_utc_unix']
    out = []
    for e in d['events']:
        if ids is not None:
            if e['id'] not in ids:
                continue
        elif e['label'] not in labels or e['score'] < min_score:
            continue
        trs = sorted(e['tracks'], key=lambda x: -len(x['t']) * np.median(x['snr']))[:max_tracks]
        man = (manual_dir / f"{e['id']}.json") if manual_dir else None
        if man is not None and man.exists():          # manual curves (report editor), in capture seconds
            mj = json.loads(man.read_text())
            mt = [{'t': c_['t'], 'f': c_['f'], 'snr': [10.0] * len(c_['t'])} for c_ in mj['curves']]
            trs = mt if mj.get('mode', 'replace') == 'replace' else mt + trs
        cur = []
        for x in trs:
            t = np.asarray(x['t'], float) + t0; f = np.asarray(x['f'], float); w = np.clip(np.asarray(x['snr']) / 10, 0.3, 3)
            k = max(1, len(t) // max_pts)
            cur.append((t[::k], f[::k], w[::k]))
        out.append({'id': e['id'], 'curves': cur, 't_start': e['t_start'] + t0, 't_end': e['t_end'] + t0,
                    'fit': e.get('fit'), 'manual': bool(man is not None and man.exists())})
    return out


# ---------------------------------------------------------------- candidates

def candidate_aircraft(tracks, ev, enu, max_horiz_m, geoid_n=0.0, max_alt_m=13000, delta_range=(-2, 10),
                       speed_check=True):
    """Aircraft airborne near the search region while the event's sound could have been emitted."""
    a = ev['t_start'] - delta_range[1] - 60; b = ev['t_end'] - delta_range[0] + 5
    out = []
    for tr in tracks:
        m = (tr.t >= a) & (tr.t <= b) & (tr.alt_kind != 'ground')
        if m.sum() < 1:
            continue
        p = enu.from_geodetic(tr.lat[m], tr.lon[m], geo.msl_to_ellipsoid(tr.alt_msl_m[m], geoid_n))
        if not np.isfinite(p).all(1).any():
            continue
        if np.nanmin(np.hypot(p[:, 0], p[:, 1])) >= max_horiz_m or np.nanmin(p[:, 2]) >= max_alt_m:
            continue
        fit = ev.get('fit') or {}
        if speed_check and fit.get('t_cpa_sigma', 99) < 5 and fit.get('d', 1e9) < 30000:
            gs = np.asarray(tr.meta.get('gs_kt', []), float)
            if len(gs) == len(tr.t):
                g = gs[m]; g = g[np.isfinite(g)] * config.KT
                if len(g) and not (0.55 * fit['v'] <= np.median(g) <= 1.8 * fit['v'] + 15):
                    continue
        out.append(tr)
    return out


# ---------------------------------------------------------------- grid search

class Problem:
    def __init__(self, events, tracks, t_ground_c, terrain, lat0, lon0, radius_m, mic_height=2.0, geoid_n=0.0,
                 delta_range=(-2.0, 10.0), sigma_log=0.0015, cap=4.0):
        self.terrain = terrain
        self.radius, self.mic_height, self.geoid_n = float(radius_m), float(mic_height), float(geoid_n)
        h0 = float(self.terrain.elevation(lat0, lon0)) + self.geoid_n
        if not np.isfinite(h0):
            raise ValueError('terrain has no elevation at the search centre')
        self.enu = geo.ENU(lat0, lon0, h0)
        self.events, self.t_ground_c = events, t_ground_c
        self.delta_range, self.sigma, self.cap = delta_range, sigma_log, cap
        self.cands = []
        for ev in events:
            cs = []
            for tr in candidate_aircraft(tracks, ev, self.enu, self.radius + 15000, self.geoid_n,
                                         delta_range=delta_range):
                try:
                    cs.append((tr, physics.Trajectory(tr, self.enu, ev['t_start'] - delta_range[1] - 120,
                                                      ev['t_end'] - delta_range[0] + 10, geoid_n=self.geoid_n)))
                except ValueError:
                    pass
            self.cands.append(cs)

    def __getstate__(self):          # worker processes need trajectories, not the DEM
        d = self.__dict__.copy(); d['terrain'] = None
        return d

    def observer(self, E, N):
        """ENU of observers at (E, N) on the terrain surface + mic height."""
        E, N = np.asarray(E, float), np.asarray(N, float)
        lat, lon, _ = self.enu.to_geodetic(np.stack([E, N, np.zeros_like(E)], -1))
        h = self.terrain.elevation(lat, lon) + self.mic_height + self.geoid_n
        return self.enu.from_geodetic(lat, lon, h)

    def c_for(self, traj, obs):
        z_src = np.nanmedian(traj.P[:, 2])
        return physics.path_sound_speed(self.t_ground_c, np.median(obs[:, 2]), z_src)

    def event_cost(self, k, obs, deltas, t_step=1.0):
        """Normalised robust cost (M, D) of event k for observers obs and offsets deltas, best aircraft index."""
        ev = self.events[k]
        M, D = len(obs), len(deltas)
        best = np.full((M, D), self.cap); arg = np.full((M, D), -1)
        tg = np.arange(ev['t_start'] - max(deltas) - 1, ev['t_end'] - min(deltas) + 1, t_step)
        for ai, (tr, traj) in enumerate(self.cands[k]):
            c = self.c_for(traj, obs)
            g, R, _ = physics.doppler(traj, obs, tg, c)
            lg = np.log(g)                                     # (M, G)
            J = np.zeros((M, D)); W = 0.0
            for (t, f, w) in ev['curves']:
                lf = np.log(f)
                for di, dl in enumerate(deltas):
                    x = (t - dl - tg[0]) / t_step
                    i = np.clip(np.floor(x).astype(int), 0, len(tg) - 2); u = x - i
                    lm = lg[:, i] * (1 - u) + lg[:, i + 1] * u     # (M, K)
                    r = lf[None, :] - lm
                    r -= (r * w).sum(1, keepdims=True) / w.sum()   # eliminate f0 (weighted mean in log space)
                    rho = np.log1p((r / self.sigma) ** 2)          # Cauchy loss
                    J[:, di] += (rho * w).sum(1)
                W += w.sum()
            J /= W
            J = np.where(np.isfinite(J), J, self.cap)
            better = J < best
            best = np.where(better, J, best); arg = np.where(better, ai, arg)
        return best, arg

    def grid(self, half=None, step=400.0, deltas=None, progress=True, workers=5):
        half = self.radius if half is None else half
        deltas = np.arange(self.delta_range[0], self.delta_range[1] + 1e-9, 0.5) if deltas is None else deltas
        e = np.arange(-half, half + 1e-9, step)
        EE, NN = np.meshgrid(e, e)
        inside = np.hypot(EE, NN) <= half
        G = self.costs(EE[inside], NN[inside], deltas, progress=progress, workers=workers)
        G['step'] = step
        return G

    def costs(self, E, N, deltas, progress=False, workers=5):
        """Per-event capped costs (K, M, D) and best-aircraft indices for observer positions E, N and offsets."""
        E, N, deltas = np.asarray(E, float), np.asarray(N, float), np.asarray(deltas, float)
        obs = self.observer(E, N)
        ok = np.isfinite(obs[:, 2])
        costs, args = [None] * len(self.events), [None] * len(self.events)
        if workers > 1 and len(self.events) > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(workers) as ex:
                futs = {ex.submit(_event_cost_job, self, k, obs, deltas): k for k in range(len(self.events))}
                for fu in futs:
                    k = futs[fu]; costs[k], args[k] = fu.result()
                    if progress:
                        print(f'  event {self.events[k]["id"]}: {len(self.cands[k])} candidates done', flush=True)
        else:
            for k in range(len(self.events)):
                costs[k], args[k] = self.event_cost(k, obs, deltas)
                if progress:
                    print(f'  event {self.events[k]["id"]}: {len(self.cands[k])} candidates done', flush=True)
        for k in range(len(self.events)):
            costs[k][~ok] = self.cap
        return {'E': E, 'N': N, 'obs': obs, 'deltas': deltas, 'cost': np.array(costs).reshape(-1, len(E), len(deltas)),
                'arg': np.array(args).reshape(-1, len(E), len(deltas))}

    # ------------------------------------------------------------ refinement

    def residuals(self, x, assoc, fit_temp=False):
        E, N, dl = x[:3]
        tc = x[3] if fit_temp else self.t_ground_c
        obs = self.observer(np.array([E]), np.array([N]))
        out = []
        for k, ai in enumerate(assoc):
            if ai < 0:
                continue
            tr, traj = self.cands[k][ai]
            c = physics.path_sound_speed(tc, obs[0, 2], np.nanmedian(traj.P[:, 2]))
            for (t, f, w) in self.events[k]['curves']:
                g, _, _ = physics.doppler(traj, obs, t - dl, c)
                r = np.log(f) - np.log(g[0])
                r -= (r * w).sum() / w.sum()
                out.append(np.where(np.isfinite(r), r, 0.05) * np.sqrt(w))
        if fit_temp:
            out.append(np.array([(tc - self.t_ground_c) / 5.0 * self.sigma * 3]))  # weak prior, +/- 5 C
        return np.concatenate(out) / self.sigma

    def refine(self, E0, N0, d0, assoc, fit_temp=True):
        x0 = [E0, N0, d0] + ([self.t_ground_c] if fit_temp else [])
        lb = [-self.radius * 1.2] * 2 + [self.delta_range[0] - 2] + ([-15] if fit_temp else [])
        ub = [self.radius * 1.2] * 2 + [self.delta_range[1] + 2] + ([40] if fit_temp else [])
        r = least_squares(self.residuals, x0, args=(assoc, fit_temp), loss='cauchy', f_scale=1.0,
                          x_scale=[100, 100, 0.5] + ([2] if fit_temp else []), bounds=(lb, ub), diff_step=1e-4)
        return r


def _event_cost_job(pb, k, obs, deltas):
    return pb.event_cost(k, obs, deltas)


def posterior(G, temperature=None, n_eff=None):
    """Pseudo-posterior over (position, delta) from the capped costs; temperature calibrates for correlated
    residuals and model error. Returns normalised probabilities (M, D) and the marginal over position (M,)."""
    tot = G['cost'].sum(0)                            # (M, D) mean Cauchy loss summed over events
    n_eff = n_eff or 15.0                             # independent samples per event (curve ~ tens of s, ~2 s corr.)
    temperature = temperature or 1.0
    ll = -0.5 * n_eff * (tot - tot.min()) / temperature
    p = np.exp(ll); p /= p.sum()
    return p, p.sum(1)


def credible_mask(pm, level):
    o = np.argsort(-pm); c = np.cumsum(pm[o])
    m = np.zeros(len(pm), bool); m[o[:np.searchsorted(c, level) + 1]] = True
    return m
