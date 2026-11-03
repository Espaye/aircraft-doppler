"""Locate the microphone: coarse grid -> local grids around the best modes -> least-squares refinement ->
uncertainty (bootstrap and leave-one-out over events, pseudo-posterior) -> time-shift null test.

The blind prediction is written to <case>/match/result.json (with a SHA-256 stamp) before any reference
position is consulted; validate() compares it with a reference afterwards and stores that separately.
"""
import datetime as dt, hashlib, json, os, time
from pathlib import Path

import numpy as np

from . import __version__, adsb, geo, match, physics

EXPLAINED = 2.0       # per-event mean Cauchy cost below which an event counts as explained (~0.35 % rms)
REFINE_MAX = 2.5      # events above this cost at the local best are left out of the least-squares refinement
N_EFF = 10.0          # assumed independent samples per event for the pseudo-posterior (residuals are correlated)
FINE_HALF, FINE_STEP = 600.0, 25.0   # uncertainty grid around the refined point (m)


def _modes(E, N, score, n, sep):
    out = []
    for i in np.argsort(score):
        if all(np.hypot(E[i] - E[j], N[i] - N[j]) >= sep for j in out):
            out.append(int(i))
            if len(out) == n:
                break
    return out


def _best(cost):
    tot = cost.sum(0)
    mi, di = np.unravel_index(np.argmin(tot), tot.shape)
    return int(mi), int(di), float(tot[mi, di])


def _square(E0, N0, half, step):
    e = np.arange(-half, half + 1e-9, step)
    EE, NN = np.meshgrid(E0 + e, N0 + e)
    return EE.ravel(), NN.ravel()


def _wrap_shift(events, s, a, b):
    """Null-test events: each event moved by about s seconds, wrapped inside the capture window [a, b)."""
    out = []
    for e in events:
        d = a + (e['t_start'] - a + s) % (b - a) - e['t_start']
        out.append({**e, 'curves': [(t + d, f, w) for t, f, w in e['curves']],
                    't_start': e['t_start'] + d, 't_end': e['t_end'] + d})
    return out


def _latlon(enu, E, N):
    lat, lon, _ = enu.to_geodetic(np.stack([np.asarray(E, float), np.asarray(N, float), np.zeros(np.shape(E))], -1))
    return lat, lon


def _simulate(pb, k, ai, obs, dl, tc):
    """Simulated curves, residual and geometry for event k explained by candidate ai at the solution."""
    ev = pb.events[k]
    tr, traj = pb.cands[k][ai]
    c = physics.path_sound_speed(tc, obs[0, 2], np.nanmedian(traj.P[:, 2]))
    tt = np.arange(ev['t_start'] - 10, ev['t_end'] + 10, 0.5)
    gg, RR, tau = physics.doppler(traj, obs, tt - dl, c)
    sims, rms = [], []
    for t, f, w in ev['curves']:
        g, _, _ = physics.doppler(traj, obs, t - dl, c)
        lr = np.log(f) - np.log(g[0]); ok = np.isfinite(lr)
        if ok.sum() < 3:
            continue
        lf0 = np.sum(lr[ok] * w[ok]) / np.sum(w[ok])
        rms.append(float(np.sqrt(np.average((lr[ok] - lf0) ** 2, weights=w[ok]))))
        sims.append({'f0': float(np.exp(lf0)), 'f': np.round(np.exp(lf0) * gg[0], 3).tolist()})
    R0 = np.where(np.isfinite(RR[0]), RR[0], np.inf)
    i = int(np.argmin(R0))
    P, V = traj.at(tau[0, i:i + 1])
    te = np.arange(ev['t_start'] - dl - 150, ev['t_end'] - dl + 40, 4.0)
    Pp, _ = traj.at(te); okp = np.isfinite(Pp[:, 0])
    plat, plon, _ = pb.enu.to_geodetic(Pp[okp])
    clat, clon, _ = pb.enu.to_geodetic(P)
    return {'icao24': tr.icao24, 'aircraft': tr.label, 'registration': tr.meta.get('registration'),
            'type': tr.icao_type, 'desc': tr.meta.get('desc'), 'rms_pct': 100 * float(np.mean(rms)) if rms else None,
            'sim_t_utc': tt.round(2).tolist(), 'sims': sims, 'sound_speed_mps': float(c),
            'min_range_rx_utc': float(tt[i]), 'min_range_m': float(RR[0, i]),
            'height_above_mic_m': float(P[0, 2] - obs[0, 2]), 'speed_mps': float(np.linalg.norm(V[0])),
            'path': [np.round(plat, 5).tolist(), np.round(plon, 5).tolist()],
            'closest_latlon': [float(clat[0]), float(clon[0])]}


class MatchError(RuntimeError):
    """The data cannot support a location estimate (message is meant for the user)."""


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _workers(n):
    return max(1, min(8, (os.cpu_count() or 2) - 1)) if not n else int(n)


def run(case, ids=None, min_score=None, step=None, force=False, workers=None, n_modes=None, n_null=None, log=print):
    ms = case.s['match']
    ids = ids if ids is not None else ms.get('events')
    min_score = ms['min_score'] if min_score is None else min_score
    step = float(ms['coarse_step_m'] if step is None else step)
    workers = _workers(ms.get('workers') if workers is None else workers)
    n_modes = int(ms['modes'] if n_modes is None else n_modes)
    n_null = int(ms['null_tests'] if n_null is None else n_null)
    out_dir = case.match_dir; out_dir.mkdir(exist_ok=True)
    res_p = case.result_path
    if res_p.exists() and not force:
        return json.loads(res_p.read_text())
    if not case.events_path.exists():
        raise MatchError('no detected events yet: run the analysis first')
    if not case.tracks_path.exists():
        raise MatchError('no ADS-B tracks yet: fetch flight data first')
    ev_json = json.loads(case.events_path.read_text())
    events = match.load_events(ev_json, ids=ids, min_score=min_score, manual_dir=case.manual_dir)
    if ids is None:
        events = [e for e in events if e['manual'] or (e['fit'] and e['fit']['t_cpa_sigma'] < 30)]
    if len(events) < 2:
        raise MatchError(f'only {len(events)} usable aircraft event(s) (score >= {min_score}, defined closest '
                         'approach); at least 2 are needed. Lower match.min_score, draw curves by hand, or use a '
                         'longer recording.')
    tracks = adsb.load_tracks(case.tracks_path)
    if not tracks:
        raise MatchError('the ADS-B file has no tracks for this window')
    terrain = case.terrain()
    geoid_n = case.geoid_n()
    t_win = case.window
    lat0, lon0 = case.center

    def problem(evs, t_ground_c):
        return match.Problem(evs, tracks, t_ground_c, terrain, lat0, lon0, case.radius_m, case.mic_height, geoid_n,
                             delta_range=case.delay_prior)
    pb = problem(events, case.ground_temp_c)
    if not any(pb.cands):
        raise MatchError('no ADS-B aircraft was near the search area while any event was heard; check '
                         'audio_start_utc, the search centre and the flight data window')
    ids_ = [e['id'] for e in events]
    K = len(events)
    log(f'{K} events; candidate aircraft: ' + ', '.join(f'{i} {len(cs)}' for i, cs in zip(ids_, pb.cands)))

    # 1. coarse grid over the whole search disc
    t0 = time.time()
    G = pb.grid(step=step, workers=workers, progress=False)
    prof = G['cost'].sum(0).min(1)
    np.savez_compressed(out_dir / 'grid_coarse.npz', E=G['E'], N=G['N'], cost=G['cost'], deltas=G['deltas'], step=step)
    log(f"coarse grid: {len(G['E'])} positions x {len(G['deltas'])} offsets, {time.time() - t0:.0f} s")

    # 2. local grids around the best distinct modes
    sols = []
    for mi in _modes(G['E'], G['N'], prof, n_modes, 3 * step):
        d0 = G['deltas'][int(np.argmin(G['cost'].sum(0)[mi]))]
        dls = np.arange(max(pb.delta_range[0], d0 - 3), min(pb.delta_range[1], d0 + 3) + 1e-9, 0.25)
        E, N = _square(G['E'][mi], G['N'][mi], 1.5 * step, step / 8)
        L = pb.costs(E, N, dls, workers=workers); L['step'] = step / 8
        li, di, tot = _best(L['cost'])
        per = L['cost'][:, li, di]
        assoc = [int(L['arg'][k, li, di]) if per[k] < REFINE_MAX else -1 for k in range(K)]
        sols.append({'L': L, 'li': li, 'di': di, 'tot': tot, 'per': per, 'assoc': assoc})
        log(f"  mode E {L['E'][li]:.0f} N {L['N'][li]:.0f} delta {dls[di]:.2f}: total {tot:.2f}, "
            f"{int((per < EXPLAINED).sum())}/{K} explained")
    sols.sort(key=lambda s: s['tot'])
    S = sols[0]; L = S['L']
    Eg, Ng, dg = float(L['E'][S['li']]), float(L['N'][S['li']]), float(L['deltas'][S['di']])

    # 3. least-squares refinement of E, N, delta and ground temperature with the associations fixed
    sig = [np.nan] * 3
    if sum(a >= 0 for a in S['assoc']) >= 2:
        r = pb.refine(Eg, Ng, dg, S['assoc'], fit_temp=True)
        E, N, dl, tc = (float(v) for v in r.x)
        try:
            s2 = float(np.median(r.fun ** 2) / 0.4549) * 4.0     # robust scale, x4 for residual correlation
            cov = np.linalg.pinv(r.jac.T @ r.jac) * s2
            sig = np.sqrt(np.clip(np.diag(cov)[:3], 0, None)).tolist()
        except Exception:
            pass
    else:
        E, N, dl, tc = Eg, Ng, dg, case.ground_temp_c
    # 4. uncertainty. Fine grid (25 m) around the refined point at the fitted temperature: event bootstrap and
    # pseudo-posterior. Local grid (100 m, wider): leave-one-out. Coarse grid: mode stability.
    t_save = pb.t_ground_c; pb.t_ground_c = tc
    F = pb.costs([E], [N], [dl], workers=1)
    dls_f = np.arange(max(pb.delta_range[0], dl - 1.5), min(pb.delta_range[1], dl + 1.5) + 1e-9, 0.125)
    U = pb.costs(*_square(E, N, FINE_HALF, FINE_STEP), dls_f, workers=workers)
    pb.t_ground_c = t_save
    per, arg = F['cost'][:, 0, 0], F['arg'][:, 0, 0]
    log(f'refined E {E:.0f} N {N:.0f} delta {dl:.2f} s T {tc:.1f} C: {int((per < EXPLAINED).sum())}/{K} explained')
    edge = np.maximum(np.abs(U['E'] - E), np.abs(U['N'] - N)) > FINE_HALF - FINE_STEP / 2
    # bootstrap units: events explained by the same aircraft are resampled together (they are not independent)
    unit = [pb.cands[k][arg[k]][0].icao24 if arg[k] >= 0 and per[k] < EXPLAINED else ids_[k] for k in range(K)]
    members = [np.flatnonzero([u == v for u in unit]) for v in dict.fromkeys(unit)]
    rng = np.random.default_rng(0)
    boots, stab, edge_hits = [], 0, 0
    for _ in range(500):
        s = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        mi, di, _ = _best(U['cost'][s])
        boots.append((U['E'][mi], U['N'][mi], U['deltas'][di])); edge_hits += edge[mi]
        mc, _, _ = _best(G['cost'][s])
        stab += np.hypot(G['E'][mc] - Eg, G['N'][mc] - Ng) < 3 * step
    boots = np.array(boots); br = np.hypot(boots[:, 0] - E, boots[:, 1] - N)
    loo = []
    for k in range(K):
        keep = np.arange(K) != k
        mi, di, _ = _best(U['cost'][keep]); R, gname = U, 'fine'
        if edge[mi]:                                      # moved beyond the fine grid: use the wider local grid
            (mi, di, _), R, gname = _best(L['cost'][keep]), L, 'local'
        mc, _, _ = _best(G['cost'][keep])
        loo.append({'left_out': ids_[k], 'E_m': float(R['E'][mi]), 'N_m': float(R['N'][mi]), 'grid': gname,
                    'delta_s': float(R['deltas'][di]), 'shift_m': float(np.hypot(R['E'][mi] - E, R['N'][mi] - N)),
                    'coarse_mode_kept': bool(np.hypot(G['E'][mc] - Eg, G['N'][mc] - Ng) < 3 * step)})
    tot = U['cost'].sum(0)
    pM = np.exp(-0.5 * N_EFF * (tot - tot.min())).sum(1); pM /= pM.sum()
    a68 = float(match.credible_mask(pM, 0.68).sum() * (FINE_STEP / 1000) ** 2)
    m95 = match.credible_mask(pM, 0.95)
    np.savez_compressed(out_dir / 'grid_local.npz', E=U['E'], N=U['N'], p=pM, step=FINE_STEP, tot=tot.min(1))

    # 5. null test: same events, time-shifted against the ADS-B (should not find a comparable solution)
    real_coarse = _best(G['cost'])
    nulls = []
    for j in range(n_null):
        s = (j + 1) * (t_win[1] - t_win[0]) / (n_null + 1)
        pbn = problem(_wrap_shift(events, s, *t_win), case.ground_temp_c)
        Gn = pbn.grid(step=step, workers=workers, progress=False)
        mi, di, tn = _best(Gn['cost'])
        nulls.append({'shift_s': round(s, 1), 'total': tn, 'n_explained': int((Gn['cost'][:, mi, di] < EXPLAINED).sum()),
                      'E_m': float(Gn['E'][mi]), 'N_m': float(Gn['N'][mi])})
        log(f'  null shift {s:.0f} s: best total {tn:.2f}, {nulls[-1]["n_explained"]} explained')

    # 6. per-event details at the solution
    obs = pb.observer(np.array([E]), np.array([N]))
    per_event = []
    for k, ev in enumerate(events):
        d = {'id': ev['id'], 'cost': float(per[k]), 'explained': bool(per[k] < EXPLAINED), 'manual': ev['manual'],
             'used_in_refinement': S['assoc'][k] >= 0, 't_start_utc': ev['t_start'], 't_end_utc': ev['t_end'],
             'n_candidates': len(pb.cands[k])}
        if arg[k] >= 0:
            d.update(_simulate(pb, k, int(arg[k]), obs, dl, tc))
        per_event.append(d)
    lat, lon = _latlon(pb.enu, [E], [N])
    blat, blon = _latlon(pb.enu, boots[:200, 0], boots[:200, 1])
    result = {
        'case': case.name, 'prediction_created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'blind': 'computed without any reference position; prior = search centre '
                 f'({lat0}, {lon0}) +/- {case.radius_m / 1000:.0f} km',
        'inputs': {'adoppler_version': __version__, 'events_sha256': _sha(case.events_path),
                   'tracks_sha256': _sha(case.tracks_path), 'geoid_undulation_m': geoid_n,
                   'terrain': case.s['terrain'], 'manual_events': [e['id'] for e in events if e['manual']]},
        'best': {'lat': float(lat[0]), 'lon': float(lon[0]), 'E_m': E, 'N_m': N, 'delta_s': dl, 'ground_temp_c': tc,
                 'ground_elev_msl_m': float(terrain.elevation(lat[0], lon[0])),
                 'n_explained': int((per < EXPLAINED).sum()), 'n_events': K, 'grid_E_m': Eg, 'grid_N_m': Ng,
                 'grid_delta_s': dg, 'local_total': S['tot']},
        'alternatives': [{'E_m': float(s['L']['E'][s['li']]), 'N_m': float(s['L']['N'][s['li']]),
                          'delta_s': float(s['L']['deltas'][s['di']]), 'local_total': s['tot'],
                          'n_explained': int((s['per'] < EXPLAINED).sum()),
                          'latlon': [float(v[0]) for v in _latlon(pb.enu, [s['L']['E'][s['li']]], [s['L']['N'][s['li']]])]}
                         for s in sols[1:]],
        'uncertainty': {
            'bootstrap_median_m': float(np.median(br)), 'bootstrap_68_m': float(np.percentile(br, 68)),
            'bootstrap_90_m': float(np.percentile(br, 90)), 'bootstrap_delta_sd_s': float(np.std(boots[:, 2])),
            'mode_stability': stab / 500, 'bootstrap_edge_fraction': edge_hits / 500, 'bootstrap_units': len(members), 'loo': loo,
            'posterior_area68_km2': a68, 'posterior_area95_km2': float(m95.sum() * (FINE_STEP / 1000) ** 2),
            'posterior_95_touches_edge': bool((m95 & edge).any()), 'n_eff_per_event': N_EFF,
            'jacobian_sigma_E_m': sig[0], 'jacobian_sigma_N_m': sig[1], 'jacobian_sigma_delta_s': sig[2],
            'bootstrap_points': [np.round(blat, 5).tolist(), np.round(blon, 5).tolist()],
            'note': 'Approximate. Residuals are correlated; wind, refraction and ADS-B position/latency errors are not '
                    f'modelled. The bootstrap/leave-one-out spread over events is the more honest measure. Grid steps: '
                    f"bootstrap/posterior/leave-one-out {FINE_STEP:.0f} m within +/-{FINE_HALF:.0f} m ({L['step']:.0f} m beyond)."},
        'null_test': {'real_coarse_total': real_coarse[2],
                      'real_coarse_n_explained': int((G['cost'][:, real_coarse[0], real_coarse[1]] < EXPLAINED).sum()),
                      'shifted': nulls},
        'events': per_event,
        'settings': {'coarse_step_m': step, 'local_step_m': float(L['step']), 'fine_step_m': FINE_STEP, 'delta_range_s': list(pb.delta_range),
                     'sigma_log': pb.sigma, 'cap': pb.cap, 'explained_below': EXPLAINED, 'min_score': min_score,
                     'mic_height_agl_m': case.mic_height, 'search_radius_m': case.radius_m,
                     'ground_temp_prior_c': case.ground_temp_c, 'null_tests': n_null, 'modes': n_modes},
        'enu_origin': [pb.enu.lat0, pb.enu.lon0, pb.enu.h0],
    }
    result['assessment'] = assess(result)
    log(f"assessment: {result['assessment']['verdict']}" +
        ''.join(f"\n  - {m['message']}" for m in result['assessment']['flags']))
    txt = json.dumps(result, indent=1)
    res_p.write_text(txt)
    (out_dir / 'result.sha256').write_text(f"{hashlib.sha256(txt.encode()).hexdigest()}  result.json  "
                                          f"{result['prediction_created_utc']}\n")
    return result


def validate(case, lat, lon, ref_source):
    """Compare the stored blind prediction with an independently obtained reference position (after the fact)."""
    out_dir = case.match_dir
    if not case.result_path.exists():
        raise MatchError('no prediction yet: run the match first')
    txt = (out_dir / 'result.json').read_text()
    res = json.loads(txt)
    stamp = (out_dir / 'result.sha256').read_text().split()[0]
    enu = geo.ENU(*res['enu_origin'])
    p = enu.from_geodetic(np.array([lat]), np.array([lon]), np.array([res['enu_origin'][2]]))[0]
    b = res['best']
    v = {'reference': {'lat': lat, 'lon': lon, 'source': ref_source,
                       'entered_utc': dt.datetime.now(dt.timezone.utc).isoformat()},
         'prediction_created_utc': res['prediction_created_utc'],
         'prediction_unchanged': hashlib.sha256(txt.encode()).hexdigest() == stamp,
         'ref_E_m': float(p[0]), 'ref_N_m': float(p[1]),
         'error_m': float(np.hypot(p[0] - b['E_m'], p[1] - b['N_m'])),
         'error_E_m': float(b['E_m'] - p[0]), 'error_N_m': float(b['N_m'] - p[1]),
         'bootstrap_90_m': res['uncertainty']['bootstrap_90_m'],
         'alternative_errors_m': [float(np.hypot(p[0] - a['E_m'], p[1] - a['N_m'])) for a in res['alternatives']],
         'loo_errors_m': {x['left_out']: float(np.hypot(p[0] - x['E_m'], p[1] - x['N_m'])) for x in res['uncertainty']['loo']}}
    v['within_bootstrap_90'] = v['error_m'] <= v['bootstrap_90_m']
    gl = np.load(out_dir / 'grid_local.npz')
    j = int(np.argmin(np.hypot(gl['E'] - p[0], gl['N'] - p[1])))
    if np.hypot(gl['E'][j] - p[0], gl['N'][j] - p[1]) <= float(gl['step']):
        v['ref_posterior_level'] = float(gl['p'][gl['p'] >= gl['p'][j]].sum())
    else:
        v['ref_posterior_level'] = 'outside local grid'
    gc = np.load(out_dir / 'grid_coarse.npz')
    prof = gc['cost'].sum(0).min(1)
    jc = int(np.argmin(np.hypot(gc['E'] - p[0], gc['N'] - p[1])))
    v['ref_coarse_cell_rank'] = int((prof < prof[jc]).sum()) + 1
    v['coarse_cells'] = int(len(prof))
    (out_dir / 'validation.json').write_text(json.dumps(v, indent=1))
    return v


# thresholds of the assessment (sums of per-event capped costs; one explained event lowers the total by ~2-3)
NULL_MARGIN = 3.0      # null tests within this of the real coarse total: the match may be chance
ALT_MARGIN = 2.0       # a distinct alternative within this of the best local total: ambiguous
MODE_STABILITY = 0.8   # coarse mode kept in fewer bootstrap resamples than this: ambiguous


def assess(r):
    """Plain-language verdict on a result: 'ok', 'weak', 'ambiguous' or 'failed', with the reasons."""
    b, u, nt = r['best'], r['uncertainty'], r['null_test']
    flags = []

    def flag(level, msg):
        flags.append({'level': level, 'message': msg})
    if b['n_explained'] < 2:
        flag('failed', f"only {b['n_explained']} of {b['n_events']} events are explained by an ADS-B aircraft at the "
                       'best position; two or more are needed for a location')
    elif b['n_explained'] == 2:
        flag('weak', 'only two events are explained; the position rests on two curves and a common time offset')
    if nt['shifted']:
        margin = min(x['total'] for x in nt['shifted']) - nt['real_coarse_total']
        n_null = max(x['n_explained'] for x in nt['shifted'])
        # the nulls are only searched on the coarse grid, which may explain no event at all; compare their count
        # with the better of the coarse and final counts so a null explaining nothing is not a competitor
        n_real = max(nt['real_coarse_n_explained'], b['n_explained'])
        if margin < NULL_MARGIN and n_null >= n_real - 1:
            flag('ambiguous', f'time-shifted null tests (events moved against the flight data) fit almost as well '
                              f'({n_null} events explained): the match may be coincidental')
    if u['mode_stability'] < MODE_STABILITY:
        few = u['bootstrap_units'] < 5
        flag('weak' if few else 'ambiguous',
             f"the coarse best area changes in {100 * (1 - u['mode_stability']):.0f}% of event-bootstrap resamples"
             + (f" (only {u['bootstrap_units']} independent aircraft)" if few else ''))
    moved = [x['left_out'] for x in u['loo'] if not x['coarse_mode_kept']]
    if moved:
        flag('weak', 'leaving out ' + ', '.join(moved) + ' moves the solution to another area: it rests on '
                     + ('that event' if len(moved) == 1 else 'those events'))
    for a in r['alternatives']:
        if a['local_total'] - b['local_total'] < ALT_MARGIN:
            d = np.hypot(a['E_m'] - b['E_m'], a['N_m'] - b['N_m'])
            flag('ambiguous', f"a distinct solution {d / 1000:.1f} km away fits almost as well "
                              f"({a['n_explained']} events explained)")
            break
    if u['bootstrap_edge_fraction'] > 0.1 or u['posterior_95_touches_edge']:
        flag('weak', 'the uncertainty reaches the edge of the fine grid, so the quoted spread is a lower bound')
    if np.hypot(b['E_m'], b['N_m']) > 0.9 * r['settings']['search_radius_m']:
        flag('weak', 'the estimate lies at the edge of the search disc; increase search_radius_km')
    order = ['ok', 'weak', 'ambiguous', 'failed']
    verdict = max((f['level'] for f in flags), key=order.index, default='ok')
    summary = {'ok': 'consistent solution; uncertainty from the event bootstrap',
               'weak': 'a solution exists but is weakly constrained; treat it as preliminary',
               'ambiguous': 'several positions explain the data about equally well; no reliable location',
               'failed': 'the data do not support a location estimate'}[verdict]
    return {'verdict': verdict, 'summary': summary, 'flags': flags,
            'radius_68_m': u['bootstrap_68_m'], 'radius_90_m': u['bootstrap_90_m'],
            'note': 'Synthetic tests only show internal consistency; accuracy on real recordings comes from '
                    'independent validation.'}
