"""The four stages of a case, shared by the command line and the browser UI.

    analyse   audio -> spectrogram, tonal enhancement, Doppler events          (analysis/events.json)
    adsb      historical ADS-B for the recording window and area               (adsb/tracks.json)
    match     events x tracks -> microphone position with uncertainty          (match/result.json)
    validate  compare the stored blind result with a reference position        (match/validation.json)

Every stage takes a log callback and raises a StageError with a message meant for the user when it cannot run.
"""
import datetime as dt, json, shutil
from pathlib import Path

from . import adsb, case as case_mod, detect, locate

STAGES = ('analyse', 'adsb', 'match')
ODBL = ('ADS-B data from adsb.lol (https://adsb.lol), licensed under the Open Database License (ODbL) 1.0; '
        'attribute adsb.lol when you share derived data.')


class StageError(RuntimeError):
    pass


def _clear(paths):
    for p in paths:
        p = Path(p)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


def analyse(case, force=False, log=print):
    info = case.audio_info()
    if info['duration_s'] < 60:
        raise StageError(f"the recording is {info['duration_s']:.0f} s long; at least a few minutes are needed "
                         'to see complete Doppler curves')
    if info['samplerate'] < 4000:
        raise StageError(f"sample rate {info['samplerate']} Hz is too low (need >= 4 kHz)")
    log(f"{case.audio_path.name}: {info['duration_s'] / 60:.1f} min, {info['samplerate']} Hz, "
        f"{info['channels']} channel(s)")
    if force:
        _clear([case.analysis_dir / n for n in ('events.json', 'tracks.json', 'peaks.npy')])
        _clear(case.analysis_dir.glob('spec*')); _clear(case.analysis_dir.glob('tonal*'))
    out = detect.run(case.audio_path, case.analysis_dir, case.t0, force=force, log=log)
    if not out['events']:
        log('no Doppler events found: is there aircraft sound in the recording, and is it more than a hum?')
    return out


def _days(t_from, t_to):
    d = dt.datetime.fromtimestamp(t_from, dt.timezone.utc).date()
    last = dt.datetime.fromtimestamp(t_to, dt.timezone.utc).date()
    while d <= last:
        yield d
        d += dt.timedelta(days=1)


def fetch_adsb(case, force=False, source=None, log=print):
    """Tracks for the recording window +/- 10 min within aircraft_radius_km of the search centre."""
    tp = case.tracks_path
    if tp.exists() and not force:
        log(f'ADS-B tracks already present ({tp.name}); use force to fetch again')
        return adsb.load_tracks(tp)
    src = source or case.s['adsb_source']
    t_from, t_to = case.window
    t_from -= 600; t_to += 600
    lat, lon = case.center
    radius = case.aircraft_radius_m
    geoid_n = case.geoid_n()
    case.adsb_dir.mkdir(parents=True, exist_ok=True)
    tracks, info = None, {}
    if src == 'file':
        p = Path(case.s['adsb_file'])
        p = p if p.is_absolute() else case.dir / p
        if not p.exists():
            raise StageError(f'adsb_file {p} not found')
        try:
            tracks = adsb.load_file(p, geoid_n)
        except (ValueError, KeyError) as e:
            raise StageError(str(e))
        info = {'source': f'local file {p.name}'}
    if tracks is None and src in ('auto', 'adsb.lol'):
        days = list(_days(t_from, t_to))
        ready = []
        for day in days:
            arch = case.adsb_dir / f'archive_{day:%Y%m%d}'
            try:
                ok = (arch / '_complete.json').exists() or bool(adsb.release_assets(adsb.release_tag(day), day))
            except Exception as e:
                ok = False
                log(f'adsb.lol archive check failed: {e}')
            ready.append(ok)
        if all(ready):
            tracks, infos = [], []
            for day in days:
                arch = case.adsb_dir / f'archive_{day:%Y%m%d}'
                log(f'adsb.lol archive {adsb.release_tag(day)}: streaming (~3 GB, once), keeping traces within '
                    f'{radius / 1000:.0f} km')
                infos.append(adsb.stream_archive(day, t_from, t_to, arch, lat, lon, radius, log=log))
                tracks += adsb.load_archive_dir(arch, geoid_n)
            info = {'source': 'adsb.lol globe_history archive (raw readsb traces)', 'archives': infos,
                    'license': ODBL}
        elif src == 'adsb.lol':
            raise StageError('the adsb.lol daily archive for this date is not published yet (normally a few hours '
                             'after midnight UTC); try again later or use adsb_source "adsb.aero"')
        else:
            log('adsb.lol archive not available for this date; trying adsb.aero')
    if tracks is None:
        cache = case.adsb_dir / 'aero_query.json'
        try:
            d = adsb.aero_query(t_from, t_to, lat, lon, radius, cache=None if force else cache)
        except Exception as e:
            raise StageError(f'adsb.aero query failed: {e}')
        if not d['flights']:
            raise StageError('adsb.aero returned no flights for this window (not published yet, or no traffic); '
                             'try later, or supply your own tracks with adsb_source "file"')
        cache.write_text(json.dumps(d))
        tracks = [adsb.parse_aero(f) for f in d['flights']]
        info = {'source': 'adsb.aero /query (simplified paths)', 'n_flights': len(d['flights'])}
    tracks = [t for t in tracks if len(t.t) and t.t.max() >= t_from and t.t.min() <= t_to]
    if not tracks:
        raise StageError('no aircraft tracks in the recording window and area')
    adsb.save_tracks(tracks, tp)
    (case.adsb_dir / 'source.json').write_text(json.dumps(
        {**info, 'n_tracks': len(tracks), 'window_utc': [t_from, t_to], 'radius_m': radius,
         'retrieved_utc': dt.datetime.now(dt.timezone.utc).isoformat()}, indent=1, default=str))
    log(f'{len(tracks)} aircraft tracks saved')
    return tracks


def match(case, force=False, log=print, **kw):
    if force:
        _clear([case.result_path, case.result_path.with_suffix('.sha256'), case.validation_path,
                case.match_dir / 'grid_coarse.npz', case.match_dir / 'grid_local.npz'])
    try:
        return locate.run(case, force=force, log=log, **kw)
    except locate.MatchError as e:
        raise StageError(str(e))


def validate(case, lat, lon, ref_source, log=print):
    try:
        v = locate.validate(case, lat, lon, ref_source)
    except locate.MatchError as e:
        raise StageError(str(e))
    log(f"error {v['error_m']:.0f} m ({v['error_E_m']:+.0f} E, {v['error_N_m']:+.0f} N)")
    return v


def validate_synthetic(case, log=print):
    """For synthetic cases: compare with the hidden truth written by `adoppler synth`."""
    p = case.dir / 'truth.json'
    if not p.exists():
        raise StageError('this case has no truth.json (only synthetic cases do)')
    t = json.loads(p.read_text())
    return validate(case, t['mic_lat'], t['mic_lon'], 'synthetic truth (truth.json)', log=log)


def run_all(case, force=False, log=print):
    analyse(case, force=force, log=log)
    fetch_adsb(case, force=force, log=log)
    return match(case, force=force, log=log)


def new_case(path, audio, start_utc, center, radius_km=None, **settings):
    s = {'name': settings.pop('name', None) or Path(path).name, 'audio': str(audio), 'audio_start_utc': start_utc,
         'search_center': [float(center[0]), float(center[1])]}
    if radius_km is not None:
        s['search_radius_km'] = float(radius_km)
    s.update({k: v for k, v in settings.items() if v is not None})
    return case_mod.create(path, s)

