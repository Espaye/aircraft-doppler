"""Command line: `adoppler <command> ...` (or `python -m adoppler`). Run `adoppler -h` for the list."""
import argparse, json, sys
from pathlib import Path

from . import __version__, case as case_mod, pipeline


def _case(a):
    return case_mod.Case(a.case)


def _center(s):
    try:
        lat, lon = (float(x) for x in s.replace(' ', '').split(','))
        return lat, lon
    except ValueError:
        raise argparse.ArgumentTypeError('use LAT,LON in decimal degrees, e.g. 52.08,5.83')


def cmd_new(a):
    audio = Path(a.audio).resolve()
    d = Path(a.case).resolve()
    rel = audio.relative_to(d) if audio.is_relative_to(d) else audio
    c = pipeline.new_case(d, rel, a.start, a.center, a.radius, name=a.name, ground_temp_c=a.temp, terrain=a.terrain,
                          ground_elevation_m=a.elevation, adsb_source='file' if a.adsb_file else None,
                          adsb_file=a.adsb_file)
    info = c.audio_info()
    print(f"created {c.dir / case_mod.CASE_FILE} ({info['duration_s'] / 60:.1f} min of audio)")


def cmd_synth(a):
    from . import synth
    synth.make_case(a.case, a.scene, seed=a.seed)


def cmd_analyse(a):
    pipeline.analyse(_case(a), force=a.force)


def cmd_adsb(a):
    pipeline.fetch_adsb(_case(a), force=a.force, source=a.source)


def cmd_match(a):
    c = _case(a)
    r = pipeline.match(c, force=a.force, workers=a.workers, ids=a.events)
    _print_result(r)


def cmd_run(a):
    c = _case(a)
    pipeline.analyse(c, force=a.force)
    pipeline.fetch_adsb(c, force=a.force)
    _print_result(pipeline.match(c, force=a.force, workers=a.workers))


def _radius(m):
    return f'{m:.0f} m' if m >= 25 else 'one 25 m grid cell'


def _print_result(r):
    b, u, s = r['best'], r['uncertainty'], r['assessment']
    print(f"\nestimate {b['lat']:.5f}, {b['lon']:.5f}  ({b['n_explained']}/{b['n_events']} events explained)")
    print(f"uncertainty (event bootstrap): 68% within {_radius(u['bootstrap_68_m'])}, 90% within "
          f"{_radius(u['bootstrap_90_m'])}")
    print(f"assessment: {s['verdict']} - {s['summary']}")
    for f in s['flags']:
        print(f"  - {f['message']}")


def cmd_validate(a):
    c = _case(a)
    if a.synthetic:
        v = pipeline.validate_synthetic(c)
    else:
        if a.lat is None or a.lon is None or not a.source:
            sys.exit('give --lat, --lon and --source (or --synthetic for a synthetic case)')
        v = pipeline.validate(c, a.lat, a.lon, a.source)
    print(f"within the 90% bootstrap radius ({v['bootstrap_90_m']:.0f} m): {v['within_bootstrap_90']}")


def cmd_status(a):
    c = _case(a)
    print(json.dumps(c.summary(), indent=1))


def cmd_serve(a):
    from . import server
    server.serve(Path(a.cases), port=a.port, open_browser=not a.no_browser)


def cmd_export(a):
    from . import export
    export.export_cases([case_mod.Case(x) for x in a.cases], Path(a.out), audio=not a.no_audio, allow_real=a.allow_real)


def main(argv=None):
    p = argparse.ArgumentParser(prog='adoppler', description='Aircraft Doppler detection and acoustic '
                                'geolocation of a microphone from historical ADS-B.')
    p.add_argument('--version', action='version', version=__version__)
    sp = p.add_subparsers(dest='cmd', required=True, metavar='command')

    def add(name, fn, help_, case=True):
        q = sp.add_parser(name, help=help_, description=help_)
        if case:
            q.add_argument('case', help='case directory (holds case.json)')
        q.set_defaults(fn=fn)
        return q

    q = add('new', cmd_new, 'create a case from an audio file')
    q.add_argument('--audio', required=True, help='audio file (WAV, FLAC, OGG, MP3)')
    q.add_argument('--start', required=True, help='UTC time of the first sample, e.g. 2026-06-01T12:00:00Z')
    q.add_argument('--center', required=True, type=_center, help='search centre LAT,LON (a town, an area)')
    q.add_argument('--radius', type=float, help='search radius in km (default 20)')
    q.add_argument('--temp', type=float, help='ground air temperature in C (default 15)')
    q.add_argument('--terrain', choices=['flat', 'usgs-3dep'], help='terrain model (default flat)')
    q.add_argument('--elevation', type=float, help='ground elevation in m above sea level for flat terrain')
    q.add_argument('--adsb-file', help='your own ADS-B tracks (JSON or CSV) instead of downloading')
    q.add_argument('--name')
    q = add('synth', cmd_synth, 'render a synthetic test case with known answer')
    q.add_argument('--scene', default='example', choices=['example', 'mini'])
    q.add_argument('--seed', type=int, default=1)
    q = add('analyse', cmd_analyse, 'detect Doppler events in the audio')
    q.add_argument('--force', action='store_true', help='recompute')
    q = add('adsb', cmd_adsb, 'fetch historical ADS-B tracks for the recording')
    q.add_argument('--source', choices=['auto', 'adsb.lol', 'adsb.aero', 'file'])
    q.add_argument('--force', action='store_true', help='fetch again')
    for name, fn, h in (('match', cmd_match, 'estimate the microphone position (blind)'),
                        ('run', cmd_run, 'analyse + adsb + match')):
        q = add(name, fn, h)
        q.add_argument('--force', action='store_true', help='recompute (overwrites the stored prediction)')
        q.add_argument('--workers', type=int, help='parallel processes (default: CPUs - 1)')
        if name == 'match':
            q.add_argument('--events', nargs='+', help='use only these event ids')
    q = add('validate', cmd_validate, 'compare the stored result with a reference position')
    q.add_argument('--lat', type=float); q.add_argument('--lon', type=float)
    q.add_argument('--source', help='where the reference comes from (stored with the validation)')
    q.add_argument('--synthetic', action='store_true', help='use truth.json of a synthetic case')
    add('status', cmd_status, 'show settings and which stages are done')
    q = add('serve', cmd_serve, 'open the browser interface', case=False)
    q.add_argument('--cases', default='cases', help='folder that holds the case directories (default ./cases)')
    q.add_argument('--port', type=int, default=8765)
    q.add_argument('--no-browser', action='store_true')
    q = add('export', cmd_export, 'write a static, read-only copy of finished cases for a website', case=False)
    q.add_argument('out', help='output folder')
    q.add_argument('cases', nargs='+', help='case folders')
    q.add_argument('--no-audio', action='store_true', help='leave out the per-event audio clips')
    q.add_argument('--allow-real', action='store_true',
                   help='also export non-synthetic cases (publishes their location: only with permission from the recording owner)')
    a = p.parse_args(argv)
    try:
        a.fn(a)
    except (case_mod.CaseError, pipeline.StageError) as e:
        sys.exit(f'error: {e}')
    except KeyboardInterrupt:
        sys.exit('interrupted')


if __name__ == '__main__':
    main()
