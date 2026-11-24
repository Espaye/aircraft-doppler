"""A case is one recording plus its settings, in its own directory:

    <case>/case.json           settings (below); everything else is derived and can be deleted
    <case>/<audio file>        WAV, FLAC, OGG or MP3 (anything libsndfile reads)
    <case>/analysis/           spectrogram, tonal enhancement, detected events (events.json)
    <case>/adsb/tracks.json    normalised ADS-B tracks for the recording window
    <case>/manual/<event>.json hand-drawn curves (optional, from the browser UI)
    <case>/match/result.json   location estimate with uncertainty; result.sha256 stamps it
    <case>/match/validation.json comparison with a reference position (optional)

case.json keys (defaults in adoppler.config.DEFAULTS):
    name                 free text
    audio                audio file name inside the case directory (or an absolute path)
    audio_start_utc      ISO 8601 UTC time of the first audio sample, e.g. "2026-06-01T12:00:00Z"
    audio_end_utc        optional wall-clock time the recording stopped (phone recorders write it as the
                         MP4/M4A creation_time). If it is later than the start plus the audio duration, the
                         recording has a gap (a pause or dropout) and sounds after it arrive late by up to that
                         much; the matching then searches that later clock too
    search_center        [lat, lon]: prior centre of the microphone search (e.g. a town centre)
    search_radius_km     radius of the search disc around search_center
    aircraft_radius_km   radius for retrieving ADS-B traffic
    ground_temp_c        ground air temperature (sets the sound speed; refined within +/- 5 C)
    delay_prior_s        [lo, hi]: allowed recording-clock minus acoustic-time offset
    mic_height_agl_m     microphone height above ground
    terrain              "flat" (uses ground_elevation_m, metres above sea level) or "usgs-3dep" (US only)
    geoid_undulation_m   ellipsoid minus sea level height; null = EGM96 via PROJ's network grid
    adsb_source          "auto" (adsb.lol archive, then adsb.aero), "adsb.lol", "adsb.aero" or "file"
    adsb_file            tracks file for adsb_source "file" (adoppler tracks JSON or CSV, see adsb.load_file)
    match                {coarse_step_m, min_score, workers, null_tests, modes, events}
"""
import copy, datetime as dt, json, math, re
from pathlib import Path

from . import config

CASE_FILE = 'case.json'
GAP_TOLERANCE_S = 10.0     # audio_end_utc within this of start + duration: no gap (file finalisation, rounding)


class CaseError(ValueError):
    """Invalid or incomplete case settings (message is meant for the user)."""


def _parse_utc(s):
    try:
        t = dt.datetime.fromisoformat(str(s).replace('Z', '+00:00'))
    except ValueError:
        raise CaseError(f'audio_start_utc {s!r} is not an ISO 8601 time such as 2026-06-01T12:00:00Z')
    if t.tzinfo is None:
        raise CaseError('audio_start_utc needs a time zone; use UTC with a trailing Z')
    return t.timestamp()


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def slug(name):
    s = re.sub(r'[^a-z0-9]+', '-', str(name).lower()).strip('-')
    return s[:60] or 'case'


def audio_info(path):
    import soundfile as sf
    path = Path(path)
    if not path.exists():
        raise CaseError(f'audio file {path} not found')
    try:
        i = sf.info(str(path))
    except Exception as e:
        raise CaseError(f'cannot read {path.name} ({e}); convert it to WAV or FLAC, e.g. '
                        f'ffmpeg -i input.m4a audio.flac')
    return {'samplerate': i.samplerate, 'channels': i.channels, 'frames': i.frames,
            'duration_s': i.frames / i.samplerate, 'format': i.format}


class Case:
    def __init__(self, path):
        self.dir = Path(path).resolve()
        p = self.dir / CASE_FILE
        if not p.exists():
            raise CaseError(f'{p} not found (create a case with `adoppler new` or the browser UI)')
        try:
            raw = json.loads(p.read_text(encoding='utf-8'))
        except json.JSONDecodeError as e:
            raise CaseError(f'{p} is not valid JSON: {e}')
        self.raw = raw
        self.s = _merge(config.DEFAULTS, raw)
        self.validate()

    # ------------------------------------------------------------ settings

    def validate(self):
        s = self.s
        for k in ('audio', 'audio_start_utc', 'search_center'):
            if s.get(k) in (None, ''):
                raise CaseError(f'case.json: "{k}" is required')
        lat, lon = self.center
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise CaseError('search_center must be [lat, lon] in degrees')
        if not 0 < float(s['search_radius_km']) <= 200:
            raise CaseError('search_radius_km must be between 0 and 200')
        lo, hi = (float(x) for x in s['delay_prior_s'])
        if not lo < hi:
            raise CaseError('delay_prior_s must be [lo, hi] with lo < hi')
        if s['terrain'] not in ('flat', 'usgs-3dep'):
            raise CaseError('terrain must be "flat" or "usgs-3dep"')
        if s['adsb_source'] not in ('auto', 'adsb.lol', 'adsb.aero', 'file'):
            raise CaseError('adsb_source must be auto, adsb.lol, adsb.aero or file')
        if s['adsb_source'] == 'file' and not s.get('adsb_file'):
            raise CaseError('adsb_source "file" needs adsb_file')
        self.t0 = _parse_utc(s['audio_start_utc'])
        self.t_end = None if s.get('audio_end_utc') in (None, '') else _parse_utc(s['audio_end_utc'])
        if self.t_end is not None and self.t_end <= self.t0:
            raise CaseError('audio_end_utc must be later than audio_start_utc')

    @property
    def name(self):
        return self.s.get('name') or self.dir.name

    @property
    def center(self):
        try:
            lat, lon = (float(x) for x in self.s['search_center'])
        except (TypeError, ValueError):
            raise CaseError('search_center must be [lat, lon]')
        return lat, lon

    @property
    def radius_m(self):
        return float(self.s['search_radius_km']) * 1000

    @property
    def aircraft_radius_m(self):
        return float(self.s['aircraft_radius_km']) * 1000

    @property
    def delay_prior(self):
        return tuple(float(x) for x in self.s['delay_prior_s'])

    @property
    def clock_gap_s(self):
        """Wall-clock time missing from the audio (pause or dropout), from audio_end_utc.

        None when audio_end_utc is not set, 0 when it agrees with the start plus the duration."""
        if self.t_end is None:
            return None
        gap = self.t_end - self.t0 - self.audio_info()['duration_s']
        if gap < -GAP_TOLERANCE_S:
            raise CaseError(f'audio_end_utc is {-gap:.0f} s before audio_start_utc plus the audio duration; '
                            'one of the two times is wrong')
        return gap if gap > GAP_TOLERANCE_S else 0.0

    @property
    def delay_range(self):
        """delay_prior_s, widened so that sounds after a recording gap can also be matched."""
        lo, hi = self.delay_prior
        return lo - (self.clock_gap_s or 0.0), hi

    @property
    def ground_temp_c(self):
        return float(self.s['ground_temp_c'])

    def warnings(self):
        """Settings that are probably wrong or missing, in plain language."""
        out = []
        if 'ground_temp_c' not in self.raw:
            out.append(f"ground_temp_c is not set, so {self.ground_temp_c:.0f} C is assumed; a 10 C error changes "
                       f"the sound speed by about {100 * (1 - math.sqrt(1 - 10 / 288.15)):.1f} %. Set the measured "
                       "temperature (for example from the nearest weather station).")
        try:
            gap = self.clock_gap_s
        except CaseError as e:                  # also: audio missing or unreadable
            return out + [str(e)]
        if gap:
            out.append(f'the recording is {gap:.0f} s shorter than the time between audio_start_utc and '
                       'audio_end_utc: it probably has a pause or dropout, so sounds after it arrive later than '
                       'the start time implies. The matching searches that later clock as well.')
        return out

    @property
    def mic_height(self):
        return float(self.s['mic_height_agl_m'])

    # ------------------------------------------------------------ paths

    @property
    def audio_path(self):
        p = Path(self.s['audio'])
        return p if p.is_absolute() else self.dir / p

    analysis_dir = property(lambda self: self.dir / 'analysis')
    adsb_dir = property(lambda self: self.dir / 'adsb')
    match_dir = property(lambda self: self.dir / 'match')
    manual_dir = property(lambda self: self.dir / 'manual')
    events_path = property(lambda self: self.analysis_dir / 'events.json')
    tracks_path = property(lambda self: self.adsb_dir / 'tracks.json')
    result_path = property(lambda self: self.match_dir / 'result.json')
    validation_path = property(lambda self: self.match_dir / 'validation.json')

    def audio_info(self):
        return audio_info(self.audio_path)

    @property
    def window(self):
        """UTC unix seconds covered by the recording."""
        return self.t0, self.t0 + self.audio_info()['duration_s']

    # ------------------------------------------------------------ derived settings

    def geoid_n(self):
        """Geoid undulation N (h_ellipsoid = h_msl + N). Resolved once and cached in the case directory."""
        if self.s.get('geoid_undulation_m') is not None:
            return float(self.s['geoid_undulation_m'])
        p = self.dir / 'analysis' / 'geoid.json'
        if p.exists():
            return json.loads(p.read_text())['n_m']
        from . import geo
        n, src = geo.geoid_undulation(*self.center)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({'n_m': n, 'source': src}))
        return n

    def terrain(self):
        from . import geo
        if self.s['terrain'] == 'flat':
            return geo.FlatTerrain(float(self.s['ground_elevation_m']))
        lat, lon = self.center
        r = self.radius_m * 1.3
        dlat = r / 111_000
        import math
        dlon = dlat / max(0.1, math.cos(math.radians(lat)))
        bbox = (round(lon - dlon, 4), round(lat - dlat, 4), round(lon + dlon, 4), round(lat + dlat, 4))
        return geo.Terrain3DEP(self.dir / 'terrain' / 'usgs_3dep.tif', bbox)

    # ------------------------------------------------------------ status

    def status(self):
        st = {'audio': self.audio_path.exists(), 'events': self.events_path.exists(),
              'adsb': self.tracks_path.exists(), 'match': self.result_path.exists(),
              'validated': self.validation_path.exists()}
        st['manual'] = sorted(p.stem for p in self.manual_dir.glob('*.json')) if self.manual_dir.exists() else []
        return st

    def summary(self):
        return {'id': self.dir.name, 'name': self.name, 'settings': self.s, 'status': self.status(),
                'warnings': self.warnings()}


def create(path, settings, overwrite=False):
    """Write case.json (merged with nothing: only the given keys are stored) and return the Case."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    p = path / CASE_FILE
    if p.exists() and not overwrite:
        raise CaseError(f'{p} already exists')
    p.write_text(json.dumps(settings, indent=1), encoding='utf-8')
    try:
        return Case(path)
    except CaseError:
        if not overwrite:
            p.unlink()
        raise


def update(case, changes):
    """Change settings; derived results that depend on them are kept (rerun stages with --force)."""
    raw = _merge(case.raw, changes)
    p = case.dir / CASE_FILE
    old = p.read_text(encoding='utf-8')
    p.write_text(json.dumps(raw, indent=1), encoding='utf-8')
    try:
        return Case(case.dir)
    except CaseError:
        p.write_text(old, encoding='utf-8')
        raise
