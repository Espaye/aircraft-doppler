"""Historical ADS-B retrieval and normalisation.

Sources:
- adsb.lol daily archive (github.com/adsblol/globe_history_<year>, ODbL): raw readsb traces with GNSS altitude,
  streamed once (~3 GB per day, global) and filtered to the region without storing the tarball. Preferred.
- adsb.aero /query: targeted geometric + time query with simplified paths (derived from adsb.lol). Fallback.
- A local file (adoppler tracks JSON or CSV), e.g. for synthetic cases or other providers' exports.

All are normalised to `Track` objects: time (unix s), lat, lon and a best-estimate altitude in metres above
mean sea level, with the altitude's provenance recorded per point.
"""
import csv, datetime as dt, gzip, io, json, math, re, tarfile, time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import requests

from . import config

AERO = 'https://adsb.aero/api/v1'


@dataclass
class Track:
    icao24: str
    callsign: str | None
    icao_type: str | None
    category: str | None
    source: str
    t: np.ndarray            # unix seconds
    lat: np.ndarray
    lon: np.ndarray
    alt_msl_m: np.ndarray    # best estimate of true altitude above MSL
    alt_kind: np.ndarray     # per point: 'geom' (GNSS->MSL), 'baro+qnh', 'baro' (uncorrected), 'ground'
    meta: dict = field(default_factory=dict)

    def to_json(self):
        d = asdict(self)
        for k in ('t', 'lat', 'lon', 'alt_msl_m', 'alt_kind'):
            d[k] = np.asarray(d[k]).tolist()
        return d

    @staticmethod
    def from_json(d):
        d = dict(d)
        for k in ('t', 'lat', 'lon', 'alt_msl_m'):
            d[k] = np.asarray(d[k], float)
        d['alt_kind'] = np.asarray(d['alt_kind'])
        return Track(**d)

    @property
    def label(self):
        return f"{(self.callsign or '').strip() or self.icao24} ({self.icao_type or self.category or '?'})"


def save_tracks(tracks, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([t.to_json() for t in tracks]))


def load_tracks(path):
    return [Track.from_json(d) for d in json.loads(Path(path).read_text())]


def _iso(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# ---------------------------------------------------------------- adsb.aero

def aero_data_range():
    return requests.get(f'{AERO}/data-range', timeout=60).json()


def aero_query(t_from, t_to, lat, lon, radius_m, cache=None, page_limit=100):
    """All flights whose trajectory enters the circle while inside [t_from, t_to] (unix s), with full paths."""
    if cache and Path(cache).exists():
        return json.loads(Path(cache).read_text())
    body = {'match': {'trajectory_intersects': {
                'geometry': {'type': 'Circle', 'coordinates': [lon, lat], 'radius': radius_m},
                'time_from': _iso(t_from), 'time_to': _iso(t_to)}},
            # end_date bounds departure time (exclusive); flights may have departed the previous day
            'end_date': _iso(t_to + 60), 'start_from': _iso(t_from - 86400), 'window_days': 2,
            'limit': page_limit, 'include_path': True}
    flights, pages = [], []
    while True:
        for k in range(5):
            r = requests.post(f'{AERO}/query', json=body, timeout=300)
            if r.status_code == 200:
                break
            time.sleep(5 * (k + 1))
        r.raise_for_status()
        d = r.json()
        flights += d['flights']
        pages.append({'n': len(d['flights']), 'window_from': d.get('window_from')})
        if not d.get('cursor'):
            break
        body['cursor'] = d['cursor']
        time.sleep(1)
    out = {'query': {k: v for k, v in body.items() if k != 'cursor'}, 'pages': pages,
           'retrieved_utc': _iso(time.time()), 'flights': flights}
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(out))
    return out


def _ffill(series, t):
    """Evaluate a sparse [[ts, value], ...] change-point series (forward fill) at times t."""
    pts = [p for seg in (series or []) for p in seg if p[1] is not None]
    if not pts:
        return np.full(len(t), np.nan)
    pts.sort(key=lambda p: p[0])
    ts = np.array([p[0] for p in pts]); v = np.array([float(p[1]) for p in pts])
    i = np.searchsorted(ts, t, side='right') - 1
    return np.where(i >= 0, v[np.clip(i, 0, None)], v[0])


def parse_aero(f):
    coords = [c for seg in f['path']['coordinates'] for c in seg]
    ts = np.array([x for seg in f['timestamps'] for x in seg], float)
    lon = np.array([c[0] for c in coords], float); lat = np.array([c[1] for c in coords], float)
    p_alt_ft = np.array([c[2] if c[2] is not None else np.nan for c in coords], float)
    corr = _ffill(f.get('alt_correction_ft'), ts)
    has = np.isfinite(corr)
    alt = (p_alt_ft + np.where(has, corr, 0.0)) * config.FT
    kind = np.where(has, 'baro+qnh', 'baro')
    seg_id = np.concatenate([np.full(len(s), i) for i, s in enumerate(f['timestamps'])])
    return Track(f['icao24'], f.get('callsign'), f.get('icao_type'), f.get('emitter_category'), 'adsb.aero',
                 ts, lat, lon, alt, kind,
                 meta={'flight_id': f['flight_id'], 'registration': f.get('registration'),
                       'operator': f.get('operator'), 'start_airport': f.get('start_airport_ident'),
                       'end_airport': f.get('end_airport_ident'), 'point_count': f.get('point_count'),
                       'raw_point_count': f.get('raw_point_count'), 'segment': seg_id.tolist(),
                       'gs_kt': _ffill(f.get('path_gs'), ts).tolist(), 'track_deg': _ffill(f.get('path_tracks'), ts).tolist(),
                       'vr_fpm': _ffill(f.get('path_vr'), ts).tolist()})


# ---------------------------------------------------------------- adsb.lol daily archive (fallback)

class _Concat(io.RawIOBase):
    """Read several HTTP streams back to back as one file object."""

    def __init__(self, urls):
        self.urls = list(urls); self.r = None; self.it = None; self.bytes = 0

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            if self.it is None:
                if not self.urls:
                    return 0
                self.r = requests.get(self.urls.pop(0), stream=True, timeout=120)
                self.r.raise_for_status()
                self.it = self.r.raw
            n = self.it.readinto(b)
            if n:
                self.bytes += n
                return n
            self.it = None


def release_tag(day, instance='prod-0'):
    return f'v{day:%Y.%m.%d}-planes-readsb-{instance}'


def release_repo(day):
    return f'adsblol/globe_history_{day:%Y}'


def release_assets(tag, day):
    r = requests.get(f'https://api.github.com/repos/{release_repo(day)}/releases/tags/{tag}', timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return sorted(a['browser_download_url'] for a in r.json()['assets'])


def stream_archive(day, t_from, t_to, out_dir, lat, lon, radius_m, instance='prod-0', log=print):
    """Stream a daily archive, keep only trace files with a point inside the circle and time window."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / '_complete.json'
    if done.exists():
        info = json.loads(done.read_text())
        a, b = info.get('window', (t_from, t_to))
        clat, clon, r = info.get('area', (lat, lon, radius_m))
        if a <= t_from and t_to <= b and (clat, clon) == (lat, lon) and r >= radius_m:
            return info
        for f in out_dir.glob('*.json'):                   # kept for another window or area: stream again
            f.unlink()
    tag = release_tag(day, instance)
    urls = release_assets(tag, day)
    if not urls:
        raise RuntimeError(f'release {tag} not published yet')
    dlat = radius_m / 111_000; dlon = dlat / math.cos(math.radians(lat))
    src = _Concat(urls)
    kept, seen, t0 = 0, 0, time.time()
    with tarfile.open(fileobj=io.BufferedReader(src, 1 << 20), mode='r|') as tf:
        for m in tf:
            if not m.isfile() or '/traces/' not in m.name and not m.name.startswith('traces/'):
                continue
            seen += 1
            raw = tf.extractfile(m).read()
            try:
                d = json.loads(gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw)
            except Exception:
                continue
            base = d.get('timestamp', 0)
            hit = False
            for p in d.get('trace', ()):
                t = base + p[0]
                if t_from <= t <= t_to and abs(p[1] - lat) < dlat and abs(p[2] - lon) < dlon:
                    hit = True
                    break
            icao = str(d.get('icao', ''))
            if hit and re.fullmatch(r'~?[0-9a-f]{6}', icao):     # untrusted input: hex address as file name only
                (out_dir / f"{icao}.json").write_text(json.dumps(d))
                kept += 1
            if log and seen % 20000 == 0:
                log(f'  {seen} traces scanned, {kept} kept, {src.bytes / 1e9:.2f} GB, {time.time() - t0:.0f} s')
    info = {'tag': tag, 'urls': urls, 'scanned': seen, 'kept': kept, 'bytes': src.bytes,
            'window': [t_from, t_to], 'area': [lat, lon, radius_m], 'seconds': time.time() - t0}
    done.write_text(json.dumps(info))
    return info


def parse_readsb(d, geoid_n, qnh_fn=None):
    """readsb trace_full JSON -> Track. Uses GNSS altitude (WGS84 HAE -> MSL) where present, else baro (+QNH fn)."""
    base = d['timestamp']
    tr = d['trace']
    t = np.array([base + p[0] for p in tr], float)
    lat = np.array([p[1] for p in tr], float); lon = np.array([p[2] for p in tr], float)
    flags = np.array([p[6] for p in tr], int)
    baro = np.array([p[3] if isinstance(p[3], (int, float)) else np.nan for p in tr], float)
    ground = np.array([p[3] == 'ground' for p in tr])
    geom = np.array([p[10] if len(p) > 10 and isinstance(p[10], (int, float)) else np.nan for p in tr], float)
    alt_is_geom = (flags & 8) > 0
    geom = np.where(alt_is_geom & np.isnan(geom), baro, geom)
    baro = np.where(alt_is_geom, np.nan, baro)
    alt = np.full(len(t), np.nan); kind = np.full(len(t), 'none', dtype=object)
    g = np.isfinite(geom)
    alt[g] = geom[g] * config.FT - geoid_n; kind[g] = 'geom'
    b = ~g & np.isfinite(baro)
    if qnh_fn is not None:
        alt[b] = (baro[b] + qnh_fn(baro[b], t[b])) * config.FT; kind[b] = 'baro+qnh'
    else:
        alt[b] = baro[b] * config.FT; kind[b] = 'baro'
    kind[ground] = 'ground'
    callsign = None
    for p in tr:
        if len(p) > 8 and isinstance(p[8], dict) and p[8].get('flight'):
            callsign = p[8]['flight'].strip()
            break
    stale = (flags & 1) > 0
    keep = ~stale & np.isfinite(lat)
    return Track(d['icao'], callsign, d.get('t'), None, 'adsb.lol-archive', t[keep], lat[keep], lon[keep],
                 alt[keep], kind[keep].astype(str),
                 meta={'registration': d.get('r'), 'desc': d.get('desc'),
                       'gs_kt': np.array([p[4] if p[4] is not None else np.nan for p in tr], float)[keep].tolist(),
                       'track_deg': np.array([p[5] if p[5] is not None else np.nan for p in tr], float)[keep].tolist(),
                       'source_type': [p[9] if len(p) > 9 else None for p, k in zip(tr, keep) if k],
                       'geom_minus_baro_ft': (geom - baro)[keep].tolist()})


def load_archive_dir(arch, geoid_n):
    return [parse_readsb(json.loads(f.read_text()), geoid_n) for f in sorted(Path(arch).glob('*.json'))
            if not f.name.startswith('_')]


# ---------------------------------------------------------------- local files

CSV_HELP = ('CSV columns: icao24, time (unix seconds or ISO 8601 UTC), lat, lon, alt_m, and optionally alt_ref '
            '(msl, hae or baro; default msl), callsign, type, gs_kt. One row per position report.')


def _time(v):
    try:
        return float(v)
    except ValueError:
        return dt.datetime.fromisoformat(v.replace('Z', '+00:00')).timestamp()


def load_file(path, geoid_n):
    """Tracks from an adoppler tracks JSON (list of Track dicts) or a CSV (see CSV_HELP)."""
    path = Path(path)
    if path.suffix.lower() == '.json':
        return load_tracks(path)
    rows = {}
    with open(path, newline='', encoding='utf-8-sig') as fh:
        rd = csv.DictReader(fh)
        need = {'icao24', 'time', 'lat', 'lon', 'alt_m'}
        if not need <= set(rd.fieldnames or []):
            raise ValueError(f'{path.name}: missing columns {sorted(need - set(rd.fieldnames or []))}. ' + CSV_HELP)
        for r in rd:
            rows.setdefault(r['icao24'].strip().lower(), []).append(r)
    tracks = []
    for icao, rs in rows.items():
        rs.sort(key=lambda r: _time(r['time']))
        t = np.array([_time(r['time']) for r in rs])
        alt = np.array([float(r['alt_m']) if r['alt_m'] not in ('', None) else np.nan for r in rs])
        ref = np.array([(r.get('alt_ref') or 'msl').strip().lower() for r in rs])
        alt = np.where(ref == 'hae', alt - geoid_n, alt)
        kind = np.where(ref == 'baro', 'baro', 'geom')
        gs = [float(r['gs_kt']) if r.get('gs_kt') not in ('', None) else np.nan for r in rs]
        tracks.append(Track(icao, (rs[0].get('callsign') or '').strip() or None, (rs[0].get('type') or '').strip() or None,
                            None, f'file:{path.name}', t, np.array([float(r['lat']) for r in rs]),
                            np.array([float(r['lon']) for r in rs]), alt, kind, meta={'gs_kt': gs}))
    return tracks
