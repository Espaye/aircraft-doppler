"""Local web server for the browser UI: `adoppler serve`. Listens on 127.0.0.1 only.

Security: the server accepts requests only with a local Host header (against DNS rebinding), state-changing
requests only with the custom X-Adoppler header and a local or absent Origin (a cross-site page cannot send
either without a CORS preflight, which is never granted), case ids are restricted to slugs inside the cases
folder, uploads are size-limited and written under fixed names, and static files come from a fixed list.
"""
import http.server, json, math, re, shutil, sys, threading, time, traceback, urllib.parse, uuid, webbrowser
from pathlib import Path

from . import __version__, case as case_mod, pipeline, views

WEB = Path(__file__).parent / 'web'
STATIC = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
          '/style.css': ('style.css', 'text/css; charset=utf-8'), '/favicon.svg': ('favicon.svg', 'image/svg+xml')}
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: https:; "
       "connect-src 'self' https:; worker-src blob:; font-src 'self' data: https:; media-src 'self' blob:; "
       "frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
ID_RE = re.compile(r'[a-z0-9][a-z0-9-]{0,59}')
AUDIO_EXT = {'.wav', '.flac', '.ogg', '.mp3', '.aiff', '.aif'}
ADSB_EXT = {'.json', '.csv'}
MAX_UPLOAD = 4 << 30
MAX_JSON = 1 << 20


def plotly_js():
    try:
        import plotly
    except ImportError:
        return None
    p = Path(plotly.__file__).parent / 'package_data' / 'plotly.min.js'
    return p if p.exists() else None


class Jobs:
    """Background stage runs; one at a time per case, logs kept in memory."""

    def __init__(self):
        self.jobs, self.lock = {}, threading.Lock()

    def busy(self, case_id):
        with self.lock:
            return any(j['case'] == case_id and j['state'] == 'running' for j in self.jobs.values())

    def start(self, case_id, stage, fn):
        if self.busy(case_id):
            raise ApiError(409, 'a stage is already running for this case')
        jid = uuid.uuid4().hex[:12]
        job = {'id': jid, 'case': case_id, 'stage': stage, 'state': 'running', 'log': [], 'error': None,
               'started': time.time(), 'finished': None}

        def log(msg):
            job['log'].append(f'{time.strftime("%H:%M:%S")} {msg}')

        def work():
            try:
                fn(log)
                job['state'] = 'done'
            except (case_mod.CaseError, pipeline.StageError, ValueError) as e:
                job['state'], job['error'] = 'failed', str(e)
            except Exception as e:                     # unexpected: keep the traceback in the server console
                traceback.print_exc()
                job['state'], job['error'] = 'failed', f'{type(e).__name__}: {e}'
            job['finished'] = time.time()
        with self.lock:
            self.jobs[jid] = job
        threading.Thread(target=work, daemon=True).start()
        return job


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = f'adoppler/{__version__}'
    root: Path = None
    jobs: Jobs = None
    port: int = 0

    def log_message(self, fmt, *args):
        if '/api/jobs/' not in self.path:
            sys.stderr.write(f'{self.address_string()} {fmt % args}\n')

    # ---------------------------------------------------------------- plumbing

    def _local_host(self):
        host = (self.headers.get('Host') or '').lower()
        return host in (f'127.0.0.1:{self.port}', f'localhost:{self.port}')

    def _check_write(self):
        if self.headers.get('X-Adoppler') != '1':
            raise ApiError(403, 'missing X-Adoppler header')
        origin = self.headers.get('Origin')
        if origin and origin.lower() not in (f'http://127.0.0.1:{self.port}', f'http://localhost:{self.port}'):
            raise ApiError(403, 'cross-origin request refused')

    def _send(self, status, body, ctype, extra=None):
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Security-Policy', CSP)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(status, json.dumps(_clean(obj), allow_nan=False, default=_default).encode(), 'application/json')

    def _body_json(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n > MAX_JSON:
            raise ApiError(413, 'request too large')
        if n and 'application/json' not in (self.headers.get('Content-Type') or ''):
            raise ApiError(415, 'expected application/json')
        try:
            return json.loads(self.rfile.read(n) or b'{}')
        except json.JSONDecodeError:
            raise ApiError(400, 'invalid JSON')

    def _case_dir(self, cid, must_exist=True):
        if not ID_RE.fullmatch(cid or ''):
            raise ApiError(400, 'invalid case id')
        d = (self.root / cid).resolve()
        if d.parent != self.root:
            raise ApiError(400, 'invalid case id')
        if must_exist and not (d / case_mod.CASE_FILE).exists():
            raise ApiError(404, f'case {cid} not found')
        return d

    def _case(self, cid):
        try:
            return case_mod.Case(self._case_dir(cid))
        except case_mod.CaseError as e:
            raise ApiError(400, str(e))

    def _dispatch(self, method):
        if not self._local_host():
            return self._send(421, b'only local requests are served', 'text/plain')
        u = urllib.parse.urlsplit(self.path)
        q = {k: v[-1] for k, v in urllib.parse.parse_qs(u.query).items()}
        parts = [urllib.parse.unquote(p) for p in u.path.strip('/').split('/') if p]
        try:
            if method in ('GET', 'HEAD'):
                if u.path in STATIC:
                    name, ctype = STATIC[u.path]
                    return self._send(200, (WEB / name).read_bytes(), ctype)
                if u.path == '/vendor/plotly.min.js':
                    p = plotly_js()
                    if not p:
                        raise ApiError(404, 'plotly is not installed (pip install plotly)')
                    return self._send(200, p.read_bytes(), 'text/javascript; charset=utf-8',
                                      {'Cache-Control': 'max-age=86400'})
            if not parts or parts[0] != 'api':
                raise ApiError(404, 'not found')
            if method not in ('GET', 'HEAD'):
                self._check_write()
            return self._api(method, parts[1:], q)
        except ApiError as e:
            self._json({'error': str(e)}, e.status)
        except (case_mod.CaseError, pipeline.StageError, ValueError, FileNotFoundError) as e:
            self._json({'error': str(e)}, 400)
        except Exception as e:
            traceback.print_exc()
            self._json({'error': f'internal error: {type(e).__name__}'}, 500)

    def do_GET(self):
        self._dispatch('GET')

    def do_HEAD(self):
        self._dispatch('HEAD')

    def do_POST(self):
        self._dispatch('POST')

    def do_PUT(self):
        self._dispatch('PUT')

    # ---------------------------------------------------------------- API

    def _api(self, method, p, q):
        if p == ['info'] and method == 'GET':
            return self._json({'version': __version__, 'cases_dir': str(self.root), 'plotly': bool(plotly_js())})
        if p == ['cases'] and method == 'GET':
            return self._json(self._list())
        if p == ['cases'] and method == 'POST':
            return self._json(self._create(self._body_json()), 201)
        if p == ['synth'] and method == 'POST':
            return self._json(self._synth(self._body_json()), 202)
        if len(p) == 2 and p[0] == 'jobs' and method == 'GET':
            j = self.jobs.jobs.get(p[1])
            if not j:
                raise ApiError(404, 'no such job')
            since = int(q.get('since', 0))
            return self._json({**j, 'log': j['log'][since:], 'log_total': len(j['log'])})
        if len(p) < 2 or p[0] != 'cases':
            raise ApiError(404, 'not found')
        cid, rest = p[1], p[2:]
        if method == 'GET':
            c = self._case(cid)
            if not rest:
                s = c.summary()
                try:
                    s['audio_info'] = c.audio_info()
                except case_mod.CaseError as e:
                    s['audio_error'] = str(e)
                s['running'] = self.jobs.busy(cid)
                s['jobs'] = [{k: j[k] for k in ('id', 'stage', 'state', 'error', 'started', 'finished')}
                             for j in self.jobs.jobs.values() if j['case'] == cid][-6:]
                return self._json(s)
            what = rest[0]
            if what == 'events':
                return self._need(c.events_path, 'run the analysis first') or self._json(views.events(c))
            if what == 'flights':
                return self._need(c.tracks_path, 'fetch flight data first') or self._json(views.flights(c))
            if what == 'result':
                return self._need(c.result_path, 'run the match first') or self._json(views.result(c))
            if what == 'spectrogram':
                self._need(c.events_path, 'run the analysis first', raise_=True)
                f = lambda k: float(q[k]) if q.get(k) not in (None, '') else None
                img, ext = views.spectrogram(c, f('t0'), f('t1'), f('f0') or 0.0, f('f1'),
                                             width=min(2400, int(q.get('w', 1600))),
                                             height=min(1200, int(q.get('h', 600))), kind=q.get('kind', 'tonal'))
                return self._send(200, img, 'image/png', {'X-Extent': json.dumps(ext)})
            if what == 'audio':
                wav = views.audio_clip(c, float(q.get('t0', 0)), float(q.get('t1', 60)))
                return self._send(200, wav, 'audio/wav')
            raise ApiError(404, 'not found')
        if method == 'PUT' and rest == ['settings']:
            c = self._case(cid)
            if self.jobs.busy(cid):
                raise ApiError(409, 'a stage is running')
            ch = self._body_json()
            if not isinstance(ch, dict):
                raise ApiError(400, 'expected an object')
            ch.pop('synthetic', None)
            return self._json(case_mod.update(c, ch).summary())
        if method == 'PUT' and rest == ['upload']:
            return self._json(self._upload(cid, q))
        if method == 'POST' and rest == ['run']:
            b = self._body_json()
            return self._json(self._run(cid, b.get('stage'), bool(b.get('force'))), 202)
        if method == 'POST' and rest == ['validate']:
            c = self._case(cid)
            b = self._body_json()
            if b.get('synthetic'):
                return self._json(pipeline.validate_synthetic(c))
            try:
                lat, lon = float(b['lat']), float(b['lon'])
            except (KeyError, TypeError, ValueError):
                raise ApiError(400, 'lat and lon are required')
            src = str(b.get('source') or '').strip()
            if not src:
                raise ApiError(400, 'describe where the reference position comes from')
            return self._json(pipeline.validate(c, lat, lon, src[:300]))
        raise ApiError(404, 'not found')

    def _need(self, path, msg, raise_=False):
        if not path.exists():
            if raise_:
                raise ApiError(409, msg)
            self._json({'error': msg, 'missing': True}, 409)
            return True
        return None

    def _list(self):
        out = []
        for d in sorted(self.root.iterdir()) if self.root.exists() else []:
            if d.is_dir() and ID_RE.fullmatch(d.name) and (d / case_mod.CASE_FILE).exists():
                try:
                    out.append(case_mod.Case(d).summary())
                except case_mod.CaseError as e:
                    out.append({'id': d.name, 'name': d.name, 'error': str(e)})
        return out

    def _new_id(self, name):
        base = case_mod.slug(name)
        cid, k = base, 2
        while (self.root / cid).exists():
            cid = f'{base[:55]}-{k}'; k += 1
        return cid

    def _create(self, b):
        if not isinstance(b, dict):
            raise ApiError(400, 'expected an object')
        audio = str(b.get('audio') or '')
        ext = Path(audio).suffix.lower()
        if ext not in AUDIO_EXT:
            raise ApiError(400, f'audio must be one of {", ".join(sorted(AUDIO_EXT))}')
        s = {k: v for k, v in b.items() if k in _ALLOWED}
        s['audio'] = 'audio' + ext                       # stored under a fixed name, uploaded next
        cid = self._new_id(b.get('name') or 'case')
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            c = case_mod.create(self.root / cid, s)
        except case_mod.CaseError:
            shutil.rmtree(self.root / cid, ignore_errors=True)
            raise
        return c.summary()

    def _upload(self, cid, q):
        c = self._case(cid)
        if self.jobs.busy(cid):
            raise ApiError(409, 'a stage is running')
        kind = q.get('kind', 'audio')
        ext = Path(q.get('name', '')).suffix.lower()
        allowed = AUDIO_EXT if kind == 'audio' else ADSB_EXT if kind == 'adsb' else set()
        if ext not in allowed:
            raise ApiError(400, f'{kind} file must be one of {", ".join(sorted(allowed))}')
        n = int(self.headers.get('Content-Length') or -1)
        if n < 0:
            raise ApiError(411, 'Content-Length required')
        if n > MAX_UPLOAD:
            raise ApiError(413, 'file larger than 4 GB')
        name = ('audio' if kind == 'audio' else 'flights') + ext
        tmp = c.dir / (name + '.part')
        left = n
        with open(tmp, 'wb') as fh:
            while left > 0:
                buf = self.rfile.read(min(left, 1 << 20))
                if not buf:
                    break
                fh.write(buf); left -= len(buf)
        if left:
            tmp.unlink(missing_ok=True)
            raise ApiError(400, 'upload incomplete')
        tmp.replace(c.dir / name)
        ch = {'audio': name} if kind == 'audio' else {'adsb_source': 'file', 'adsb_file': name}
        c = case_mod.update(c, ch)
        s = c.summary()
        if kind == 'audio':
            try:
                s['audio_info'] = c.audio_info()
            except case_mod.CaseError as e:
                s['audio_error'] = str(e)
        return s

    def _run(self, cid, stage, force):
        c = self._case(cid)
        fns = {'analyse': lambda log: pipeline.analyse(c, force=force, log=log),
               'adsb': lambda log: pipeline.fetch_adsb(c, force=force, log=log),
               'match': lambda log: pipeline.match(c, force=force, log=log),
               'all': lambda log: pipeline.run_all(c, force=force, log=log)}
        if stage not in fns:
            raise ApiError(400, 'stage must be analyse, adsb, match or all')
        return self.jobs.start(cid, stage, fns[stage])

    def _synth(self, b):
        from . import synth
        scene = b.get('scene', 'example')
        if scene not in synth.SCENES:
            raise ApiError(400, 'unknown scene')
        cid = self._new_id(f'synthetic-{scene}')
        d = self.root / cid

        def work(log):
            synth.make_case(d, scene, seed=int(b.get('seed', 1)), log=log)
            c = case_mod.Case(d)
            pipeline.run_all(c, log=log) if b.get('run') else None
        job = self.jobs.start(cid, 'synth', work)
        return {**job, 'case_id': cid}


_ALLOWED = {'name', 'audio_start_utc', 'search_center', 'search_radius_km', 'aircraft_radius_km', 'ground_temp_c',
            'delay_prior_s', 'mic_height_agl_m', 'terrain', 'ground_elevation_m', 'geoid_undulation_m', 'adsb_source',
            'match'}


def _clean(o):
    """NaN/inf -> null (JSON has no NaN; results contain NaN for undefined quantities)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _default(o):
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def serve(root, port=8765, open_browser=True):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    Handler.root, Handler.jobs, Handler.port = root, Jobs(), port
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', port), Handler)
    httpd.daemon_threads = True
    url = f'http://127.0.0.1:{port}/'
    print(f'adoppler {__version__}: {url}  (cases in {root}; Ctrl+C to stop)')
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('stopped')
    finally:
        httpd.server_close()
