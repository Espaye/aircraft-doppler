"""Static, read-only copy of finished cases for a website: the browser UI plus pre-computed JSON, images and
short audio clips. The copy contains positions (estimate, bootstrap, flights), so real recordings are refused
unless the owner has agreed to publish them."""
import io, json, shutil
from pathlib import Path

from . import server, views

PNG_W, PNG_H = 2400, 900


def _write_json(path, obj):
    path.write_text(json.dumps(server._clean(obj), allow_nan=False, default=server._default, separators=(',', ':')),
                    encoding='utf-8')


def _mp3(case, t0, t1):
    import soundfile as sf
    wav = views.audio_clip(case, t0, t1)
    y, fs = sf.read(io.BytesIO(wav), dtype='float32')
    buf = io.BytesIO()
    sf.write(buf, y, fs, format='MP3')
    return buf.getvalue()


def export_cases(cases, out, audio=True, allow_real=False, log=print):
    """Write <out>/index.html (static mode), assets and <out>/data/<case id>/... for each case."""
    out = Path(out)
    for c in cases:
        if not c.s.get('synthetic') and not allow_real:
            raise ValueError(f'{c.name} is not synthetic: exporting publishes its location and audio; '
                             'pass --allow-real only with the recording owner\'s permission')
        if not c.result_path.exists():
            raise ValueError(f'{c.name}: run the match before exporting')
    js = server.plotly_js()
    if not js:
        raise ValueError('plotly is not installed (pip install plotly)')
    (out / 'vendor').mkdir(parents=True, exist_ok=True)
    html = (server.WEB / 'index.html').read_text(encoding='utf-8').replace('<body>', '<body class="static">', 1)
    (out / 'index.html').write_text(html, encoding='utf-8')
    for name in ('app.js', 'style.css', 'favicon.svg'):
        shutil.copyfile(server.WEB / name, out / name)
    shutil.copyfile(js, out / 'vendor' / 'plotly.min.js')
    listing = []
    for c in cases:
        d = out / 'data' / c.dir.name
        if d.exists():
            shutil.rmtree(d)
        (d / 'audio').mkdir(parents=True)
        s = c.summary()
        s['settings'] = {k: v for k, v in s['settings'].items() if k not in ('audio', 'adsb_file')}
        s['audio_info'] = {k: v for k, v in c.audio_info().items() if k in ('duration_s', 'samplerate', 'channels')}
        _write_json(d / 'case.json', s)
        listing.append({k: s[k] for k in ('id', 'name', 'settings', 'status')})
        ev = views.events(c)
        _write_json(d / 'events.json', ev)
        _write_json(d / 'flights.json', views.flights(c))
        _write_json(d / 'result.json', views.result(c))
        for kind in ('tonal', 'power'):
            img, ext = views.spectrogram(c, width=PNG_W, height=PNG_H, kind=kind)
            (d / f'spec_{kind}.png').write_bytes(img)
            _write_json(d / f'spec_{kind}.json', ext)
        if audio:
            for e in ev['events']:
                (d / 'audio' / f"{e['id']}.mp3").write_bytes(_mp3(c, max(0, e['t_start'] - 5), e['t_end'] + 5))
        size = sum(p.stat().st_size for p in d.rglob('*') if p.is_file())
        log(f'{c.name}: {size / 1e6:.1f} MB in {d}')
    _write_json(out / 'data' / 'cases.json', listing)
    return out
