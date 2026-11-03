"""Build the static website in site/: render and run the synthetic example, export it to site/demo and write its
numbers into site/index.html. Deterministic (fixed seed), about two minutes.

    python tools/build_site.py [--cases DIR] [--rerun]
"""
import argparse, json, re, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adoppler import case, export, pipeline, synth  # noqa: E402

CASE_ID = 'synth-example'


def facts(c):
    r = json.loads(c.result_path.read_text())
    v = json.loads((c.dir / 'match' / 'validation.json').read_text())
    t = json.loads((c.dir / 'truth.json').read_text())
    n_det = json.loads((c.dir / 'analysis' / 'events.json').read_text())['counts']['events']
    b, u = r['best'], r['uncertainty']
    return [('Events detected', f"{n_det} ({b['n_events']} pass the quality filter, {b['n_explained']} explained "
                                 'by ADS-B flights)'),
            ('Error against the hidden position', f"{v['error_m']:.0f} m (90% bootstrap radius "
                                                   f"{u['bootstrap_90_m']:.0f} m)"),
            ('Clock offset fitted', f"{b['delta_s']:.2f} s ± {u['bootstrap_delta_sd_s']:.2f} s "
                                    f"(true {t['delay_s']:g} s)")]


def write_facts(rows):
    p = ROOT / 'site' / 'index.html'
    s = p.read_text(encoding='utf-8')
    m = re.search(r'(<!-- facts:[^>]*-->\n)(.*?)(\s*<!-- /facts -->)', s, re.S)
    if not m:
        raise SystemExit('site/index.html: facts markers not found')
    keep = [ln for ln in m.group(2).splitlines() if not any(f'<th>{k}</th>' in ln for k, _ in rows)]
    new = [f'    <tr><th>{k}</th><td>{v}</td></tr>' for k, v in rows] + keep
    p.write_text(s[:m.start(2)] + '\n'.join(new) + s[m.end(2):], encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--cases', default=str(ROOT / 'cases'), help='where to keep the example case (default: cases/)')
    ap.add_argument('--rerun', action='store_true', help='render and match again even if a result exists')
    a = ap.parse_args()
    d = Path(a.cases) / CASE_ID
    t = time.time()
    if a.rerun and d.exists():
        shutil.rmtree(d)
    if not (d / 'case.json').exists():
        synth.make_case(d, 'example', seed=1)
    c = case.Case(d)
    pipeline.run_all(c)
    pipeline.validate_synthetic(c)
    out = ROOT / 'site' / 'demo'
    if out.exists():
        shutil.rmtree(out)
    export.export_cases([c], out)
    (ROOT / 'site' / 'vendor').mkdir(exist_ok=True)
    shutil.copyfile(out / 'vendor' / 'plotly.min.js', ROOT / 'site' / 'vendor' / 'plotly.min.js')
    shutil.copyfile(ROOT / 'adoppler' / 'web' / 'favicon.svg', ROOT / 'site' / 'favicon.svg')
    rows = facts(c)
    write_facts(rows)
    for k, v in rows:
        print(f'{k}: {v}')
    print(f'site/ ready in {time.time() - t:.0f} s; preview with: python -m http.server -d site 8000')


if __name__ == '__main__':
    main()
