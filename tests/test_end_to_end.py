"""Render a short synthetic recording, run the whole pipeline blind and compare with the hidden position.

This proves internal consistency only: the renderer and the matcher share the same physics."""
import json

import pytest

from adoppler import case, cli, export, pipeline, synth


@pytest.fixture(scope='module')
def mini(tmp_path_factory):
    d = tmp_path_factory.mktemp('cases') / 'mini'
    synth.make_case(d, 'mini', seed=1, log=lambda *a: None)
    c = case.Case(d)
    pipeline.run_all(c, log=lambda *a: None)
    return c


@pytest.mark.slow
def test_blind_estimate_finds_hidden_microphone(mini):
    r = json.loads(mini.result_path.read_text())
    assert r['assessment']['verdict'] in ('ok', 'weak')
    assert r['best']['n_explained'] >= 2
    truth = json.loads((mini.dir / 'truth.json').read_text())
    assert 'mic_lat' in truth and 'mic_lat' not in (mini.dir / 'case.json').read_text()   # truth kept apart
    v = pipeline.validate_synthetic(mini, log=lambda *a: None)
    assert v['prediction_unchanged']
    assert v['error_m'] < max(100.0, 2 * v['bootstrap_90_m'])


@pytest.mark.slow
def test_rerun_without_force_keeps_the_stamped_prediction(mini):
    before = mini.result_path.read_bytes()
    pipeline.match(mini, log=lambda *a: None)
    assert mini.result_path.read_bytes() == before


@pytest.mark.slow
def test_cli_status_and_export(mini, tmp_path, capsys):
    assert cli.main(['status', str(mini.dir)]) in (None, 0)
    assert 'match' in capsys.readouterr().out
    out = export.export_cases([mini], tmp_path / 'site', audio=False, log=lambda *a: None)
    files = {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}
    assert {'index.html', 'app.js', 'vendor/plotly.min.js', 'data/cases.json', 'data/mini/result.json',
            'data/mini/spec_tonal.png'} <= files
    assert 'class="static"' in (out / 'index.html').read_text(encoding='utf-8')
    assert str(mini.dir) not in (out / 'data' / 'mini' / 'case.json').read_text(encoding='utf-8')


def test_export_refuses_real_recordings(tmp_path):
    (tmp_path / 'case.json').write_text(json.dumps({'audio': 'a.wav', 'audio_start_utc': '2026-06-01T12:00:00Z',
                                                    'search_center': [52.0, 5.0]}))
    with pytest.raises(ValueError, match='not synthetic'):
        export.export_cases([case.Case(tmp_path)], tmp_path / 'out')
