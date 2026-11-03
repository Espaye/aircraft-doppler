"""The local server must refuse anything a web page on another origin could send."""
import http.client, http.server, json, threading

import pytest

from adoppler import server


@pytest.fixture()
def srv(tmp_path):
    (tmp_path / 'cases').mkdir()
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
    server.Handler.root, server.Handler.jobs = (tmp_path / 'cases').resolve(), server.Jobs()
    server.Handler.port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown(); httpd.server_close()


def req(port, method, path, body=None, headers=None, host=None):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    h = {'Host': host or f'127.0.0.1:{port}', **(headers or {})}
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    if isinstance(body, dict):
        h['Content-Type'] = 'application/json'
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    out = r.status, dict(r.getheaders()), r.read()
    c.close()
    return out


W = {'X-Adoppler': '1'}


def test_index_has_security_headers(srv):
    st, h, body = req(srv, 'GET', '/')
    assert st == 200 and b'adoppler' in body
    assert "frame-ancestors 'none'" in h['Content-Security-Policy'] and h['X-Content-Type-Options'] == 'nosniff'


def test_foreign_host_header_is_refused(srv):            # DNS rebinding
    assert req(srv, 'GET', '/api/cases', host='evil.example:80')[0] == 421


def test_writes_need_header_and_local_origin(srv):
    body = {'scene': 'mini'}
    assert req(srv, 'POST', '/api/synth', body)[0] == 403
    assert req(srv, 'POST', '/api/synth', body, {**W, 'Origin': 'https://evil.example'})[0] == 403


def test_case_ids_cannot_escape_the_cases_folder(srv):
    for p in ('/api/cases/..%2f..%2fetc', '/api/cases/Upper', '/api/cases/a%5cb', '/api/cases/..'):
        assert req(srv, 'GET', p)[0] in (400, 404)


def test_create_case_upload_and_errors(srv):
    st, _, b = req(srv, 'POST', '/api/cases', {'name': 'My test', 'audio': 'x.exe'}, W)
    assert st == 400 and b'audio must be one of' in b
    st, _, b = req(srv, 'POST', '/api/cases', {'name': 'My test', 'audio': 'rec.wav', 'search_center': [52, 5],
                                                'audio_start_utc': '2026-06-01T12:00:00Z', 'evil': 1}, W)
    assert st == 201
    cid = json.loads(b)['id']
    assert cid == 'my-test'
    st, _, b = req(srv, 'GET', f'/api/cases/{cid}')
    assert st == 200 and 'evil' not in json.loads(b)['settings'] and 'audio_error' in json.loads(b)
    assert req(srv, 'PUT', f'/api/cases/{cid}/upload?kind=audio&name=a.html', b'x', W)[0] == 400
    assert req(srv, 'GET', f'/api/cases/{cid}/result')[0] == 409
    st, _, b = req(srv, 'POST', f'/api/cases/{cid}/run', {'stage': 'analyse'}, W)
    assert st == 202
    assert req(srv, 'POST', f'/api/cases/{cid}/run', {'stage': 'rm -rf'}, W)[0] == 400
