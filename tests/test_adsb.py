import json

import numpy as np
import pytest

from adoppler import adsb, config


def test_parse_readsb_altitudes_and_stale_points():
    d = {'icao': 'a1b2c3', 't': 'C172', 'r': 'N123AB', 'desc': 'CESSNA 172', 'timestamp': 1_700_000_000.0,
         'trace': [
             # dt, lat, lon, baro_ft, gs_kt, track, flags, vrate, details, source, geom_ft
             [0.0, 52.0, 5.0, 1000, 100.0, 90.0, 0, 0, {'flight': 'ABC123  '}, 'adsb_icao', 1100],
             [1.0, 52.0, 5.001, 1000, 100.0, 90.0, 1, 0, None, 'adsb_icao', 1100],          # stale: dropped
             [2.0, 52.0, 5.002, 1000, 100.0, 90.0, 0, 0, None, 'adsb_icao', None],           # baro only
             [3.0, 52.0, 5.003, 'ground', None, None, 0, 0, None, 'adsb_icao', None],
         ]}
    tr = adsb.parse_readsb(d, geoid_n=43.0)
    assert tr.callsign == 'ABC123' and tr.icao_type == 'C172'
    assert list(tr.t - 1_700_000_000.0) == [0.0, 2.0, 3.0]
    assert tr.alt_msl_m[0] == pytest.approx(1100 * config.FT - 43.0)   # GNSS height (HAE) to MSL
    assert tr.alt_kind[1] == 'baro' and tr.alt_msl_m[1] == pytest.approx(1000 * config.FT)
    assert tr.alt_kind[2] == 'ground'


def test_csv_file(tmp_path):
    p = tmp_path / 'flights.csv'
    p.write_text('icao24,time,lat,lon,alt_m,alt_ref,callsign\n'
                 'AB12CD,2026-06-01T12:00:10Z,52.01,5.01,510,hae,TEST1\n'
                 'ab12cd,2026-06-01T12:00:00Z,52.00,5.00,500,hae,TEST1\n'
                 'ffffff,1780315200,52.1,5.1,900,msl,\n', encoding='utf-8')
    tracks = {t.icao24: t for t in adsb.load_file(p, geoid_n=40.0)}
    a = tracks['ab12cd']
    assert np.all(np.diff(a.t) > 0)                  # sorted by time
    assert a.alt_msl_m[0] == pytest.approx(460.0)    # HAE 500 m - geoid 40 m
    assert tracks['ffffff'].alt_msl_m[0] == pytest.approx(900.0)


def test_csv_missing_columns_explains_format(tmp_path):
    p = tmp_path / 'bad.csv'
    p.write_text('icao,when,lat,lon\nab,1,2,3\n', encoding='utf-8')
    with pytest.raises(ValueError, match='CSV columns'):
        adsb.load_file(p, geoid_n=0.0)


def test_tracks_json_round_trip(tmp_path):
    t = np.array([0.0, 1.0, 2.0])
    tr = adsb.Track('abc', 'X', None, None, 'test', t, t + 50, t + 5, t * 10, np.array(['geom'] * 3), meta={'k': 1})
    adsb.save_tracks([tr], tmp_path / 'tracks.json')
    back = adsb.load_tracks(tmp_path / 'tracks.json')[0]
    assert back.icao24 == 'abc' and np.allclose(back.alt_msl_m, tr.alt_msl_m) and back.meta == {'k': 1}
    json.loads((tmp_path / 'tracks.json').read_text())
