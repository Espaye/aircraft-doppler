"""Unexplained events: report aircraft whose ADS-B track has a coverage gap over the closest approach."""
import numpy as np

from adoppler import adsb, geo, match

ENU = geo.ENU(52.0, 5.0, 0.0)
T_CPA = 10_000.0
EVENT = {'id': 'E01', 'tau_c_utc': T_CPA + 4.0, 'fit': {'v': 35.0, 'd': 350.0, 'tau_c': 900.0}}


def track(ident, v=35.0, h=350.0, gap=(T_CPA - 270, T_CPA + 1000), stop=False, typ='PA18'):
    """Straight pass over the origin at T_CPA, fixes every 8 s, none inside the gap."""
    t = np.arange(T_CPA - 900, T_CPA + 1900, 8.0)
    x = v * (t - T_CPA)
    if stop:                                            # landed in the gap: the next fix is near the last one
        x = np.where(t > gap[1], v * (gap[0] - T_CPA) + v * (t - gap[1]), x)
    keep = (t < gap[0]) | (t > gap[1])
    P = np.stack([x, np.zeros_like(t), np.full_like(t, h)], 1)[keep]
    lat, lon, hae = ENU.to_geodetic(P)
    return adsb.Track(ident, ident.upper(), typ, None, 'test', t[keep], lat, lon, hae, np.full(keep.sum(), 'geom'))


def found(tracks):
    return [g['icao24'] for g in match.coverage_gaps(tracks, EVENT, ENU, 15000.0, 0.0, (-2, 10))]


def test_gap_over_closest_approach_is_reported():
    g = match.coverage_gaps([track('aaa111')], EVENT, ENU, 15000.0, 0.0, (-2, 10))
    assert [x['icao24'] for x in g] == ['aaa111']
    assert g[0]['gap_s'] > 1000 and abs(g[0]['speed_before_mps'] - 35) < 1 and g[0]['doppler_speed_mps'] == 35.0


def test_tracks_that_cannot_be_the_source_are_not_reported():
    assert found([track('nogap', gap=(0, 0))]) == []                     # covered: the normal matcher handles it
    assert found([track('fast', v=120.0)]) == []                        # speed does not match the Doppler speed
    assert found([track('high', h=3000.0)]) == []                       # far above the closest distance
    assert found([track('landed', stop=True)]) == []                    # no progress across the gap
    assert found([track('short', gap=(T_CPA - 60, T_CPA + 40))]) == []   # ~100 s gap: the matcher interpolates it
