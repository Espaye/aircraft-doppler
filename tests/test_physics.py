"""Forward model against the closed-form Doppler curve of a straight, level, constant-speed pass."""
import numpy as np
import pytest

from adoppler import adsb, geo, physics


def analytic(t, v, d, h, c):
    """Emission time from (t - tau) c = |p(tau)| with p = (v tau, d, h): a quadratic in tau."""
    r2 = d * d + h * h
    a = c * c - v * v
    tau = (c * c * t - np.sqrt(c ** 4 * t * t - a * (c * c * t * t - r2))) / a
    R = np.sqrt((v * tau) ** 2 + r2)
    return 1 / (1 + v * v * tau / (R * c))


def straight_track(enu, v, d, h, t_cpa=1000.0, span=400.0):
    t = np.arange(t_cpa - span, t_cpa + span, 1.0)
    P = np.stack([v * (t - t_cpa), np.full_like(t, d), np.full_like(t, h)], 1)
    lat, lon, hae = enu.to_geodetic(P)
    return adsb.Track('abc123', 'TEST1', 'C172', None, 'test', t, lat, lon, hae, np.full(len(t), 'geom'))


@pytest.mark.parametrize('v,d,h', [(60.0, 500.0, 300.0), (230.0, 3000.0, 9000.0), (45.0, 0.0, 150.0)])
def test_doppler_matches_closed_form(v, d, h):
    enu = geo.ENU(52.0, 5.0, 0.0)
    tr = straight_track(enu, v, d, h)
    traj = physics.Trajectory(tr, enu, 600, 1400, dt=0.5, geoid_n=0.0)
    c = physics.sound_speed(15.0)
    t_rx = np.linspace(1000 - 150, 1000 + 150, 61) + np.hypot(d, h) / c
    g, R, tau = physics.doppler(traj, [0.0, 0.0, 0.0], t_rx, c)
    expect = analytic(t_rx - 1000, v, d, h, c)
    assert np.allclose(g[0], expect, rtol=2e-5, atol=0)
    assert np.allclose(tau[0] + R[0] / c, t_rx, atol=1e-3)


def test_sound_speed():
    assert physics.sound_speed(0.0) == pytest.approx(331.3)
    assert physics.sound_speed(15.0) == pytest.approx(340.3, abs=0.05)
    # lapse rate: the path to a source 1 km up is slower than at the ground
    assert physics.path_sound_speed(15.0, 0.0, 1000.0) < physics.sound_speed(15.0)


def test_enu_round_trip():
    enu = geo.ENU(-33.9, 151.2, 150.0)
    P = np.array([[0.0, 0.0, 0.0], [12000.0, -8000.0, 3000.0], [-500.0, 250.0, -100.0]])
    lat, lon, h = enu.to_geodetic(P)
    assert np.allclose(enu.from_geodetic(lat, lon, h), P, atol=1e-6)
    assert geo.msl_to_ellipsoid(100.0, -33.0) == pytest.approx(67.0)
