"""Acoustic forward model: moving source (aircraft from ADS-B), stationary microphone.

Model (straight rays, still air, sound speed c):
    reception time  t = tau + |p(tau) - o| / c          (solved for emission time tau by Newton iteration)
    observed freq   f(t) = f0 / (1 + v_r(tau) / c)       with v_r = v(tau) . (p(tau) - o) / |p(tau) - o|
f0 (emitted frequency of each harmonic) is unknown and is eliminated per track in the matching stage.
Positions are in a local ENU frame (metres) with WGS84 ellipsoidal heights converted from MSL.
Known simplifications: no wind, no refraction (sound speed taken at the mean height of the path), simplified
ADS-B paths are interpolated and smoothed, so velocity errors of a few m/s remain.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import config, geo


def sound_speed(t_c):
    return 331.3 * np.sqrt(1 + t_c / 273.15)


def path_sound_speed(t_ground_c, h_ground_m, h_src_m):
    """Sound speed averaged over a straight path with a standard lapse rate (harmonic mean ~ arithmetic here)."""
    z_mid = 0.5 * (h_ground_m + h_src_m) - h_ground_m
    return sound_speed(t_ground_c - config.LAPSE_K_PER_M * np.maximum(z_mid, 0))


class Trajectory:
    """Aircraft trajectory on a uniform time grid in ENU, with smoothed velocity. NaN where data has gaps."""

    def __init__(self, track, enu, t_from, t_to, dt=0.5, smooth_s=None, max_gap_s=150.0, geoid_n=0.0):
        t = np.asarray(track.t, float)
        ok = np.isfinite(track.alt_msl_m) & np.isfinite(track.lat)
        t = t[ok]
        if len(t) < 2:
            raise ValueError('track has < 2 valid points')
        P = enu.from_geodetic(track.lat[ok], track.lon[ok], geo.msl_to_ellipsoid(track.alt_msl_m[ok], geoid_n))
        o = np.argsort(t); t, P = t[o], P[o]
        t, u = np.unique(t, return_index=True); P = P[u]
        a, b = max(t_from, t[0]), min(t_to, t[-1])
        if b - a < 10:
            raise ValueError('track does not cover the window')
        self.dt = dt
        self.t0 = a
        g = np.arange(a, b + dt / 2, dt)
        X = np.stack([np.interp(g, t, P[:, k]) for k in range(3)], 1)
        # smoothing: simplified (aero) paths are piecewise linear with sparse vertices -> smooth more
        med_dt = float(np.median(np.diff(t))) if len(t) > 2 else 10.0
        if smooth_s is None:
            smooth_s = float(np.clip(0.7 * med_dt, 1.5, 8.0))
        sig = smooth_s / dt
        Xs = np.stack([gaussian_filter1d(X[:, k], sig, mode='nearest') for k in range(3)], 1)
        V = np.gradient(Xs, dt, axis=0)
        # gap mask: grid points farther than max_gap_s/2 from any real point within a long gap
        i = np.searchsorted(t, g)
        prev = t[np.clip(i - 1, 0, len(t) - 1)]; nxt = t[np.clip(i, 0, len(t) - 1)]
        bad = (nxt - prev) > max_gap_s
        Xs[bad] = np.nan; V[bad] = np.nan
        self.t, self.P, self.V = g, Xs, V
        self.smooth_s, self.med_dt = smooth_s, med_dt

    def at(self, tau):
        """Linear interpolation of position and velocity; NaN outside the grid."""
        x = (np.asarray(tau) - self.t0) / self.dt
        x = np.where(np.isfinite(x), x, -1.0)
        i = np.floor(x).astype(int)
        out = (i < 0) | (i >= len(self.t) - 1)
        i = np.clip(i, 0, len(self.t) - 2)
        w = (x - i)[..., None]
        P = self.P[i] * (1 - w) + self.P[i + 1] * w
        V = self.V[i] * (1 - w) + self.V[i + 1] * w
        P[out] = np.nan; V[out] = np.nan
        return P, V


def doppler(traj, obs, t_rx, c, n_iter=4):
    """Observed/emitted frequency ratio, range and elevation for observers obs (M,3) at reception times t_rx (K,).

    Returns g (M,K), R (M,K) metres, emission time tau (M,K)."""
    obs = np.asarray(obs, float).reshape(-1, 1, 3)
    t_rx = np.asarray(t_rx, float).reshape(1, -1)
    P, _ = traj.at(np.broadcast_to(t_rx, (obs.shape[0], t_rx.shape[1])))
    tau = t_rx - np.linalg.norm(P - obs, axis=-1) / c
    for _ in range(n_iter):
        P, V = traj.at(tau)
        D = P - obs
        R = np.linalg.norm(D, axis=-1)
        vr = np.einsum('mkj,mkj->mk', V, D) / R
        F = tau + R / c - t_rx
        tau = tau - F / (1 + vr / c)
    P, V = traj.at(tau)
    D = P - obs
    R = np.linalg.norm(D, axis=-1)
    vr = np.einsum('mkj,mkj->mk', V, D) / R
    return 1.0 / (1.0 + vr / c), R, tau


def simulate_track(traj, obs, t_rx, f0, c):
    g, R, tau = doppler(traj, np.asarray(obs).reshape(1, 3), t_rx, c)
    return f0 * g[0], R[0]
