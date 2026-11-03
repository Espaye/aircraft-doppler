"""Geodesy helpers: WGS84 <-> local ENU, geoid offset and terrain elevation (flat or USGS 3DEP)."""
from pathlib import Path
import json

import numpy as np
from pyproj import Transformer

_to_ecef = Transformer.from_crs('EPSG:4979', 'EPSG:4978', always_xy=True)
_from_ecef = Transformer.from_crs('EPSG:4978', 'EPSG:4979', always_xy=True)


class ENU:
    """Local east-north-up frame (metres) tangent to the WGS84 ellipsoid at (lat0, lon0, h0)."""

    def __init__(self, lat0, lon0, h0=0.0):
        self.lat0, self.lon0, self.h0 = lat0, lon0, h0
        self.o = np.array(_to_ecef.transform(lon0, lat0, h0))
        la, lo = np.radians(lat0), np.radians(lon0)
        self.R = np.array([[-np.sin(lo), np.cos(lo), 0],
                           [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                           [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])

    def from_geodetic(self, lat, lon, h_ellipsoid):
        x, y, z = _to_ecef.transform(np.asarray(lon, float), np.asarray(lat, float), np.asarray(h_ellipsoid, float))
        d = np.stack([x, y, z], -1) - self.o
        return d @ self.R.T

    def to_geodetic(self, enu):
        enu = np.asarray(enu, float)
        xyz = enu @ self.R + self.o
        lon, lat, h = _from_ecef.transform(xyz[..., 0], xyz[..., 1], xyz[..., 2])
        return lat, lon, h


def msl_to_ellipsoid(h_msl, geoid_n):
    """Orthometric (MSL) height to WGS84 ellipsoidal height with a regional constant geoid undulation N."""
    return np.asarray(h_msl, float) + geoid_n


def geoid_undulation(lat, lon):
    """EGM96 geoid undulation N in metres (h_ellipsoid = h_msl + N) and its source. Uses PROJ's network grid
    (cdn.proj.org, cached in the PROJ user directory); returns (0.0, 'unavailable') when that fails."""
    try:
        import pyproj
        pyproj.network.set_network_enabled(True)
        t = Transformer.from_crs('EPSG:4979', 'EPSG:4326+5773', always_xy=True)
        h = t.transform(lon, lat, 0.0)[2]
        if np.isfinite(h) and h != 0.0:
            return float(-h), 'EGM96 (PROJ grid us_nga_egm96_15)'
    except Exception:
        pass
    return 0.0, 'unavailable: set geoid_undulation_m in case.json (error up to ~100 m in heights)'


class FlatTerrain:
    """Constant ground elevation (m above sea level)."""

    def __init__(self, elevation_m=0.0):
        self.z0 = float(elevation_m)

    def elevation(self, lat, lon):
        lat = np.asarray(lat, float)
        return np.where(np.isfinite(lat), self.z0, np.nan) + 0 * np.asarray(lon, float)


class Terrain3DEP:
    """Bare-earth elevation (m MSL, NAVD88) on a regular lat/lon grid from USGS 3DEP (downloaded once)."""

    def __init__(self, path, bbox=None):
        import tifffile
        path = Path(path)
        if not path.exists():
            if bbox is None:
                raise FileNotFoundError(path)
            fetch_dem(path, bbox)
        self.z = tifffile.imread(path).astype(np.float32)
        meta = json.loads(path.with_suffix('.json').read_text())
        self.lon_min, self.lat_min, self.lon_max, self.lat_max = meta['bbox']
        ny, nx = self.z.shape
        self.dlon = (self.lon_max - self.lon_min) / nx
        self.dlat = (self.lat_max - self.lat_min) / ny
        if not np.isfinite(self.z).any() or np.nanmax(self.z) < -1e4:
            raise ValueError('USGS 3DEP returned no data here (outside the United States?); use terrain "flat"')

    def elevation(self, lat, lon):
        """Bilinear interpolation; NaN outside the tile."""
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        fx = (lon - self.lon_min) / self.dlon - 0.5
        fy = (self.lat_max - lat) / self.dlat - 0.5  # row 0 = north edge
        ny, nx = self.z.shape
        x0 = np.clip(np.floor(fx).astype(int), 0, nx - 2); y0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
        ax, ay = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
        z = self.z
        v = (z[y0, x0] * (1 - ax) * (1 - ay) + z[y0, x0 + 1] * ax * (1 - ay) +
             z[y0 + 1, x0] * (1 - ax) * ay + z[y0 + 1, x0 + 1] * ax * ay)
        out = (fx < -0.5) | (fy < -0.5) | (fx > nx - 0.5) | (fy > ny - 0.5)
        return np.where(out, np.nan, v)


def fetch_dem(path, bbox, size=2000):
    """Download a bare-earth DEM tile (float32 GeoTIFF, EPSG:4326) from the USGS 3DEP ImageServer."""
    import requests
    url = ('https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage'
           f'?bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}&bboxSR=4326&imageSR=4326&size={size},{size}'
           '&format=tiff&pixelType=F32&interpolation=RSP_BilinearInterpolation&f=image')
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    if not r.content.startswith((b'II*\x00', b'MM\x00*')):
        raise RuntimeError(f'USGS 3DEP did not return a GeoTIFF: {r.text[:200]}')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    path.with_suffix('.json').write_text(json.dumps({'bbox': list(bbox), 'source': url, 'size': size}))
