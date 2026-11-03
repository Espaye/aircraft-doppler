"""Chunked high-resolution spectrograms and narrowband (tonal) enhancement.

The WAV is processed in chunks so recordings of any length fit in memory. The output is a float32 power
spectrogram memmap (frames x bins) plus a JSON sidecar with its time/frequency axes.
"""
import json
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import scipy.signal as ss
import soundfile as sf


def analysis_params(samplerate):
    """Decimation and window length giving ~11 kHz and a ~0.74 s window (8192 samples at 44.1 kHz / 4)."""
    decim = max(1, int(round(samplerate / 11025)))
    nperseg = int(round(8192 / 11025 * samplerate / decim / 2)) * 2
    return decim, nperseg


def compute(wav, out_dir, decim=4, nperseg=8192, hop_s=0.25, fmax=2000.0, chunk_s=300.0, name='spec'):
    """Power spectrogram of the channel-averaged signal, after decimation. Resumable: skips if present."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meta_p = out_dir / f'{name}.json'
    if meta_p.exists():
        return load(out_dir, name)
    info = sf.info(str(wav)); fs0 = info.samplerate; fs = fs0 / decim
    hop = int(round(hop_s * fs)); hop_s = hop / fs
    nbin = int(min(fmax, fs / 2) / (fs / nperseg)) + 1
    n_total = info.frames // decim
    n_frames = 1 + (n_total - nperseg) // hop
    if n_frames < 40:
        raise ValueError(f'recording too short ({info.frames / fs0:.1f} s) for a {nperseg / fs:.2f} s analysis window')
    P = np.lib.format.open_memmap(out_dir / f'{name}.npy', 'w+', np.float32, (n_frames, nbin))
    win = ss.get_window('hann', nperseg); scale = 1.0 / (fs * (win ** 2).sum())
    pad = 4096 * decim  # filter settling at chunk edges
    chunk = int(chunk_s * fs0) // (hop * decim) * (hop * decim)
    frame = 0
    pos = 0
    while frame < n_frames:
        a = max(0, pos - pad); b = min(info.frames, pos + chunk + nperseg * decim + pad)
        x, _ = sf.read(str(wav), start=a, stop=b, dtype='float32', always_2d=True)
        y = ss.resample_poly(x.mean(1), 1, decim)
        y = y[(pos - a) // decim:]
        k = min(n_frames - frame, 1 + (len(y) - nperseg) // hop, chunk // (hop * decim))
        idx = np.arange(k)[:, None] * hop + np.arange(nperseg)[None, :]
        F = np.fft.rfft(y[idx] * win, axis=1)[:, :nbin]
        P[frame:frame + k] = (np.abs(F) ** 2 * scale * 2).astype(np.float32)
        frame += k; pos += k * hop * decim
    P.flush()
    meta = {'wav': str(wav), 'fs': fs, 'decim': decim, 'nperseg': nperseg, 'hop_s': hop_s, 'df': fs / nperseg,
            'n_frames': int(n_frames), 'n_bins': int(nbin), 't0': (nperseg / 2) / fs,
            'note': 'frame i centre time (s from WAV start) = t0 + i*hop_s; bin j = j*df Hz'}
    meta_p.write_text(json.dumps(meta, indent=1))
    return load(out_dir, name)


def load(out_dir, name='spec'):
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / f'{name}.json').read_text())
    P = np.load(out_dir / f'{name}.npy', mmap_mode='r')
    t = meta['t0'] + np.arange(meta['n_frames']) * meta['hop_s']
    f = np.arange(meta['n_bins']) * meta['df']
    return P, t, f, meta


def tonal(P, df, out_dir, hop_s=0.25, block_s=20.0, f_halfwidth_hz=20.0, chunk_frames=4800, name='tonal'):
    """Narrowband enhancement in dB, written chunk by chunk to a memmap (frames x bins).

    1. Stationary background per frequency: median over sliding ~40 s blocks (removes hums and fixed tones).
       Z = dB above that background.
    2. Spectral background per frame: median of Z over +/- f_halfwidth (removes broadband transients, wind).
       T = Z - that median, roughly 'dB above the local noise floor' for tonal components.
    Returns memmaps (T, Z). Resumable.
    """
    out_dir = Path(out_dir)
    tp, zp = out_dir / f'{name}_T.npy', out_dir / f'{name}_Z.npy'
    if tp.exists() and zp.exists():
        return np.load(tp, mmap_mode='r'), np.load(zp, mmap_mode='r')
    n, nb_ = P.shape
    nb = max(1, int(round(block_s / hop_s)))
    centers, meds = [], []
    for s0 in range(0, n, nb):
        a, b = max(0, s0 - nb // 2), min(n, s0 + nb + nb // 2)
        centers.append((s0 + min(n, s0 + nb) - 1) / 2)
        meds.append(np.median(10 * np.log10(np.asarray(P[a:b], np.float64) + 1e-20), axis=0))
    centers = np.array(centers); meds = np.array(meds, np.float32)
    T = np.lib.format.open_memmap(out_dir / f'{name}_T.tmp.npy', 'w+', np.float32, (n, nb_))
    Zm = np.lib.format.open_memmap(out_dir / f'{name}_Z.tmp.npy', 'w+', np.float32, (n, nb_))
    w = max(3, int(round(2 * f_halfwidth_hz / df)) | 1)
    for a in range(0, n, chunk_frames):
        b = min(n, a + chunk_frames)
        L = 10 * np.log10(np.asarray(P[a:b], np.float64) + 1e-20)
        fr = np.arange(a, b)
        k = np.clip(np.searchsorted(centers, fr) - 1, 0, max(0, len(centers) - 2))
        if len(centers) > 1:
            u = np.clip((fr - centers[k]) / (centers[k + 1] - centers[k]), 0, 1)[:, None]
            bg = meds[k] * (1 - u) + meds[k + 1] * u
        else:
            bg = meds[[0] * len(fr)]
        Z = (L - bg).astype(np.float32)
        T[a:b] = Z - ndi.median_filter(Z, size=(1, w), mode='nearest')
        Zm[a:b] = Z
    T.flush(); Zm.flush(); del T, Zm
    Path(out_dir / f'{name}_T.tmp.npy').replace(tp); Path(out_dir / f'{name}_Z.tmp.npy').replace(zp)
    return np.load(tp, mmap_mode='r'), np.load(zp, mmap_mode='r')
