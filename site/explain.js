'use strict';
// Interactive explainers. Same physics as adoppler (straight-line flight, emission-time solve, f = f0 / (1 + v_r / c)),
// reduced to flat ground, still air and a constant sound speed.

const C = ['#1f77b4', '#e8730c', '#2a9d50', '#d62728', '#8a56c2'];
const speedOfSound = (T) => 331.3 * Math.sqrt(1 + T / 273.15);
const cfg = { responsive: true, displaylogo: false, displayModeBar: false };
const font = { family: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif', size: 13, color: '#18202a' };

// Received frequency at reception time t for an aircraft at pos(tau) = P + V (tau - tc), height h, mic at (e, n, 0).
function received(t, ac, e, n, c) {
  const [px, py] = ac.P, [vx, vy] = ac.V;
  let tau = t - Math.hypot(px - e, py - n, ac.h) / c;
  for (let k = 0; k < 6; k++) {
    const x = px + vx * (tau - ac.tc) - e, y = py + vy * (tau - ac.tc) - n;
    const R = Math.hypot(x, y, ac.h), vr = (x * vx + y * vy) / R;
    const step = (tau + R / c - t) / (1 + vr / c);
    tau -= step;
    if (Math.abs(step) < 1e-6) break;
  }
  const x = px + vx * (tau - ac.tc) - e, y = py + vy * (tau - ac.tc) - n;
  const R = Math.hypot(x, y, ac.h), vr = (x * vx + y * vy) / R;
  return ac.f0 / (1 + vr / c);
}

// ---------------------------------------------------------------- 1. one pass

function pass() {
  const f = document.getElementById('pass');
  const v = +f.v.value / 3.6, d = +f.d.value, h = +f.h.value, f0 = +f.f0.value, T = +f.T.value, c = speedOfSound(T);
  f.v_o.value = `${f.v.value} km/h`; f.d_o.value = `${d} m`; f.h_o.value = `${h} m`; f.f_o.value = `${f0} Hz`; f.T_o.value = `${T} °C`;
  const ac = { P: [0, d], V: [v, 0], h, tc: 0, f0 };
  const R0 = Math.hypot(d, h), span = Math.max(60, Math.min(240, (6 * R0) / v));
  const t = [], y = [];
  for (let k = 0; k <= 400; k++) { const tt = -span + (2 * span * k) / 400; t.push(tt); y.push(received(tt + R0 / c, ac, 0, 0, c)); }
  const hi = f0 / (1 - v / c), lo = f0 / (1 + v / c);
  document.getElementById('pass-readout').textContent =
    `Sound speed ${c.toFixed(1)} m/s. Tone far ahead ${hi.toFixed(1)} Hz, far behind ${lo.toFixed(1)} Hz ` +
    `(swing ${((100 * (hi - lo)) / f0).toFixed(1)} %). The closest point (${R0.toFixed(0)} m) is heard ${(R0 / c).toFixed(1)} s after the aircraft is there.`;
  Plotly.react('pass-plot', [
    { x: t, y, mode: 'lines', line: { color: C[0], width: 3 }, name: 'received tone', hovertemplate: '%{y:.1f} Hz at %{x:.0f} s<extra></extra>' },
    { x: [0, 0], y: [lo, hi], mode: 'lines', line: { color: '#9aa5b1', dash: 'dot', width: 1 }, hoverinfo: 'skip' },
  ], {
    margin: { l: 55, r: 15, t: 15, b: 45 }, showlegend: false, font,
    xaxis: { title: { text: 'seconds from hearing the closest approach' }, zeroline: false },
    yaxis: { title: { text: 'frequency (Hz)' }, range: [lo * 0.97, hi * 1.03] },
  }, cfg);
}

// ---------------------------------------------------------------- 2. many passes

const SCENE = {
  c: speedOfSound(15),
  truth: { e: 640, n: -360, dt: 2.0 },
  aircraft: [
    { name: 'A, 220 km/h at 700 m', P: [0, 900], V: [61, 0], h: 700, tc: 70, f0: 96 },
    { name: 'B, 260 km/h at 1200 m', P: [-700, 0], V: [8, -71], h: 1200, tc: 170, f0: 121 },
    { name: 'C, 250 km/h at 900 m', P: [1500, -1300], V: [-49, -49], h: 900, tc: 270, f0: 148 },
  ],
};

// Fixed "recording": the hidden truth plus 0.12 % frequency noise (deterministic, so every visitor sees the same data).
let seed = 7;
const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
for (const ac of SCENE.aircraft) {
  ac.t = []; ac.f = [];
  for (let t = ac.tc - 70; t <= ac.tc + 75; t += 2.5) {
    const { e, n, dt } = SCENE.truth;
    ac.t.push(t); ac.f.push(received(t - dt, ac, e, n, SCENE.c) * (1 + 0.0012 * gauss()));
  }
}

function mismatch(e, n, dt) {     // rms of the relative frequency error per aircraft, in %
  return SCENE.aircraft.map((ac) => {
    let s = 0;
    for (let k = 0; k < ac.t.length; k++) { const r = Math.log(ac.f[k] / received(ac.t[k] - dt, ac, e, n, SCENE.c)); s += r * r; }
    return 100 * Math.sqrt(s / ac.t.length);
  });
}

let revealed = false, costGrid = null;

function mic() {
  const f = document.getElementById('mic');
  const e = +f.e.value, n = +f.n.value, dt = +f.dt.value;
  f.e_o.value = `${e} m`; f.n_o.value = `${n} m`; f.dt_o.value = `${dt.toFixed(1)} s`;
  const m = mismatch(e, n, dt);
  const fit = m.every((x) => x < 0.25);
  const ro = document.getElementById('mic-readout');
  ro.textContent = `Mismatch: ${m.map((x, i) => `${'ABC'[i]} ${x.toFixed(2)} %`).join(', ')}. ` +
    (fit ? `All three curves fit: every position where this holds lies within about 60 m of the hidden one.`
      : m.some((x) => x < 0.25) ? 'Some curves fit, not all: keep moving.' : 'None of the curves fit yet.');
  ro.classList.toggle('good', fit);

  const map = [];
  if (costGrid) map.push({ type: 'contour', x: costGrid.x, y: costGrid.y, z: costGrid.z, showscale: false, hoverinfo: 'skip',
    colorscale: [[0, '#0b3954'], [0.25, '#4f8fbf'], [1, '#f3f6f9']], contours: { coloring: 'heatmap', showlines: false }, opacity: 0.8 });
  SCENE.aircraft.forEach((ac, i) => {
    const x = [], y = [];
    for (let t = ac.tc - 110; t <= ac.tc + 110; t += 10) { x.push(ac.P[0] + ac.V[0] * (t - ac.tc)); y.push(ac.P[1] + ac.V[1] * (t - ac.tc)); }
    map.push({ x, y, mode: 'lines', line: { color: C[i], width: 3 }, name: ac.name, hovertemplate: `${ac.name}<extra></extra>` });
    map.push({ x: [x[x.length - 1]], y: [y[y.length - 1]], mode: 'markers', hoverinfo: 'skip',
      marker: { color: C[i], size: 11, symbol: 'triangle-up', angle: (Math.atan2(ac.V[0], ac.V[1]) * 180) / Math.PI } });
  });
  map.push({ x: [e], y: [n], mode: 'markers', marker: { size: 14, color: '#d62728', line: { color: '#fff', width: 2 } },
    name: 'your microphone', hovertemplate: 'your microphone<extra></extra>' });
  if (revealed) map.push({ x: [SCENE.truth.e], y: [SCENE.truth.n], mode: 'markers', name: 'hidden position',
    marker: { size: 14, color: '#2a9d50', symbol: 'star', line: { color: '#fff', width: 1 } }, hovertemplate: 'hidden position<extra></extra>' });
  Plotly.react('mic-map', map, {
    margin: { l: 45, r: 10, t: 10, b: 40 }, showlegend: false, font, plot_bgcolor: '#f7f9fb',
    xaxis: { title: { text: 'east (m)' }, range: [-4200, 4200], zeroline: false, constrain: 'domain' },
    yaxis: { title: { text: 'north (m)' }, range: [-4200, 4200], zeroline: false, scaleanchor: 'x' },
  }, cfg);

  const curves = [];
  SCENE.aircraft.forEach((ac, i) => {
    curves.push({ x: ac.t, y: ac.f, mode: 'markers', marker: { color: C[i], size: 5 }, name: `${'ABC'[i]} recorded`,
      hovertemplate: `${'ABC'[i]} recorded %{y:.1f} Hz<extra></extra>` });
    const t = [], y = [];
    for (let tt = ac.tc - 70; tt <= ac.tc + 75; tt += 1) { t.push(tt); y.push(received(tt - dt, ac, e, n, SCENE.c)); }
    curves.push({ x: t, y, mode: 'lines', line: { color: C[i], width: 2 }, name: `${'ABC'[i]} predicted`,
      hovertemplate: `${'ABC'[i]} predicted %{y:.1f} Hz<extra></extra>` });
  });
  Plotly.react('mic-curves', curves, {
    margin: { l: 55, r: 10, t: 10, b: 40 }, showlegend: false, font,
    xaxis: { title: { text: 'recording time (s)' } }, yaxis: { title: { text: 'frequency (Hz)' } },
  }, cfg);
}

// Grid search like adoppler's coarse stage: best clock offset per cell, then a finer local search.
function solve() {
  const best = { s: Infinity }, x = [], y = [], z = [];
  const total = (e, n, dt) => mismatch(e, n, dt).reduce((a, b) => a + Math.min(b, 3) ** 2, 0);
  for (let n = -4000; n <= 4000; n += 200) {
    const row = [];
    y.push(n);
    for (let e = -4000; e <= 4000; e += 200) {
      if (n === -4000) x.push(e);
      let s = Infinity, dtb = 0;
      for (let dt = -6; dt <= 6; dt += 1) { const v = total(e, n, dt); if (v < s) { s = v; dtb = dt; } }
      row.push(Math.log10(s + 0.01));
      if (s < best.s) Object.assign(best, { s, e, n, dt: dtb });
    }
    z.push(row);
  }
  costGrid = { x, y, z };
  for (const [step, de, ddt] of [[50, 300, 1], [25, 75, 0.2]]) {
    const c0 = { ...best };
    for (let n = c0.n - de; n <= c0.n + de; n += step)
      for (let e = c0.e - de; e <= c0.e + de; e += step)
        for (let dt = c0.dt - ddt; dt <= c0.dt + ddt + 1e-9; dt += ddt / 5) {
          const v = total(e, n, dt);
          if (v < best.s) Object.assign(best, { s: v, e, n, dt });
        }
  }
  const f = document.getElementById('mic');
  f.e.value = best.e; f.n.value = best.n; f.dt.value = best.dt.toFixed(1);
  mic();
}

function init() {
  if (!window.Plotly) return;
  document.getElementById('pass').addEventListener('input', pass);
  document.getElementById('mic').addEventListener('input', mic);
  document.getElementById('mic-solve').addEventListener('click', (ev) => {
    ev.target.textContent = 'Searching…'; ev.target.disabled = true;
    setTimeout(() => { solve(); ev.target.textContent = 'Search for me'; ev.target.disabled = false; }, 30);
  });
  document.getElementById('mic-reveal').addEventListener('click', () => {
    revealed = !revealed;
    document.getElementById('mic-reveal').textContent = revealed ? 'Hide' : 'Reveal';
    mic();
  });
  pass(); mic();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
