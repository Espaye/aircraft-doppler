'use strict';
// adoppler browser UI. Live mode talks to `adoppler serve`; static mode (body.static) reads exported JSON files.

const STATIC = document.body.classList.contains('static');
const COLORS = ['#1f77b4', '#e8730c', '#2a9d50', '#d62728', '#8a56c2', '#8c564b', '#d4489c', '#139fb0', '#a3a312', '#5b6b7b'];
const $ = (s) => document.querySelector(s);
const state = { cases: [], id: null, c: null, events: null, flights: null, result: null, sel: null, specReq: 0 };

// ------------------------------------------------------------------ helpers

function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (k in e && k !== 'list') e[k] = v;
    else e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) e.append(k instanceof Node ? k : String(k));
  return e;
}
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (x, d = 0) => (x == null || !isFinite(x) ? '–' : Number(x).toFixed(d));
const utc = (s) => new Date(s * 1000).toISOString().slice(11, 19);
const iso = (s) => new Date(s * 1000).toISOString();
const dist = (m) => (m == null ? '–' : m >= 1000 ? `${fmt(m / 1000, 1)} km` : `${fmt(m)} m`);
const radius = (m) => (m == null ? '–' : m < 25 ? 'within one 25 m grid cell' : `${fmt(m)} m`);
const colorOf = (id) => COLORS[(state.events?.events.findIndex((e) => e.id === id) ?? 0) % COLORS.length];

async function api(path, opts = {}) {
  const o = { ...opts, headers: { ...(opts.headers || {}) } };
  if (o.body && typeof o.body !== 'string' && !(o.body instanceof Blob)) {
    o.body = JSON.stringify(o.body); o.headers['Content-Type'] = 'application/json';
  }
  if (o.method && o.method !== 'GET') o.headers['X-Adoppler'] = '1';
  const r = await fetch(path, o);
  const ct = r.headers.get('Content-Type') || '';
  const data = ct.includes('json') ? await r.json() : r;
  if (!r.ok) { const e = new Error((data && data.error) || `${r.status} ${r.statusText}`); e.status = r.status; e.data = data; throw e; }
  return data;
}

const src = {
  cases: () => (STATIC ? api('data/cases.json') : api('api/cases')),
  case: (id) => (STATIC ? api(`data/${id}/case.json`) : api(`api/cases/${id}`)),
  events: (id) => (STATIC ? api(`data/${id}/events.json`) : api(`api/cases/${id}/events`)),
  flights: (id) => (STATIC ? api(`data/${id}/flights.json`) : api(`api/cases/${id}/flights`)),
  result: (id) => (STATIC ? api(`data/${id}/result.json`) : api(`api/cases/${id}/result`)),
  async spec(id, kind, r) {
    if (STATIC) { const ext = await api(`data/${id}/spec_${kind}.json`); return { url: `data/${id}/spec_${kind}.png`, ext }; }
    const q = new URLSearchParams({ kind, w: 1800, h: 700 });
    if (r) { q.set('t0', r.t0); q.set('t1', r.t1); q.set('f0', Math.max(0, r.f0)); q.set('f1', r.f1); }
    const resp = await fetch(`api/cases/${id}/spectrogram?${q}`);
    if (!resp.ok) throw new Error((await resp.json()).error);
    return { url: URL.createObjectURL(await resp.blob()), ext: JSON.parse(resp.headers.get('X-Extent')) };
  },
  audio: (id, ev) => (STATIC ? `data/${id}/audio/${ev.id}.mp3`
    : `api/cases/${id}/audio?t0=${Math.max(0, ev.t_start - 5)}&t1=${ev.t_end + 5}`),
};

// ------------------------------------------------------------------ case list and selection

async function loadCases(select) {
  state.cases = await src.cases();
  const s = $('#case-select');
  s.replaceChildren(...state.cases.map((c) => el('option', { value: c.id }, c.name || c.id)));
  const want = select || new URLSearchParams(location.hash.slice(1)).get('case') || state.id || state.cases[0]?.id;
  $('#empty').classList.toggle('hidden', state.cases.length > 0);
  $('#case').classList.toggle('hidden', !state.cases.length);
  if (state.cases.length) { s.value = state.cases.some((c) => c.id === want) ? want : state.cases[0].id; await openCase(s.value); }
}

async function openCase(id) {
  if (state.id !== id) { state.sel = null; state.events = state.flights = state.result = null; }
  state.id = id;
  history.replaceState(null, '', `#case=${encodeURIComponent(id)}`);
  const c = await src.case(id);
  state.c = c;
  renderHeader();
  const st = c.status || {};
  const [ev, fl, res] = await Promise.all([
    st.events ? src.events(id).catch(() => null) : null,
    st.adsb ? src.flights(id).catch(() => null) : null,
    st.match ? src.result(id).catch(() => null) : null]);
  if (state.id !== id) return;
  state.events = ev; state.flights = fl; state.result = res;
  renderResult(); renderEvents(); drawSpec(); drawMap();
  const run = (c.jobs || []).find((j) => j.state === 'running');
  if (run && !STATIC) follow(run.id);
}

function renderHeader() {
  const c = state.c, s = c.settings, st = c.status || {};
  $('#case-title').textContent = c.name;
  const a = c.audio_info;
  $('#case-meta').textContent = [
    a ? `${fmt(a.duration_s / 60, 1)} min of audio at ${a.samplerate} Hz` : c.audio_error || 'audio missing',
    `starts ${s.audio_start_utc}`,
    `search ${fmt(s.search_radius_km)} km around ${s.search_center.map((x) => fmt(x, 3)).join(', ')}`,
    s.synthetic ? 'synthetic' : null].filter(Boolean).join(' · ');
  const done = { analyse: st.events, adsb: st.adsb, match: st.match };
  const running = (c.jobs || []).find((j) => j.state === 'running');
  for (const li of document.querySelectorAll('.stages li')) {
    const k = li.dataset.stage;
    const failed = (c.jobs || []).filter((j) => j.stage === k || j.stage === 'all').slice(-1)[0]?.state === 'failed';
    li.className = running && (running.stage === k || running.stage === 'all') ? 'running' : done[k] ? 'done' : failed ? 'failed' : '';
    li.querySelector('button').textContent = done[k] ? 'Rerun' : 'Run';
    li.querySelector('button').disabled = !!running;
  }
  $('#btn-all').disabled = !!running;
}

// ------------------------------------------------------------------ running stages

async function runStage(stage) {
  const st = state.c.status || {};
  const force = $('#force').checked || (stage !== 'all' && !!st[{ analyse: 'events', adsb: 'adsb', match: 'match' }[stage]]);
  if (force && st.validated &&
      !window.confirm('Recomputing replaces the stored blind prediction and removes its validation. Continue?')) return;
  try {
    const job = await api(`api/cases/${state.id}/run`, { method: 'POST', body: { stage, force: !!force } });
    $('#log').textContent = ''; follow(job.id);
  } catch (e) { showLog(`error: ${e.message}`); }
}

function showLog(line) {
  const log = $('#log');
  log.classList.remove('hidden');
  log.textContent += line + '\n';
  log.scrollTop = log.scrollHeight;
}

async function follow(jobId, onDone) {
  let since = 0;
  const caseId = state.id;
  await openCaseQuiet(caseId);
  for (;;) {
    let j;
    try { j = await api(`api/jobs/${jobId}?since=${since}`); } catch (e) { showLog(`lost the job: ${e.message}`); return; }
    j.log.forEach(showLog); since = j.log_total;
    if (j.state !== 'running') {
      if (j.state === 'failed') showLog(`failed: ${j.error}`); else showLog('done');
      if (onDone) await onDone(j); else if (state.id === caseId) await openCase(caseId);
      return;
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
}

async function openCaseQuiet(id) {
  try { state.c = await src.case(id); renderHeader(); } catch (e) { /* case may not exist yet (synthetic) */ }
}

// ------------------------------------------------------------------ result card

function resultEvent(id) { return state.result?.result.events.find((e) => e.id === id); }

function renderResult() {
  const box = $('#result-body');
  const R = state.result?.result;
  if (!R) {
    box.replaceChildren(el('p', { class: 'muted' }, state.c.status?.events
      ? 'No location yet. Fetch flight data and run the match.' : 'No result yet. Run the three stages.'));
    return;
  }
  const b = R.best, u = R.uncertainty, a = R.assessment, nt = R.null_test;
  const nullBest = nt.shifted.length ? Math.min(...nt.shifted.map((x) => x.total)) : null;
  const loo = Math.max(0, ...u.loo.map((x) => x.shift_m));
  const kids = [
    el('p', {}, el('span', { class: `verdict ${a.verdict}` }, a.verdict), ' ', a.summary),
    a.flags.length ? el('ul', { class: 'flags' }, a.flags.map((f) => el('li', {}, f.message))) : null,
    el('dl', { class: 'kv' },
      el('dt', {}, 'Estimate'), el('dd', {}, `${fmt(b.lat, 5)}, ${fmt(b.lon, 5)}`),
      el('dt', {}, 'Uncertainty'), el('dd', {}, `68% ${radius(u.bootstrap_68_m)}, 90% ${radius(u.bootstrap_90_m)} (event bootstrap, ${u.bootstrap_units} aircraft)`),
      el('dt', {}, 'Leave-one-out'), el('dd', {}, `largest shift ${dist(loo)}`),
      el('dt', {}, 'Events explained'), el('dd', {}, `${b.n_explained} of ${b.n_events}`),
      el('dt', {}, 'Clock offset'), el('dd', {}, `${fmt(b.delta_s, 2)} s ± ${fmt(u.bootstrap_delta_sd_s, 2)} (recording time minus acoustic time)`),
      el('dt', {}, 'Ground temperature'), el('dd', {}, `${fmt(b.ground_temp_c, 1)} °C fitted (prior ${fmt(R.settings.ground_temp_prior_c, 1)} °C)`),
      el('dt', {}, 'Null test'), el('dd', {}, nullBest == null ? 'not run'
        : `real data cost ${fmt(nt.real_coarse_total, 1)} vs ${fmt(nullBest, 1)} or more with time-shifted events`)),
  ];
  const v = state.result.validation;
  if (v) {
    kids.push(el('div', { class: 'detail' }, el('b', {}, 'Validation: '),
      `error ${dist(v.error_m)} (${v.within_bootstrap_90 ? 'inside' : 'outside'} the 90% radius); reference: ${v.reference.source}`,
      v.prediction_unchanged ? null : el('span', { class: 'error' }, ' — the prediction file changed after it was stamped')));
  } else if (!STATIC) {
    kids.push(el('p', { class: 'row' },
      state.c.settings.synthetic ? el('button', { class: 'ghost small', onclick: validateSynthetic }, 'Compare with the hidden truth')
        : el('button', { class: 'ghost small', onclick: () => $('#dlg-validate').showModal() }, 'Validate against a reference…')));
  }
  kids.push(el('p', { class: 'note' }, a.note + ' ' + (R.blind || '')));
  box.replaceChildren(...kids.filter(Boolean));
}

async function validateSynthetic() {
  try { await api(`api/cases/${state.id}/validate`, { method: 'POST', body: { synthetic: true } }); await openCase(state.id); }
  catch (e) { showLog(`error: ${e.message}`); }
}

// ------------------------------------------------------------------ events table and detail

function renderEvents() {
  const tb = $('#events tbody');
  if (!state.events) { tb.replaceChildren(el('tr', {}, el('td', { colSpan: 10, class: 'muted' }, 'Run the detection to list Doppler events.'))); return; }
  const t0 = state.events.t0_utc;
  tb.replaceChildren(...state.events.events.map((e) => {
    const r = resultEvent(e.id), f = e.fit || {};
    const used = el('input', { type: 'checkbox', checked: e.used, disabled: STATIC, title: 'use this event in the match',
      onclick: (ev) => { ev.stopPropagation(); toggleUsed(e.id, ev.target.checked); } });
    const row = el('tr', { class: [state.sel === e.id ? 'sel' : '', e.used ? '' : 'unused'].join(' '), onclick: () => selectEvent(e.id) },
      el('td', {}, el('span', { class: 'swatch', style: `background:${colorOf(e.id)}` }), e.id, e.manual ? ' ✎' : ''),
      el('td', {}, f.t_cpa != null ? utc(t0 + f.t_cpa) : `${utc(t0 + e.t_start)}`),
      el('td', {}, e.label), el('td', {}, fmt(e.score, 2)),
      el('td', {}, f.v ? `${fmt(f.v * 3.6)} km/h` : '–'), el('td', {}, dist(f.d)),
      el('td', {}, r ? (r.explained ? r.aircraft : el('span', { class: 'muted' }, 'none fits')) : '–'),
      el('td', {}, r && r.explained ? `${fmt(r.rms_pct, 2)} %` : '–'),
      el('td', {}, used),
      el('td', {}, el('button', { class: 'ghost small', onclick: (ev) => { ev.stopPropagation(); selectEvent(e.id, true); } }, '▶')));
    return row;
  }));
}

async function toggleUsed(id, on) {
  const ids = state.events.events.filter((e) => (e.id === id ? on : e.used)).map((e) => e.id);
  try {
    await api(`api/cases/${state.id}/settings`, { method: 'PUT', body: { match: { events: ids } } });
    state.events = await src.events(state.id); renderEvents(); drawSpec();
    showLog(`event selection saved (${ids.length} events); rerun the match to use it`);
  } catch (e) { showLog(`error: ${e.message}`); }
}

function selectEvent(id, play = false) {
  state.sel = id;
  renderEvents();
  const e = state.events.events.find((x) => x.id === id);
  const r = resultEvent(id), f = e.fit || {}, t0 = state.events.t0_utc;
  const audio = el('audio', { controls: true, preload: 'none', src: src.audio(state.id, e) });
  const box = $('#event-detail');
  box.classList.remove('hidden');
  box.replaceChildren(
    el('div', { class: 'row' }, el('b', {}, `${e.id}`), el('span', { class: 'muted' },
      `${utc(t0 + e.t_start)}–${utc(t0 + e.t_end)} UTC · ${e.label} (score ${fmt(e.score, 2)})`), audio),
    el('p', {}, `Straight-line fit: ${f.v ? fmt(f.v * 3.6) + ' km/h' : '–'}, closest distance ${dist(f.d)}, ` +
      `closest approach ${f.t_cpa != null ? utc(t0 + f.t_cpa) : '–'} UTC, SNR ${fmt(f.snr_db, 1)} dB.`),
    r && r.explained ? el('p', {}, `Matched to ${r.aircraft}${r.desc ? `, ${r.desc}` : ''}: closest ${dist(r.min_range_m)}, ` +
      `${fmt(r.height_above_mic_m)} m above the microphone, ${fmt(r.speed_mps * 3.6)} km/h; residual ${fmt(r.rms_pct, 2)} % in frequency.`)
      : r ? el('p', { class: 'muted' }, `No ADS-B aircraft explains this event at the estimated position (${r.n_candidates} candidates tried). ` +
        'It may be an aircraft without ADS-B, a ground vehicle, or a misdetection.') : null,
    e.reasons?.length ? el('p', { class: 'muted' }, 'Classifier: ' + e.reasons.join('; ')) : null);
  if (play) audio.play().catch(() => {});
  drawSpec();
}

// ------------------------------------------------------------------ spectrogram

let specImg = null;
async function drawSpec(range) {
  const div = $('#spec');
  if (!state.events || !window.Plotly) { if (window.Plotly) Plotly.purge(div); div.replaceChildren(el('p', { class: 'muted' }, 'No spectrogram yet.')); return; }
  const req = ++state.specReq;
  const kind = $('#spec-kind').value;
  let img;
  try { img = await src.spec(state.id, kind, range); } catch (e) { div.replaceChildren(el('p', { class: 'error' }, e.message)); return; }
  if (req !== state.specReq) return;
  if (specImg && specImg.startsWith('blob:')) URL.revokeObjectURL(specImg);
  specImg = img.url;
  const t0 = state.events.t0_utc, x = img.ext;
  const traces = [];
  for (const [i, e] of state.events.events.entries()) {
    const xs = [], ys = [];
    for (const tr of e.tracks) { tr.t.forEach((t, k) => { xs.push(iso(t0 + t)); ys.push(tr.f[k]); }); xs.push(null); ys.push(null); }
    traces.push({ x: xs, y: ys, mode: 'lines', type: 'scatter', name: e.id, meta: e.id,
      line: { color: COLORS[i % COLORS.length], width: state.sel === e.id ? 3 : 1.6 }, opacity: e.used ? 0.95 : 0.4,
      hovertemplate: `${esc(e.id)} · %{y:.1f} Hz<extra></extra>` });
  }
  if ($('#show-sim').checked && state.result) {
    for (const r of state.result.result.events) {
      if (!r.explained || !r.sims) continue;
      const xs = [], ys = [];
      for (const s of r.sims) { r.sim_t_utc.forEach((t, k) => { xs.push(iso(t)); ys.push(s.f[k]); }); xs.push(null); ys.push(null); }
      traces.push({ x: xs, y: ys, mode: 'lines', type: 'scatter', name: `${r.id} model`, meta: r.id,
        line: { color: colorOf(r.id), width: 1.4, dash: 'dash' },
        hovertemplate: `${esc(r.id)} model: ${esc(r.aircraft)} · %{y:.1f} Hz<extra></extra>` });
    }
  }
  const fMax = Math.min(x.f1, Math.max(250, ...state.events.events.flatMap((e) => e.tracks.map((t) => Math.max(...t.f)))) * 1.15);
  const layout = {
    margin: { l: 55, r: 10, t: 10, b: 40 }, showlegend: false, dragmode: 'zoom', hovermode: 'closest',
    xaxis: { type: 'date', title: { text: 'UTC' }, showgrid: false, range: range ? [iso(t0 + range.t0), iso(t0 + range.t1)] : [iso(t0 + x.t0), iso(t0 + x.t1)] },
    yaxis: { title: { text: 'Hz' }, showgrid: false, zeroline: false, range: range ? [range.f0, range.f1] : [0, fMax] },
    images: [{ source: img.url, xref: 'x', yref: 'y', x: iso(t0 + x.t0), y: x.f1, sizex: (x.t1 - x.t0) * 1000, sizey: x.f1 - x.f0,
      sizing: 'stretch', layer: 'below', xanchor: 'left', yanchor: 'top' }],
    plot_bgcolor: '#fff', paper_bgcolor: '#fff',
  };
  await Plotly.react(div, traces, layout, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] });
  if (!div._wired) {
    div._wired = true;
    div.on('plotly_click', (ev) => { const id = ev.points[0]?.data.meta; if (id) selectEvent(id); });
    let timer = null;
    div.on('plotly_relayout', (ev) => {
      if (STATIC) return;
      if (ev['xaxis.autorange'] || ev['yaxis.autorange']) { clearTimeout(timer); timer = setTimeout(() => drawSpec(), 200); return; }
      const xa = div.layout.xaxis.range, ya = div.layout.yaxis.range, T0 = state.events.t0_utc;
      if (!ev['xaxis.range[0]'] && !ev['yaxis.range[0]'] && !ev['xaxis.range']) return;
      clearTimeout(timer);
      timer = setTimeout(() => drawSpec({ t0: Date.parse(xa[0] + 'Z') / 1000 - T0, t1: Date.parse(xa[1] + 'Z') / 1000 - T0, f0: ya[0], f1: ya[1] }), 350);
    });
  }
}

// ------------------------------------------------------------------ map

function circle(lat, lon, r, n = 120) {
  const la = [], lo = [];
  for (let i = 0; i <= n; i++) {
    const a = (2 * Math.PI * i) / n;
    la.push(lat + (r * Math.cos(a)) / 111320);
    lo.push(lon + (r * Math.sin(a)) / (111320 * Math.cos((lat * Math.PI) / 180)));
  }
  return { la, lo };
}

function drawMap() {
  const div = $('#map');
  if (!window.Plotly) return;
  const s = state.c.settings, [clat, clon] = s.search_center, R = s.search_radius_km * 1000;
  const traces = [];
  const ring = circle(clat, clon, R);
  traces.push({ type: 'scattermap', lat: ring.la, lon: ring.lo, mode: 'lines', line: { color: '#1f6f8b', width: 1.5 },
    name: 'search area', hoverinfo: 'skip' });
  traces.push({ type: 'scattermap', lat: [clat], lon: [clon], mode: 'markers', marker: { size: 7, color: '#1f6f8b' },
    name: 'search centre', hovertemplate: 'search centre<extra></extra>' });
  const res = state.result;
  if (res && $('#show-cost').checked) {
    const tot = res.coarse.total, lo = Math.min(...tot);
    traces.push({ type: 'scattermap', lat: res.coarse.lat, lon: res.coarse.lon, mode: 'markers', name: 'search cost',
      marker: { size: 9, color: tot.map((v) => Math.min(v - lo, 15)), colorscale: [[0, '#0b3954'], [0.35, '#4f8fbf'], [1, '#e9eef3']],
        opacity: 0.7, showscale: false }, hovertemplate: 'cost %{text}<extra></extra>', text: tot.map((v) => fmt(v, 1)) });
  }
  if (state.flights && $('#show-flights').checked) {
    const la = [], lo = [], tx = [];
    for (const f of state.flights.flights) { la.push(...f.lat, null); lo.push(...f.lon, null); tx.push(...f.lat.map(() => f.label), null); }
    traces.push({ type: 'scattermap', lat: la, lon: lo, mode: 'lines', line: { color: '#8795a3', width: 1 }, opacity: 0.6,
      name: 'flights', text: tx.map(esc), hovertemplate: '%{text}<extra></extra>' });
  }
  if (res) {
    const R0 = res.result;
    for (const e of R0.events) {
      if (!e.explained || !e.path) continue;
      traces.push({ type: 'scattermap', lat: e.path[0], lon: e.path[1], mode: 'lines', line: { color: colorOf(e.id), width: 3 },
        name: e.id, hovertemplate: `${esc(e.id)}: ${esc(e.aircraft)}<extra></extra>` });
      traces.push({ type: 'scattermap', lat: [e.closest_latlon[0]], lon: [e.closest_latlon[1]], mode: 'markers',
        marker: { size: 8, color: colorOf(e.id) }, hovertemplate: `${esc(e.id)} closest point, ${dist(e.min_range_m)}<extra></extra>` });
    }
    const p = res.posterior;
    traces.push({ type: 'densitymap', lat: p.lat, lon: p.lon, z: p.p, radius: 12, showscale: false, opacity: 0.6,
      colorscale: [[0, 'rgba(214,39,40,0)'], [1, 'rgba(214,39,40,0.9)']], hoverinfo: 'skip', name: 'posterior' });
    if (R0.alternatives.length) {
      traces.push({ type: 'scattermap', lat: R0.alternatives.map((a) => a.latlon[0]), lon: R0.alternatives.map((a) => a.latlon[1]),
        mode: 'markers', marker: { size: 9, color: '#ffffff', opacity: 0.9 }, name: 'alternatives',
        text: R0.alternatives.map((a) => `alternative: cost ${fmt(a.local_total, 1)}, ${a.n_explained} explained`), hovertemplate: '%{text}<extra></extra>' });
    }
    traces.push({ type: 'scattermap', lat: [R0.best.lat], lon: [R0.best.lon], mode: 'markers', marker: { size: 16, color: '#111' }, hoverinfo: 'skip' });
    traces.push({ type: 'scattermap', lat: [R0.best.lat], lon: [R0.best.lon], mode: 'markers', marker: { size: 10, color: '#d62728' },
      name: 'estimate', hovertemplate: `estimate<br>90%: ${radius(R0.uncertainty.bootstrap_90_m)}<extra></extra>` });
    if (res.validation) {
      const v = res.validation.reference;
      traces.push({ type: 'scattermap', lat: [v.lat], lon: [v.lon], mode: 'markers', marker: { size: 11, color: '#2a9d50' },
        name: 'reference', hovertemplate: `reference (${esc(v.source)})<br>error ${dist(res.validation.error_m)}<extra></extra>` });
    }
  }
  const centre = res ? { lat: res.result.best.lat, lon: res.result.best.lon } : { lat: clat, lon: clon };
  const h = div.clientHeight || 470;
  const zoom = Math.log2((78271 * Math.cos((clat * Math.PI) / 180) * h) / (1.8 * R));
  Plotly.react(div, traces, {
    margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false,
    map: { style: 'carto-positron', center: div._mapView?.center || centre, zoom: div._mapView?.zoom ?? zoom },
  }, { responsive: true, displaylogo: false });
  if (!div._wired) {
    div._wired = true;
    div.on('plotly_relayout', (ev) => { if (ev['map.center']) div._mapView = { center: ev['map.center'], zoom: ev['map.zoom'] }; });
  }
}

// ------------------------------------------------------------------ settings dialog

function openSettings(isNew) {
  const f = $('#form-settings');
  f.reset();
  $('#settings-title').textContent = isNew ? 'New case' : 'Case settings';
  $('#settings-error').classList.add('hidden');
  f.audio_file.required = isNew;
  $('#audio-field').querySelector('input').title = isNew ? '' : 'leave empty to keep the current audio';
  const s = isNew ? { search_radius_km: 20, ground_temp_c: 15, terrain: 'flat', ground_elevation_m: 0, adsb_source: 'auto', match: { min_score: 0.5 } }
    : state.c.settings;
  f.name.value = isNew ? '' : s.name || '';
  f.audio_start_utc.value = s.audio_start_utc || '';
  f.search_center.value = s.search_center ? s.search_center.join(', ') : '';
  f.search_radius_km.value = s.search_radius_km; f.ground_temp_c.value = s.ground_temp_c;
  f.terrain.value = s.terrain; f.ground_elevation_m.value = s.ground_elevation_m;
  f.adsb_source.value = s.adsb_source; f.min_score.value = s.match?.min_score ?? 0.5;
  f.dataset.new = isNew ? '1' : '';
  $('#dlg-settings').showModal();
}

async function saveSettings(ev) {
  const f = $('#form-settings');
  if (ev.submitter?.value !== 'save') return;
  ev.preventDefault();
  const err = $('#settings-error');
  const fail = (m) => { err.textContent = m; err.classList.remove('hidden'); };
  const cen = f.search_center.value.split(/[ ,;]+/).filter(Boolean).map(Number);
  if (cen.length !== 2 || cen.some((x) => !isFinite(x))) return fail('Search centre must be "lat, lon" in decimal degrees.');
  const start = f.audio_start_utc.value.trim();
  if (!/^\d{4}-\d\d-\d\dT\d\d:\d\d(:\d\d(\.\d+)?)?(Z|[+-]\d\d:\d\d)$/.test(start)) return fail('Recording start must be ISO 8601 with a time zone, e.g. 2026-06-01T12:00:00Z.');
  const s = { name: f.name.value.trim(), audio_start_utc: start, search_center: cen,
    search_radius_km: +f.search_radius_km.value, ground_temp_c: +f.ground_temp_c.value, terrain: f.terrain.value,
    ground_elevation_m: +f.ground_elevation_m.value, adsb_source: f.adsb_source.value, match: { min_score: +f.min_score.value } };
  const audio = f.audio_file.files[0], adsbFile = f.adsb_file.files[0];
  if (s.adsb_source === 'file' && !adsbFile && !(state.c && state.c.settings.adsb_file && !f.dataset.new)) return fail('Choose the ADS-B file.');
  $('#btn-save').disabled = true;
  try {
    let c;
    if (f.dataset.new) {
      if (s.adsb_source === 'file') s.adsb_source = 'auto';            // set by the upload below
      c = await api('api/cases', { method: 'POST', body: { ...s, audio: audio.name } });
    } else {
      if (s.adsb_source === 'file' && adsbFile) s.adsb_source = state.c.settings.adsb_source === 'file' ? 'file' : 'auto';
      c = await api(`api/cases/${state.id}/settings`, { method: 'PUT', body: s });
    }
    if (audio) await upload(c.id, 'audio', audio, fail);
    if (adsbFile) await upload(c.id, 'adsb', adsbFile, fail);
    $('#dlg-settings').close();
    await loadCases(c.id);
  } catch (e) { fail(e.message); } finally { $('#btn-save').disabled = false; }
}

function upload(id, kind, file, report) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open('PUT', `api/cases/${id}/upload?kind=${kind}&name=${encodeURIComponent(file.name)}`);
    x.setRequestHeader('X-Adoppler', '1');
    x.upload.onprogress = (e) => e.lengthComputable && report(`Uploading ${file.name}: ${fmt((100 * e.loaded) / e.total)} %`);
    x.onload = () => (x.status < 300 ? resolve(JSON.parse(x.responseText)) : reject(new Error(JSON.parse(x.responseText).error)));
    x.onerror = () => reject(new Error('upload failed'));
    x.send(file);
  });
}

async function saveValidation(ev) {
  if (ev.submitter?.value !== 'ok') return;
  ev.preventDefault();
  const f = $('#form-validate'), err = $('#validate-error');
  const p = f.ref.value.split(/[ ,;]+/).filter(Boolean).map(Number);
  if (p.length !== 2 || p.some((x) => !isFinite(x))) { err.textContent = 'Reference must be "lat, lon".'; err.classList.remove('hidden'); return; }
  try {
    await api(`api/cases/${state.id}/validate`, { method: 'POST', body: { lat: p[0], lon: p[1], source: f.source.value } });
    $('#dlg-validate').close(); await openCase(state.id);
  } catch (e) { err.textContent = e.message; err.classList.remove('hidden'); }
}

async function makeSynthetic() {
  try {
    const job = await api('api/synth', { method: 'POST', body: { scene: 'example', run: true } });
    $('#empty').classList.add('hidden'); $('#case').classList.remove('hidden');
    $('#log').textContent = ''; showLog(`rendering a synthetic case (${job.case_id}); this takes a few minutes`);
    follow(job.id, async () => loadCases(job.case_id));
  } catch (e) { alertBox(e.message); }
}

function alertBox(msg) { showLog(`error: ${msg}`); }

// ------------------------------------------------------------------ wiring

function init() {
  $('#case-select').addEventListener('change', (e) => openCase(e.target.value));
  $('#spec-kind').addEventListener('change', () => drawSpec());
  $('#show-sim').addEventListener('change', () => drawSpec());
  $('#show-cost').addEventListener('change', drawMap);
  $('#show-flights').addEventListener('change', drawMap);
  if (!STATIC) {
    for (const li of document.querySelectorAll('.stages li')) li.querySelector('button').addEventListener('click', () => runStage(li.dataset.stage));
    $('#btn-all').addEventListener('click', () => runStage('all'));
    for (const b of ['#btn-new', '#btn-new2']) $(b).addEventListener('click', () => openSettings(true));
    for (const b of ['#btn-synth', '#btn-synth2']) $(b).addEventListener('click', makeSynthetic);
    $('#btn-settings').addEventListener('click', () => openSettings(false));
    $('#form-settings').addEventListener('submit', saveSettings);
    $('#form-validate').addEventListener('submit', saveValidation);
    api('api/info').then((i) => { $('#version').textContent = i.version; }).catch(() => {});
  }
  loadCases().catch((e) => { $('#main').prepend(el('p', { class: 'card error' }, `Cannot load cases: ${e.message}`)); });
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
