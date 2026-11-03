# adoppler: aircraft Doppler detection and acoustic geolocation

A propeller aircraft passing overhead drags its engine tones down in pitch. The shape of that Doppler curve depends
on the aircraft's speed and on where the microphone was. adoppler finds these curves in an audio recording,
matches them with historical ADS-B flight tracks and estimates **where the microphone was**, together with an
honest uncertainty and a verdict on whether the answer can be trusted at all.

- **Live demo (synthetic recording, runs in the browser):** https://aircraft-doppler.vercel.app
- Everything heavy runs locally: a command line tool plus a browser interface on `127.0.0.1`.
- Method and assumptions: [docs/method.md](docs/method.md).

| What you need | Why |
|---|---|
| An audio recording, ideally 20 minutes or more, with aircraft audible overhead | each pass is one measurement |
| The UTC time of the first sample, to within a few seconds | the clock offset is fitted within `delay_prior_s` (default −2 to +10 s) |
| A rough area (a town, within about 20 km) | the search disc |
| Historical ADS-B for that time and place | downloaded automatically, or your own file |

## Status

Confirmed: on synthetic recordings (rendered from the same physics, with wind, ground reflection, a clock offset,
a wrong temperature setting and distracting sounds) the hidden microphone position is recovered to within the
reported uncertainty; the demo case is off by 9 m with a 90% radius of 25 m. This proves internal consistency
only.

Preliminary: one real recording (a public livestream) was analysed blind: 7 of 13 passes were explained, with a
bootstrap 90% radius of 125 m. The independent check against the true position has not been done yet. Its error
will be published here as a distance only; the location stays private.

## Install

Needs Python 3.10 or newer and about 350 MB of disk for dependencies. All dependencies are binary wheels, so no
compiler is needed.

**Windows** (PowerShell):

```powershell
git clone https://github.com/Espaye/aircraft-doppler
cd aircraft-doppler
py -m venv .venv
.venv\Scripts\python -m pip install .
.venv\Scripts\adoppler --version
```

**macOS and Linux:**

```sh
git clone https://github.com/Espaye/aircraft-doppler
cd aircraft-doppler
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/adoppler --version
```

Activate the environment (`.venv\Scripts\activate` or `source .venv/bin/activate`) to type `adoppler` directly, as
below. Without git, download the ZIP from GitHub instead of cloning.

## Try it on a synthetic recording

```sh
adoppler synth cases/example        # renders a 20-minute recording with a hidden microphone position
adoppler serve                      # opens http://127.0.0.1:8765 with the cases/ folder
```

In the browser, open the case and press **Run all remaining**, then **Compare with the hidden truth**. Rendering
and the full run take about 2 minutes on a laptop. `--scene mini` makes a 6-minute case for a quick check.
The same on the command line:

```sh
adoppler run cases/example
adoppler validate cases/example --synthetic
```

## Your own recording

```sh
adoppler new cases/garden --audio garden.flac --start 2026-06-01T12:00:00Z --center 52.09,5.12 --temp 18
adoppler run cases/garden
adoppler serve
```

Or press **New case** in the browser and upload the audio there. `new` copies nothing: it writes
`cases/garden/case.json` pointing at the audio file. The stages are:

1. **analyse**: spectrogram, tonal enhancement, Doppler events, each with a straight-line fit (speed,
   closest distance, time of closest approach). Seconds to a few minutes.
2. **adsb**: flight tracks for the recording window within `aircraft_radius_km` (default 60 km).
3. **match**: searches positions and clock offsets for the best agreement between the measured curves and the
   curves predicted from every flight. Several minutes on all CPU cores.

Every stage caches its output in the case folder, so you can stop and resume. `--force` redoes a stage, and
`adoppler status <case>` shows what is done. In the browser you can listen to each event, see the measured
curve against the predicted one, leave events out, or draw a curve by hand where detection failed.

### Settings (`case.json`)

| Key | Default | Meaning |
|---|---|---|
| `audio`, `audio_start_utc`, `search_center` | required | audio file, UTC start time (`2026-06-01T12:00:00Z`), `[lat, lon]` |
| `search_radius_km` | 20 | radius of the search disc |
| `ground_temp_c` | 15 | air temperature at the ground; refined within ±5 °C |
| `delay_prior_s` | [−2, 10] | allowed recording-clock offset (stream latency) |
| `terrain` | `flat` | `flat` uses `ground_elevation_m`; `usgs-3dep` downloads elevation (United States only) |
| `geoid_undulation_m` | automatic | EGM96 from PROJ's online grid; set it by hand if offline |
| `adsb_source` | `auto` | `adsb.lol` archive, `adsb.aero`, or `file` with `adsb_file` |
| `match.min_score` | 0.5 | detection quality an event needs to be used |

The full list is in [adoppler/case.py](adoppler/case.py).

### Flight data

- **adsb.lol archive** (preferred): the complete daily feed with GNSS altitudes, published about 3–4 hours after
  midnight UTC. The tool streams the day's archive (about 3 GB) once and keeps only the tracks near you.
- **adsb.aero**: a quick query with simplified paths, used when the archive is not out yet. Redo the `adsb` stage
  with `--force --source adsb.lol` later for better results.
- **Your own file**: set `adsb_source` to `file` and `adsb_file` to a CSV with the columns `icao24, time, lat,
  lon, alt_m` and optionally `alt_ref` (`msl`, `hae` or `baro`), `callsign`, `type`, `gs_kt`. `time` is unix
  seconds or ISO 8601 UTC, and each row is one position report.

Respect each provider's terms. adsb.lol data is licensed under ODbL 1.0, so if you publish derived data, credit
adsb.lol and share it under the same licence.

## Reading the result

The result card gives a position, a 68% and 90% radius, and a **verdict**:

| Verdict | Meaning |
|---|---|
| `ok` | at least three passes explained, a stable best area, and no comparable alternative |
| `weak` | too few independent passes, or leaving one out moves the answer: treat it as a hint |
| `ambiguous` | another area fits about as well, or time-shifted flight data fits almost as well: no reliable location |
| `failed` | fewer than two passes explained |

The radii come from resampling whole passes (bootstrap) and leaving each pass out. They do not come from the
optimiser's precision, which is always far too optimistic. Every flag behind the verdict is listed with the
result. The prediction is stamped with a SHA-256 hash when it is made, and `adoppler validate --lat --lon
--source` checks that stamp before comparing with a reference, so a blind test stays verifiably blind.

## Limitations

- It needs tonal sources: piston and turboprop aircraft at low altitude. Jets at cruise altitude are usually
  too faint or too slow in pitch change.
- Clock errors beyond `delay_prior_s`, heavy wind shear, echoes from buildings or a microphone indoors can bias
  the result. The verdict catches some of this, but not all.
- Aircraft without ADS-B (many light aircraft) produce events that nothing can explain. They are reported and
  left out of the result.
- The physics is a straight-ray, layered-atmosphere model, with no refraction by wind and no terrain shadowing.

## Privacy and responsible use

A location estimate of a microphone is a location of a person or their home. Only analyse recordings you made or
have permission to analyse, and do not publish estimated locations of other people's recordings without their
consent. `adoppler export` (which builds static demo sites like the one above) refuses non-synthetic cases unless
you pass `--allow-real`.

The local server listens on `127.0.0.1` only and rejects requests from other web pages: it checks the `Host` and
`Origin` headers, requires a custom header on writes, and confines case IDs to `[a-z0-9-]`. It does not send your
audio anywhere. Network access is limited to downloading flight data, elevation (`usgs-3dep`) and the geoid grid;
the map in the browser loads tiles from CARTO, which therefore sees the area you look at.

## Development

```sh
python -m pip install -e .[test]
python -m pytest                    # about 30 s; -m "not slow" skips the end-to-end run
python tools/build_site.py          # rebuilds site/ (the Vercel demo) from the synthetic example
```

Code map: `spectro` (STFT, tonal enhancement), `detect` (events, straight-line fit), `adsb` (sources), `geo`
(ENU, geoid, terrain), `physics` (trajectory, Doppler with travel time), `match` (cost grids), `locate` (search,
uncertainty, verdict, validation), `pipeline` (stages), `synth` (test recordings), `server` and `web/` (browser
interface), `export` (static copy), `cli`. Notes for coding agents are in [AGENTS.md](AGENTS.md).

## Licence and attribution

MIT, see [LICENSE](LICENSE). The browser interface uses [Plotly.js](https://plotly.com/javascript/) (MIT), loaded
from the installed `plotly` package. Map tiles © [CARTO](https://carto.com/attributions), map data ©
[OpenStreetMap](https://www.openstreetmap.org/copyright) contributors. Flight data comes from
[adsb.lol](https://adsb.lol) (ODbL 1.0) and [adsb.aero](https://adsb.aero), elevation from USGS 3DEP (public
domain), and the geoid from EGM96 via PROJ. The demo recording and its flights are synthetic.
