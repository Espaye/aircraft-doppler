# adoppler: aircraft Doppler detection and acoustic geolocation

A propeller aircraft passing overhead drags its engine tones down in pitch. The shape of that Doppler curve depends
on the aircraft's speed and on where the microphone was. adoppler finds these curves in an audio recording,
matches them with historical ADS-B flight tracks and estimates **where the microphone was**, with an honest
uncertainty and a verdict on whether to trust it.

https://github.com/user-attachments/assets/dd43477a-876e-4b3c-99ae-7a695fb0c441

- **Live demo** (synthetic recording, runs in the browser): https://aircraft-doppler.vercel.app
- Runs locally: a command-line tool plus a browser interface on `127.0.0.1`.
- How it works: [docs/method.md](docs/method.md).

You need:

- an audio recording, ideally 20 minutes or more, with aircraft audible overhead (each pass is one measurement);
- its UTC start time, to within a few seconds;
- a rough area, such as a town (the search disc is 20 km by default);
- historical ADS-B flight data for that time and place, downloaded automatically or from your own file.

## Status

On synthetic recordings the hidden microphone position is recovered within the reported uncertainty: the demo
case is off by 9 m with a 90% radius of 25 m. That proves internal consistency only.

On real recordings it has been tested blind at one site, on two days of a public livestream:

| Recording | Passes explained | Verdict | 68% radius | Error |
|---|---|---|---|---|
| day 1 | 7 of 13 | `ok` | 35 m | 17 m |
| day 2 | 4 of 5 | `ok` | 214 m | 114 m |

Both errors fall inside the 68% radius. One site is not a general accuracy figure.

## Install

Needs Python 3.10 or newer and about 350 MB of disk; no compiler.

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

Activate the environment (`.venv\Scripts\activate` or `source .venv/bin/activate`) to type `adoppler` directly.

## Try it on a synthetic recording

```sh
adoppler synth cases/example        # renders a 20-minute recording with a hidden microphone position
adoppler serve                      # opens http://127.0.0.1:8765
```

Open the case, press **Run all remaining**, then **Compare with the hidden truth** (about 2 minutes). On the command
line: `adoppler run cases/example`, then `adoppler validate cases/example --synthetic`.

## Your own recording

```sh
adoppler new cases/garden --audio garden.flac --start 2026-06-01T12:00:00Z --center 52.09,5.12 --temp 18
adoppler run cases/garden
adoppler serve
```

Or press **New case** in the browser. `run` does three stages, each cached in the case folder (`--force` redoes
one, `adoppler status <case>` shows progress):

1. **analyse** finds the Doppler events in the audio;
2. **adsb** downloads the flight tracks for the recording window;
3. **match** searches for the microphone position and clock offset that best explain the events (several minutes).

In the browser you can listen to each event, compare the measured and predicted curves, leave events out, or draw a
curve by hand where detection failed.

**Recording tips:** record outdoors with the phone lying still, away from busy roads (cars and lorries make
Doppler-shifted engine tones too). Use a plain recorder mode without noise suppression and keep the original file.
Note the start time, and write down when you hear an aircraft and whether it is a propeller aircraft or a jet.

### Telling aircraft from road traffic

Near a road, most sounds are cars and lorries. `classify` labels each sustained sound as **jet**, **prop**,
**handling** (phone handling, wind) or **not_aircraft**, and says why:

```sh
adoppler analyse cases/garden
adoppler classify cases/garden
adoppler match cases/garden --classified    # use only the events linked to propeller aircraft
```

The rules are physical rather than a trained model, but they were tuned on a single recording: treat them as a
filter and listen to what they keep and reject. See [method, section 8](docs/method.md#8-source-screening-classify).

### Jets (experimental)

The matcher needs engine tones, which jets at cruise height rarely have. The experimental `jets` command uses the
slow broadband rise and fall of their sound instead:

```sh
adoppler jets cases/garden --analyse-only   # candidate sounds and a report in cases/garden/jets/
adoppler adsb cases/garden
adoppler jets cases/garden --classified     # compare the sounds classify marked as jets with the flights
```

`--window START END` (seconds from the start) restricts it to a sound you heard. It gives rough candidate areas,
never a location estimate, has not been validated on real recordings, and never changes the main result. See
[method, section 7](docs/method.md#7-experimental-broadband-jet-screening).

## Settings (`case.json`)

| Key | Default | Meaning |
|---|---|---|
| `audio`, `audio_start_utc`, `search_center` | required | audio file, UTC start time (`2026-06-01T12:00:00Z`), `[lat, lon]` |
| `search_radius_km` | 20 | radius of the search disc |
| `aircraft_radius_km` | 60 | radius for flight data; keep it at least `search_radius_km` + 40 for `jets` |
| `ground_temp_c` | 15 | air temperature at the ground; refined within ±5 °C |
| `delay_prior_s` | [−2, 10] | allowed recording-clock offset |
| `audio_end_utc` | none | when the recording stopped; reveals a pause or dropout (see below) |
| `terrain` | `flat` | `flat` uses `ground_elevation_m`; `usgs-3dep` downloads elevation (United States only) |
| `geoid_undulation_m` | automatic | downloaded from PROJ; set it by hand when offline |
| `adsb_source` | `auto` | `adsb.lol`, `adsb.aero`, or `file` with `adsb_file` |
| `match.min_score` | 0.5 | detection quality an event needs to be used |

The full list is in [adoppler/case.py](adoppler/case.py).

**Pauses and dropouts.** A phone recorder that pauses, or drops audio, makes every later sound arrive later than
the start time implies. Give the time the recording stopped (`--end`, or `audio_end_utc`); for M4A files from
phone recorders this appears to be the container's creation time,
`ffprobe -v error -show_entries format_tags=creation_time <file>`. If the audio is more than 10 s shorter
than start to end, the case warns, `match` widens its clock range by the gap, and `jets` also tries the later
clock from each sound on. `adoppler status` lists this and other warnings, such as a missing `ground_temp_c`.

**Flight data.** The adsb.lol archive is the complete daily feed, published a few hours after midnight UTC
(the day's 3 GB is streamed once and only nearby tracks are kept). Until it is out, adsb.aero gives simplified
paths; rerun `adoppler adsb <case> --force --source adsb.lol` later for better results. Your own file is a CSV
with the columns `icao24, time, lat, lon, alt_m` (optionally `alt_ref`, `callsign`, `type`, `gs_kt`). Respect each
provider's terms: adsb.lol data is ODbL 1.0, so credit it and share derived data under the same licence.

## Reading the result

The result gives a position, a 68% and 90% radius, and a **verdict**:

| Verdict | Meaning |
|---|---|
| `ok` | at least three passes explained, a stable best area, and no comparable alternative |
| `weak` | too few independent passes, all from one aircraft, a fitted temperature far from `ground_temp_c`, or leaving one out moves the answer: treat it as a hint |
| `ambiguous` | another area, or time-shifted flight data, fits about as well: no reliable location |
| `failed` | fewer than two passes explained |

The radii come from resampling and leaving out whole passes, not from the optimiser's precision. A clear pass that
nothing explains is often a light aircraft outside ADS-B coverage; `match` lists likely candidates as
`coverage gap` lines in its log. To compare with a position you know, run
`adoppler validate <case> --lat <lat> --lon <lon> --source "<where it comes from>"`.

## Limitations

- The matcher needs tonal sources: piston and turboprop aircraft at low altitude.
- Aircraft without ADS-B, or outside receiver coverage, cannot be used.
- Clock errors beyond `delay_prior_s`, a pause in the recording without `audio_end_utc`, strong wind, echoes from
  buildings or recording indoors can bias the result.
  The verdict catches some of this, not all.
- The physics uses straight sound rays: no refraction by wind and no terrain shadowing.

## Privacy and responsible use

A microphone's location is a person's location. Only analyse recordings you made or have permission to analyse,
and do not publish estimated locations of other people's recordings without their consent. `adoppler export`
(which builds static demo sites) refuses real cases unless you pass `--allow-real`.

Your audio stays on your computer. The local server listens on `127.0.0.1` only and refuses requests from other web
pages. Network access is limited to flight data, elevation and the geoid grid; the map loads tiles from CARTO,
which therefore sees the area you look at.

## Development

```sh
python -m pip install -e .[test]
python -m pytest                    # about 30 s; -m "not slow" skips the end-to-end run
python tools/build_site.py          # rebuilds site/ (the Vercel demo) from the synthetic example
```

Code map: `spectro` (STFT, tonal enhancement), `detect` (events), `adsb` (sources), `geo` (ENU, geoid, terrain),
`physics` (Doppler with travel time), `match` (cost grids), `locate` (search, uncertainty, verdict, validation),
`pipeline` (stages), `synth` (test recordings), `server` and `web/` (browser interface). Notes for coding agents are
in [AGENTS.md](AGENTS.md).

## Licence and attribution

MIT, see [LICENSE](LICENSE). The browser interface uses [Plotly.js](https://plotly.com/javascript/) (MIT), loaded
from the installed `plotly` package. Map tiles © [CARTO](https://carto.com/attributions), map data ©
[OpenStreetMap](https://www.openstreetmap.org/copyright) contributors. Flight data comes from
[adsb.lol](https://adsb.lol) (ODbL 1.0) and [adsb.aero](https://adsb.aero), elevation from USGS 3DEP (public
domain), and the geoid from EGM96 via PROJ. The demo recording and its flights are synthetic.
