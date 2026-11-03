# Method

## 1. Detecting Doppler events (`spectro`, `detect`)

The audio is turned into a power spectrogram (STFT), processed in chunks so any length fits in memory. Tonal
enhancement then removes two backgrounds: per frequency, the median over sliding ~40 s blocks (hums and fixed
tones); per frame, the median over ±20 Hz (wind and broadband transients). What remains is roughly "dB above
the local noise floor" for narrow lines. Peaks are linked into tracks. Tracks that run together for a while and
keep a constant frequency ratio are grouped as harmonics of one source; propeller and engine tones usually form
several such lines.

Each group is fitted with the closed-form Doppler curve of a straight, level, constant-speed pass:

    f(t) = f0 / (1 + v_r(t) / c),   v_r from speed v, closest distance d and time of closest approach

This fit gives `v`, `d`, `t_cpa` and their uncertainties, which are used for screening only. The location never
relies on it. Tracks that are patchy, rough or faint are dropped first. Each event gets a quality score from
signal-to-noise, duration, fit residual, Doppler swing and number of harmonics. Events with almost no swing
(steady tones), rising steps, or the speed and distance of a road vehicle are labelled as such and down-weighted,
so they fall below `match.min_score`.

## 2. Flight data (`adsb`)

Tracks for the recording window and `aircraft_radius_km` are normalised to time, latitude, longitude and height
above mean sea level. GNSS heights are preferred; barometric heights are used when nothing else exists and are
marked as such. Ellipsoidal heights use the EGM96 geoid undulation at the search centre.

## 3. Forward model (`physics`, `geo`)

For a candidate microphone position, each aircraft's track is smoothed and interpolated in a local
east-north-up frame. For every reception time, the emission time is solved by Newton iteration on
`t_rx − τ = R(τ) / c`. The sound speed `c` comes from the ground temperature with a standard lapse rate, averaged
along the ray. The predicted received frequency is `f0 / (1 + v_r / c)`, with `v_r` the aircraft's velocity
component along the ray at emission. Terrain sets the microphone height: flat ground at a given elevation, or a
USGS 3DEP elevation model in the United States.

## 4. Matching and search (`match`, `locate`)

For each event, candidate aircraft are those airborne near the search area at the right time whose ground speed
is plausible for the fitted `v`. For one microphone position and clock offset `δ`, an event's cost is computed
like this:

- compare `log f` of the measured curve with `log f` of the prediction, after removing the unknown `f0` as a
  weighted mean
- apply a Cauchy (robust) loss, so a few wrong points cannot dominate
- keep the best candidate aircraft for that event

The total is the sum over events. An event counts as *explained* when its mean cost is below a fixed threshold
(about 0.35% rms in frequency).

The search proceeds in four steps:

1. A coarse grid over the search disc and over `δ` within `delay_prior_s`.
2. Local grids around the best few separated modes.
3. A least-squares refinement of position, `δ` and ground temperature (±5 °C), using the explained events.
4. A fine grid around the result, which is turned into a pseudo-posterior. Residuals within one event are
   strongly correlated, so each event is assumed to carry only about 10 independent samples.

## 5. Uncertainty and verdict

The reported radii come from resampling, not from the optimiser:

- **Event bootstrap**: 500 resamples of the events with replacement, grouped by aircraft so that repeated
  passes of one aircraft are not counted as independent. For each resample the per-event costs on the fine grid
  are summed again and the best point found. The 68% and 90% distances of those points from the estimate are
  reported. The result also records how often the coarse best area
  changes (*mode stability*).
- **Leave one out**: one event is dropped at a time, which shows when a single pass carries the answer.
- **Time-shift null tests**: all events are moved against the flight data (by a quarter, half and three
  quarters of the recording, wrapped around) and searched again. If shifted data fit almost as well as the real data, the match may be coincidence.
- **Alternatives**: the best other modes, with their cost and number of explained events.

`assess()` turns these into one verdict (`ok`, `weak`, `ambiguous`, `failed`) and lists every reason.

## 6. Blind validation

`match` writes `result.json` and its SHA-256 hash before any reference is consulted. `validate` re-checks the
hash, then stores the error distance, the error relative to the bootstrap radius, the rank of the true cell on
the coarse grid and the errors of the alternatives in a separate `validation.json`. Synthetic cases keep their
truth in `truth.json`, which the search never reads.

## What the synthetic tests do and do not show

`adoppler synth` renders recordings with the same propagation physics, plus effects the matcher does not model:

- wind
- a ground reflection
- distractor sounds (a car, hum, an aircraft without ADS-B, silent ADS-B tracks)
- a clock offset
- a temperature setting that is off

Recovering the hidden position therefore tests the detection, the search and the honesty of the uncertainty. It
does not test the physics itself. That needs real recordings with an independently known position; the results
so far are in the Status section of the README.
