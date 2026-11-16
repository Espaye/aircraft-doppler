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
- **Coverage gaps**: for an event that stays unexplained, `match` lists aircraft whose ADS-B track has no fixes
  for more than 150 s around the event's closest approach but that could have made it: ground speed before and
  after the gap within -25 %/+33 % of the Doppler speed, steady progress across the gap, height not far above
  the Doppler closest distance, and able to reach the search area in time. Low aircraft often drop out of
  receiver coverage exactly when they pass close to a microphone. Their path inside the gap is unknown, so they
  are reported (`events[].coverage_gap_candidates` in `result.json`) and never used for the estimate.

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

## 7. Experimental broadband jet screening

`adoppler jets` is an independent opt-in experiment, not a relaxation of the Doppler quality threshold. Each
one-second audio block yields a Welch PSD. The median spectral level within 100-250, 250-500, 500-1000 and
1000-2000 Hz suppresses isolated tones while preserving broadband energy. Five-second median and four-second
Gaussian smoothing remove short fluctuations. A recording-wide 20th-percentile baseline per band preserves
slow rises that the tonal enhancement removes. The median of the four relative band levels is an activity
envelope, in uncalibrated relative dB. It is neither sound-pressure level nor an aircraft likelihood.

Prominent sustained activity produces unclassified candidates. Overlapping automatic spans are merged;
manual windows are supported. Recording-edge clipping, rapid onsets, very long events and limited frequency
support are flagged. Section 8 (`classify`) separates jets and propeller aircraft from road traffic and handling.

A second, faster view computes RMS in roughly 0.1 s frames (rounded sample edges). A frame more than 8 dB
above its 51-frame local median and with an adjacent-frame jump above 8 dB is marked as a possible impact.
This is an experimental heuristic informed by listening examples, not a calibrated detector. Each mark is padded
by ten frames on either side. One-second PSD blocks touching those spans are interpolated from unmarked blocks
before the existing smoothing and baseline calculation, giving `clean_activity_db`. Original activity, candidate
IDs and bounds are retained so reviews remain comparable. Matching uses the cleaned envelope and omits samples
within one second of each masked span; fewer than five samples or less than 20 s of remaining support cannot
fit. Smoothing of interpolated band levels is an approximation, not recovered source audio. Retained observations
are fixed before flight comparison and before the wrong-time controls; a hypothesis cannot choose which audio
to ignore. The report overlays both envelopes and lists impact counts, the 99th percentile of absolute fast
level differences, and median low-band versus high-band PSD contrast. These are gain-relative diagnostics,
not aircraft scores. Loud sustained sounds are allowed; an impact does not veto an entire candidate.

`--review-labels` imports explicit human `likely_jet` selections from a listening-page JSON export. The case
folder ID, UTC start, complete candidate ID set and 20 s padded clip bounds (0.1 s tolerance) must agree with
the current audio analysis. Negative, mixed, uncertain and unreviewed labels are withheld, and free-text notes
are preserved without automatic interpretation. These are human source impressions, not independently verified
flight labels. All candidates remain in `events.json` and the report; `selected_for_matching` states the actual
selection. The result includes selected IDs and the review SHA-256. `--events` intersects that selection; manual
windows are incompatible with a review export. Without explicit review input no source classification or
review-based exclusion is applied. New recordings are needed to evaluate generalisation beyond the examples.

With ADS-B available, the existing retarded-time solver supplies slant range R at the emission time for each
candidate receiver and flight. A fixed-source inverse-square-intensity hypothesis gives a level proportional
to -20 log10(R). An offset removes the unknown source level; a nonnegative gain limited to [0, 2] allows modest
compression or extra attenuation. This deliberately simplified shape omits source directivity, changing thrust,
frequency-dependent atmospheric absorption, wind refraction, ground interference and phone gain control. Sound
travel time is explicitly included, so the observed peak need not coincide with visual overhead passage.

Candidate tracks have median MSL altitude at least 4000 m and median horizontal speed 140-340 m/s. These filters
select high/fast traffic, not a known aircraft class. The audibility hypothesis cuts off best slant ranges above
40 km; flight retrieval must therefore cover the prior search radius plus 40 km. A coarse search covers the prior disc (2 km default spacing). Each event profiles an extra +/-45 s timing
shift beyond the case clock prior, sampled at 10 s plus endpoints. This is nuisance slack, not a separately
identified clock correction. Full padded event support must be covered by the track; partial data cannot win
by fitting only the peak. Cost is residual energy divided by observed variation after the shape fit. A residual
fraction <=0.5 passes an exploratory shape screen; that threshold is not a calibrated probability.

Assignments are one to one, with dummy unmatched sounds. Repeated ICAO records are combined so one aircraft
cannot supply multiple independent constraints. Three circular time shifts repeat the same search as coincidence
diagnostics, not formal significance tests. Whole windows move without splicing their sound across the wrap;
a window may extend into the flight-data padding beyond the recording. Leave-one-out and 100 whole-event
bootstrap resamples redo assignments and report coarse-grid movement. They measure model-conditional stability,
not real-world accuracy; shared flight-data and propagation errors are not captured.

The result gives near-best cells within 0.05 of the best mean shape score, their sampled area and bounding box.
The set may be disconnected or follow an extended corridor. It is not a posterior or confidence region. One
flight, similarly good wrong-time fits, broad/boundary-reaching regions, unstable leave-one-out positions and
flagged events prevent an unqualified result. All results remain experimental, with status `exploratory` at best
and no claimed `location_estimate`. In particular, a very small bootstrap spread cannot establish accuracy.
Synthetic tests verify implementation and ambiguity preservation only. Independent real recordings with known
microphone positions are needed to establish whether this method can reliably narrow a region.

The physical distinction between broadband jet noise and tonal fan noise, and the importance of directivity,
spreading, atmospheric absorption and propagation time are described in NASA's
[Auralization of Air Vehicle Noise for Community Noise Assessment](https://ntrs.nasa.gov/citations/20200002351).
Our inverse-range envelope is a deliberately limited hypothesis, not an implementation of that full model.

No public server endpoints or browser UI behavior change. The private standalone HTML report embeds the installed
Plotly library and uses no map tiles. Experimental files and their input/code hashes are saved under `jets/`,
with a result SHA-256 stamp, apart from the normal blind Doppler prediction and validation/export paths.

## 8. Source screening (`classify`)

`classify` screens each broadband candidate from `jets/events.json` (created on first use if missing). The audio
is re-read in 0.1 s frames (2048-point Hann FFT) and summed into third-octave bands from 63 Hz to 4 kHz. The
100-2000 Hz power sum gives a level track; a 98th-percentile peak-to-median ratio of the 60-1000 Hz spectrum
gives a per-frame tonality. Features per candidate:

- `peak_excess_db`: 3 s-smoothed level at the peak above the recording's 10th percentile.
- `rise_s`, `fall_s`, `pass_s`: time from the peak until the level is `min(10, max(3, 0.7 x excess))` dB lower.
- `burst_db`: standard deviation of the 0.5 s-smoothed minus the 8 s-smoothed level inside that span. Separate
  vehicles, gear changes and bumps raise it; a distant aircraft (whose level changes with range) keeps it low.
- `hf_minus_lf_db`: mean excess over the quiet level in the 1-2 kHz bands minus that in the 100-400 Hz bands,
  within 5 s of the peak. Air absorption removes high frequencies over kilometres (ISO 9613-1: a few dB/km at
  1 kHz, under 1 dB/km at 125 Hz), so a jet at cruise height is strongly low-passed.
- `tonal_excess_db`: median tonality in the span minus the recording median (engine and propeller tones).
- The best overlapping tonal Doppler event labelled `aircraft` by `detect` (number of harmonic tracks, duration,
  straight-line fit) and a fitted one-pass envelope (`tau_before_s`, `tau_after_s`, `fit_rms_db`), reported but
  not used by the rule.

Each test maps a feature linearly between two limits onto 0..1, and a class score is the geometric mean of its tests:

| class | tests (0 at the first limit, 1 at the second) |
|---|---|
| jet | `burst_db` 3.0 to 2.2; `tonal_excess_db` 1.0 to 0.3; `hf_minus_lf_db` -5 to -10; `pass_s` 20 to 40; `peak_excess_db` 6 to 10 |
| prop | harmonic tracks 10 to 20; Doppler event duration 45 to 90 s; `tonal_excess_db` 0.5 to 1.5; `burst_db` 4.0 to 3.0 |

`handling` takes precedence when `burst_db` >= 4.5 together with at least 8 impact frames (section 7) or a fall
under 1.5 s. Otherwise a score of at least 0.5 assigns jet (or prop); everything else is `not_aircraft`, with the
failed tests as reasons. A speed or distance gate on the tonal fit is deliberately absent: an overhead light
aircraft (35 m/s, 350 m) and a heavy road vehicle (30 m/s, 220 m) had nearly identical fits and even the same
harmonic spacing, while the duration and completeness of the harmonic series separated them.

The limits come from one owner-verified recording, so the evaluation is small: 15/15 labelled sounds agree on
that recording (resubstitution, not a test), and on a second recording with unverified labels 5/7 agree (both
probable jets missed). `--classified` on `match` selects the tonal events linked to `prop` sounds; on `jets` it
selects the `jet` sounds. The screening file stores the start time and candidate bounds; a changed detection
requires running `classify` again.

