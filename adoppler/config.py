"""Physical constants and default case settings."""

FT = 0.3048
KT = 0.514444
FPM = 0.3048 / 60

# Sound speed: c = 331.3 * sqrt(1 + T/273.15) m/s with a standard lapse rate along the path.
LAPSE_K_PER_M = 0.0065

# Defaults for case.json (see adoppler/case.py for the meaning of each key).
DEFAULTS = {
    'search_radius_km': 20.0,      # candidate microphone positions around search_center
    'aircraft_radius_km': 60.0,    # ADS-B traffic retrieved around search_center
    'ground_temp_c': 15.0,
    'delay_prior_s': [-2.0, 10.0],  # recording clock minus acoustic time (encoder/stream latency)
    'mic_height_agl_m': 2.0,
    'terrain': 'flat',             # 'flat' (ground_elevation_m) or 'usgs-3dep' (United States only)
    'ground_elevation_m': 0.0,
    'geoid_undulation_m': None,    # None: EGM96 from PROJ's network grid, else 0 with a warning
    'adsb_source': 'auto',         # 'auto', 'adsb.lol', 'adsb.aero' or 'file'
    'adsb_file': None,
    'match': {'coarse_step_m': 800.0, 'min_score': 0.5, 'workers': None, 'null_tests': 3, 'modes': 4,
              'events': None},
}
