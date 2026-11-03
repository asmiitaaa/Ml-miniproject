"""Shared, explicit timing contracts for checkpoint training and inference."""
TIMING_PROFILES = {
    "v1": {"decision_interval_s": 5., "min_green_s": 10., "yellow_s": 3.},
    "v2": {"decision_interval_s": 3., "min_green_s": 12., "yellow_s": 6.},
}


def timing_settings(options):
    profile = options.get("timing_profile", "v1")
    if profile not in TIMING_PROFILES:
        raise ValueError("timing_profile must be v1 (legacy replay) or v2")
    settings = TIMING_PROFILES[profile]
    for key, expected in settings.items():
        if key in options and float(options[key]) != expected:
            raise ValueError(f"{profile} requires {key}={expected:g}; do not change checkpoint timing at evaluation time")
    return profile, settings


def validate_checkpoint_timing(model, profile):
    recorded = getattr(model, "timing_profile", None)
    if recorded is not None and recorded != profile:
        raise ValueError(f"Checkpoint timing_profile={recorded} cannot be evaluated as {profile}")
    if profile == "v2":
        if recorded != "v2" or getattr(model, "training_mode", None) != "sumo":
            raise ValueError("v2 requires a newly SUMO-trained checkpoint with embedded timing metadata; retrain using rl_agent/train.py. Do not rename a v1 checkpoint.")
        if getattr(model, "training_timing", None) != TIMING_PROFILES["v2"]:
            raise ValueError("Checkpoint training_timing does not match the v2 timing contract")
