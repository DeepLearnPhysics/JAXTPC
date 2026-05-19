"""Load SCE configuration and detector presets."""

import os
import yaml


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = os.path.join(os.path.dirname(_THIS_DIR), "config")
_SCE_CONFIG_PATH = os.path.join(_CONFIG_DIR, "sce_config.yaml")
_PRESETS_PATH = os.path.join(_CONFIG_DIR, "detector_presets.yaml")

EPSILON_0 = 8.854187817e-12  # F/m  (vacuum permittivity)


def load_detector_presets(presets_path=None):
    """Load detector presets YAML.  Returns dict[name -> params]."""
    path = presets_path or _PRESETS_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_params(preset=None, overrides=None):
    """Build a complete parameter dict for the SCE simulation.

    Starts from the defaults in ``sce_config.yaml``, optionally applies a
    named detector preset (which supplies Lx/Ly/Lz/E0/Q), then applies
    any explicit *overrides*.

    Parameters
    ----------
    preset : str or None
        Detector preset name (e.g. ``'microboone'``, ``'sbnd'``).
    overrides : dict or None
        Arbitrary key-value overrides applied last.

    Returns
    -------
    params : dict
        Complete parameter set including derived quantities
        ``epsilon`` (F/m) and ``v_ion`` (cm/s).
    """
    with open(_SCE_CONFIG_PATH, "r") as f:
        params = yaml.safe_load(f)

    if preset is not None:
        presets = load_detector_presets()
        if preset not in presets:
            raise ValueError(
                f"Unknown preset '{preset}'. "
                f"Available: {list(presets.keys())}"
            )
        params.update(presets[preset])

    if overrides:
        params.update(overrides)

    for key in ("Lx", "Ly", "Lz"):
        if key not in params:
            raise ValueError(
                f"Missing '{key}'. Provide a preset or pass it in overrides."
            )

    params["epsilon"] = EPSILON_0 * params["epsilon_r"]
    params["v_ion"] = params["mu_ion"] * params["E0"]

    return params
