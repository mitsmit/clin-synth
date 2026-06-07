"""Central config loader. All modules import from here — no scattered yaml.safe_load calls."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml

from llm_synth._paths import resolve_project_root

# Default: config.yaml sits one level above the package directory (project root),
# unless LLMSYNTH_ROOT is set to point at an external project directory.
_DEFAULT_CONFIG = resolve_project_root(Path(__file__).resolve().parent.parent) / "config.yaml"

# Default domain — diabetes.yaml is used when the caller has not called set_domain().
# set_domain() overrides this.
_DEFAULT_DOMAIN = _DEFAULT_CONFIG.parent / "domains" / "diabetes.yaml"
_domain_path: Path | None = _DEFAULT_DOMAIN if _DEFAULT_DOMAIN.exists() else None

# Top-level keys expected in every domain config. Missing keys produce a warning
# so callers can catch incomplete domain files early rather than getting silent defaults.
_EXPECTED_DOMAIN_KEYS: frozenset[str] = frozenset({
    "schema",
    "tstr",
    "clinical_plausibility",
})


def set_domain(path: Path | str) -> None:
    """Activate a domain config that will be merged on top of config.yaml.

    Must be called before the first import of generate, validate, tstr, or
    clinical_plausibility, since those modules load config at module level.
    """
    global _domain_path
    _domain_path = Path(path)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Lists are replaced wholesale, not merged."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load and return the merged pipeline + domain YAML config.

    Merge order: config.yaml (pipeline defaults) ← domain config (domain overrides).

    Args:
        path: explicit path to config.yaml; defaults to <project_root>/config.yaml.
    """
    p = Path(path) if path else _DEFAULT_CONFIG
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    if _domain_path is not None:
        if not _domain_path.exists():
            raise FileNotFoundError(f"Domain config not found: {_domain_path.resolve()}")
        domain_cfg = yaml.safe_load(_domain_path.read_text(encoding="utf-8")) or {}
        missing = _EXPECTED_DOMAIN_KEYS - set(domain_cfg.keys())
        if missing:
            warnings.warn(
                f"Domain config {_domain_path.name!r} is missing expected section(s): "
                f"{sorted(missing)}. Pipeline will fall back to base config defaults.",
                stacklevel=2,
            )
        cfg = _deep_merge(cfg, domain_cfg)

    return cfg


def get_root(config_path: Path | str | None = None) -> Path:
    """Return the project root directory (parent of config.yaml)."""
    p = Path(config_path) if config_path else _DEFAULT_CONFIG
    return p.parent


def get_domain_name() -> str | None:
    """Return the active domain name (stem of the domain YAML path), or None."""
    return _domain_path.stem if _domain_path is not None else None
