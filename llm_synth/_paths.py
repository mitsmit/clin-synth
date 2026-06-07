"""Project-root override shared by the package and the web backend.

The project layout is "project-root-centric": config.yaml, domains/, data/,
processed_data/, frontend/, and .env all live alongside each other in one
directory that the rest of the code locates relative to. Each module has
historically derived that root from its own file location
(Path(__file__)...parents[N] / Path(__file__).resolve().parent.parent), which
only resolves correctly when running from a checkout — once installed into
site-packages those parent-walks land in the wrong place.

resolve_project_root() tries, in order:

  1. LLMSYNTH_ROOT environment variable — explicit override.
  2. Walk up from the current working directory looking for config.yaml —
     matches the existing "cd /path/to/project && uvicorn/llm-synth ..."
     convention exactly: in a checkout, CWD *is* the project root, so this
     finds config.yaml at depth zero and returns the same path the old
     Path(__file__)-relative computation did.
  3. The caller-supplied fallback (its own previous Path(__file__)-relative
     computation) — preserved as a last resort so behaviour never regresses
     versus before this helper existed.

This makes the change additive for the checkout/systemd deployment (step 2
resolves identically to the old per-file computation there) while letting an
installed package — where Path(__file__) lands in site-packages — locate the
project directory it's being run against.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_project_root(fallback: Path) -> Path:
    env_root = os.environ.get("LLMSYNTH_ROOT")
    if env_root:
        return Path(env_root).resolve()

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "config.yaml").exists():
            return candidate

    return fallback
