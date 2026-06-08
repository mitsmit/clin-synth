"""
Shared pytest setup.

clin_synth.generate instantiates an OpenAI client at *import time*
(`client = _make_client()`), which raises if OPENAI_API_KEY is unset —
even for tests that only need pure helper functions like parse_csv_rows.
Set a placeholder key before any clin_synth module is imported so the
package can be imported in CI/dev environments with no real credentials.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
