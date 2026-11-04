"""Module execution entrypoint (`python3 -m panflat`)."""

from __future__ import annotations

import sys

from panflat.cli import main

if __name__ == "__main__":
    sys.exit(main())
