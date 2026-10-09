"""Stable CLI/import entry for the formal pipeline's separated operational stages."""
import sys
from scripts.production import runtime as _implementation

if __name__ == '__main__':
    _implementation.cli()
else:
    # Retain existing public helpers, import paths and injected test services.
    sys.modules[__name__] = _implementation
