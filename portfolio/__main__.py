"""`python -m portfolio serve` is the same as `portfolio serve`."""

import sys

from .cli import main

sys.exit(main())
