"""One router per concern. `app.py` mounts them all under /api."""

from . import allocate, instruments, settings, simulate, snapshot, transactions

ROUTERS = [settings.router, snapshot.router, instruments.router,
           transactions.router, simulate.router, allocate.router]

__all__ = ["ROUTERS"]
