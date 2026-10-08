from __future__ import annotations

import os

# Many engine processes share the host under `-n 16`; worker-thread
# confinement would stack them on the cores that were idle at launch time.
os.environ["NUMSIM_WORKER_AFFINITY"] = "off"
