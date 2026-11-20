"""Python batch throughput against the native C++ driver (docs/benchmarks.md Section 8).

    python -m pulselob.bench [--events 5000000] [--repeat 3]

Generates a synthetic stream (two-sided scenario: trades on both venues, book updates, heartbeats),
writes it to a .plev file, then times on that same file:

* the native driver `pulselob-replay` (C++ only: file read, engine loop, no Python);
* `pulselob.Engine.run` on the memory-mapped file (zero-copy input, GIL released, output arrays
  allocated before the loop), inline and split modes.

The best of `repeat` runs is reported for each, with a fresh engine per run.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from . import Engine, EngineConfig, __version__, native_replay_path, open_plev, synth


def _best(fn, repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pulselob.bench")
    ap.add_argument("--events", type=int, default=5_000_000)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    # The two-sided scenario produces about 1,100 MarketEvents per simulated second.
    horizon = max(10.0, args.events / 1100.0)
    t0 = time.perf_counter()
    ds = synth.generate("two_sided", seed=1, horizon_s=horizon)
    events = ds.events[: args.events]
    gen_s = time.perf_counter() - t0
    cfg = EngineConfig.default()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.plev"
        from .plev import write_plev

        write_plev(path, events)
        (Path(tmp) / "cfg.txt").write_text(cfg.to_text())
        mm, _ = open_plev(path)
        n = int(mm.shape[0])

        exe = native_replay_path()
        out = subprocess.run([str(exe), "--config", str(Path(tmp) / "cfg.txt"), "--input", str(path),
                              "--repeat", str(args.repeat)], check=True, capture_output=True, text=True)
        native = json.loads(out.stdout.strip().splitlines()[-1])

        py_inline = _best(lambda: Engine(cfg).run(mm), args.repeat)
        py_split = _best(lambda: Engine(cfg, mode="split").run(mm), args.repeat)
        py_states = _best(lambda: Engine(cfg).run(mm, state_every=1000), args.repeat)

    res = {
        "version": __version__,
        "machine": f"{platform.machine()} {platform.platform()}",
        "events": n,
        "trades": int(np.count_nonzero(events["kind"] == 1)),
        "generation_seconds": round(gen_s, 3),
        "native_inline_events_per_s": native["events_per_second"],
        "python_inline_events_per_s": round(n / py_inline),
        "python_inline_vs_native": round((n / py_inline) / native["events_per_second"], 4),
        "python_split_events_per_s": round(n / py_split),
        "python_inline_with_states_events_per_s": round(n / py_states),
    }
    print(json.dumps(res, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
