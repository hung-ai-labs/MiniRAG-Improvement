"""Deferred CE-equivalent strict judge, with immutable attempt-journal resume.

Same model, prompt, passes, concurrency, parsing and retry policy as +CE.
Successful records from prior immutable attempts seed the standard resume state.
"""
from __future__ import annotations

import os
from pathlib import Path
import platform
import sys

from reproduce.mhr_set import offline


def main():
    platform.uname()
    offline.setup()
    offline.exclusive_library_io()
    offline.guard(network=True)
    args, prior = [], []
    source = iter(sys.argv[1:])
    for arg in source:
        if arg == '--prior-journal':
            prior.append(Path(next(source)))
        else:
            args.append(arg)
    if prior:
        import judge_run
        original = judge_run.read_records
        target = Path(args[args.index('--run-dir') + 1]) / 'judge.jsonl'
        seeded = {}
        for path in prior:
            for key, row in original(path).items():
                if row['status'] == 'ok':
                    seeded[key] = row

        def read_records(path):
            current = original(path)
            return seeded | current if Path(path).resolve() == target.resolve() else current
        judge_run.read_records = read_records
    # Reuse the CE entry's environment sanitisation and exact judge CLI flags.
    sys.argv = [sys.argv[0]] + args
    from reproduce.mhr_ce.judge_entry import main as ce_main
    return ce_main()


if __name__ == '__main__':
    raise SystemExit(main())
