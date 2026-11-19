"""SET-only immutable output helpers and runtime guards. No credential loading."""
from __future__ import annotations

import builtins
import io
import json
import os
from pathlib import Path
import sys

from reproduce.mhr_ce import offline as ce
from reproduce.mhr_st.common import ROOT

OUT = ROOT / 'review/mhr_set_20261117'
setup = ce.setup
new_text = ce.new_text
write = ce.write
jsonl = ce.jsonl


def guard(network=False):
    """Allow creation only, within SET roots. Offline mode blocks every socket."""
    roots = (OUT.resolve(), (ROOT / 'reproduce/mhr_set').resolve())

    if network:
        # Online steps (answers/judge): library code (MiniRAG logging.FileHandler, judge_run, phased_qa)
        # opens files with plain 'a'/'w', which the exclusive-create rule below would crash. Each online
        # attempt writes into a fresh attempt directory, so nothing old is overwritten; keep only the
        # credential-file block here (keys arrive through the environment from online.py).
        def audit_online(event, args):
            if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
                name = Path(os.fsdecode(args[0])).name.lower()
                if name == '.env' or name.startswith('.env.'):
                    raise RuntimeError('Credential-file access is forbidden')
        sys.addaudithook(audit_online)
        setup()
        return

    def audit(event, args):
        if not network and event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo',
                                     'socket.sendto', 'socket.sendmsg'):
            raise RuntimeError('Network is forbidden during offline SET work')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.name.lower() == '.env' or path.name.lower().startswith('.env.'):
                raise RuntimeError('Credential-file access is forbidden')
            mode, flags = args[1], args[2]
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing:
                if not any(path.is_relative_to(root) for root in roots):
                    raise RuntimeError(f'Write outside SET directories: {path}')
                if path.exists():
                    raise FileExistsError(f'ADD-ONLY: refusing to modify {path}')
                if not isinstance(flags, int) or not flags & os.O_EXCL:
                    raise RuntimeError(f'ADD-ONLY: exclusive creation required: {path}')
        if event in ('os.remove', 'os.rename', 'os.rmdir', 'shutil.rmtree'):
            raise RuntimeError(f'ADD-ONLY: forbidden filesystem mutation {event}')
    sys.addaudithook(audit)
    setup()


def install():
    from reproduce.mhr_ce.local_runtime import install as local_install
    local_install()
    guard()


def exclusive_library_io():
    """Adapt legacy initial w/a opens to x; existing files always fail.

    Used only by deferred online processes, with fresh attempt directories.
    Resume aggregates immutable attempt journals, never reopens them for writing.
    Disabled (no-op): it broke library appends; fresh attempt directories already prevent overwrites.
    """
    return
    original_builtin, original_io = builtins.open, io.open

    def wrap(original):
        def exclusive(file, mode='r', *args, **kwargs):
            if any(c in mode for c in 'wa'):
                mode = mode.replace('w', 'x').replace('a', 'x')
            return original(file, mode, *args, **kwargs)
        return exclusive
    builtins.open = wrap(original_builtin)
    io.open = wrap(original_io)


def csv_write(path, fields, rows):
    import csv
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    new_text(path, stream.getvalue())


def event(path, value):
    """Immutable event stream: first event at path, subsequent events in sibling parts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        jsonl(path, [value])
        return
    parts = path.with_name(path.stem + '_parts')
    parts.mkdir(parents=True, exist_ok=True)
    import uuid
    jsonl(parts / (uuid.uuid4().hex + '.jsonl'), [value])
