"""Attempt-scoped process cleanup and gated execution."""

import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from .core import alive, identity
from .gpu import process_group


def detached(command, **kwargs):
    child = subprocess.Popen(command, start_new_session=True, **kwargs)
    threading.Thread(target=child.wait, daemon=True).start()
    return child


def members(token, group=None):
    found = {}
    # Attempts launched by a pre-rename worker carry the old variable name.
    markers = {f'TETRIS_ATTEMPT_TOKEN={token}'.encode(), f'UNIFIED_ATTEMPT_TOKEN={token}'.encode()}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            pid = int(path.name)
            same_group = group is not None and process_group(pid) == group
            if same_group or markers & set((path / 'environ').read_bytes().split(b'\0')):
                stamp = identity(pid)
                if stamp:
                    found[pid] = stamp
        except (OSError, ValueError):
            continue
    return found


def signal_members(token, sig, group=None):
    for pid, stamp in members(token, group).items():
        if alive(pid, stamp):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


def cleanup(token, group=None):
    signal_members(token, signal.SIGTERM, group)
    deadline = time.monotonic() + 2
    while members(token, group) and time.monotonic() < deadline:
        time.sleep(0.05)
    signal_members(token, signal.SIGKILL, group)
    deadline = time.monotonic() + 2
    while members(token, group) and time.monotonic() < deadline:
        time.sleep(0.05)
    if members(token, group):
        raise ValueError('Attempt processes remain alive; GPU slot must be retained')


def gate(fd, command):
    allowed = os.read(fd, 1)
    os.close(fd)
    if allowed != b'1':
        return 125
    os.execvpe(command[0], command, os.environ)
    return 127


if __name__ == '__main__':
    import sys
    raise SystemExit(gate(int(sys.argv[1]), sys.argv[2:]))
