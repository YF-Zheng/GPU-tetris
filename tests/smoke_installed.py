"""Run with a fresh venv's Python to verify installed entry points."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    binary = Path(sys.executable).parent
    cli = str(binary / 'tetris')
    smi = str(binary / 'tetris-smi')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    with tempfile.TemporaryDirectory() as root:
        env['TETRIS_POOL'] = str(Path(root) / 'pool')
        def run(*args, executable=cli):
            return subprocess.check_output([executable, *args], env=env, cwd=root, text=True, timeout=30).strip()
        run('init', '--simulate')
        run('config', '--set', 'observation_seconds=0', '--set', 'launch_cooldown_seconds=0',
            '--set', 'poll_seconds=0.1', '--set', 'simulate_gpu_count=4',
            '--set', 'queue_policy="priority"', '--set', 'placement_policy="pack"')
        run('start')
        try:
            key = run('submit', '--name', 'installed-wheel', '--gpu-count', '2', '--gpus', '1,3',
                      '--priority', '10', '--', sys.executable, '-c', 'print("portable install works")')
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                data = json.loads(run('--json', executable=smi))
                job = next(j for j in data['jobs'] if j['id'] == key)
                if job['status'] == 'complete':
                    break
                time.sleep(.1)
            assert job['status'] == 'complete', job
            assert data['schema_version'] == 2, data
            assert job['gpu_ids'] == [1, 3] and job['sharing'] == 'exclusive', job
            assert 'portable install works' in run('logs', key)
            run('config', '--set', 'filler_enabled=true', '--set', 'filler_utilization_threshold=30')
            filler = run('submit', '--filler', '--name', 'installed-filler', '--gpus', '0', '--',
                         sys.executable, '-c', 'print("managed filler works")')
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                data = json.loads(run('--json', executable=smi))
                job = next(j for j in data['jobs'] if j['id'] == filler)
                if job['status'] == 'complete':
                    break
                time.sleep(.1)
            assert job['status'] == 'complete' and job['priority_tier'] == 'lowest', job
            assert job['filler'] and job['gpu_ids'] == [0], job
            assert 'managed filler works' in run('logs', filler)
            print(run(executable=smi))
            print('installed wheel end-to-end: OK')
        finally:
            run('stop')


if __name__ == '__main__':
    main()
