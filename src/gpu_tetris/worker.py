"""Detached per-attempt supervisor; survives scheduler shutdown."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .core import atomic_json, get_config, get_job, identity, transaction
from .process import cleanup


def run(pool, job_id, token):
    child = None
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with transaction(pool) as db:
        job = get_job(db, job_id)
        if job['token'] != token or job['status'] != 'starting':
            return
        if job['cancel_requested']:
            db.execute('UPDATE jobs SET status=? WHERE id=?', ('cancelled', job_id))
            return
        config = get_config(db)
        spec = json.loads(job['spec'])
        db.execute('UPDATE jobs SET pid=?, identity=? WHERE id=?',
                   (os.getpid(), identity(os.getpid()), job_id))
        binding = json.loads(db.execute('SELECT value FROM runtime WHERE key=?',
                                        ('binding:' + token,)).fetchone()['value'])
    env = os.environ.copy()
    env.update(spec['env'])
    binding = [binding] if isinstance(binding, str) else binding
    env['CUDA_VISIBLE_DEVICES'] = '' if config['simulate'] else ','.join(binding)
    env['TETRIS_GPU_COUNT'] = str(spec.get('gpu_count', 1))
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    env['TETRIS_JOB_ID'] = job_id
    env['TETRIS_ATTEMPT_TOKEN'] = token
    # Deprecated pre-rename aliases for existing task scripts.
    env['UNIFIED_GPU_COUNT'], env['UNIFIED_JOB_ID'], env['UNIFIED_ATTEMPT_TOKEN'] = env['TETRIS_GPU_COUNT'], job_id, token
    read_fd, write_fd = os.pipe()
    receipt = pool / 'logs' / f'{job_id}.{token}.result.json'
    runtime_file = pool / 'logs' / f'{job_id}.{token}.worker.json'
    code = None
    error = None
    cancelled = False
    try:
        with transaction(pool) as db:
            current = get_job(db, job_id)
            if current['cancel_requested']:
                cancelled = True
            else:
                with Path(job['log']).open('a') as log:
                    child = subprocess.Popen([sys.executable, '-m', 'gpu_tetris.process',
                                              str(read_fd), *spec['command']],
                                             cwd=spec['cwd'], env=env, pass_fds=(read_fd,),
                                             stdout=log, stderr=subprocess.STDOUT)
                atomic_json(runtime_file, dict(pid=child.pid, identity=identity(child.pid)))
                db.execute('UPDATE jobs SET status=? WHERE id=?', ('running', job_id))
        os.close(read_fd)
        read_fd = None
        if child is not None:
            os.write(write_fd, b'1')
        os.close(write_fd)
        write_fd = None
        while child is not None and child.poll() is None:
            with transaction(pool) as db:
                cancelled = bool(get_job(db, job_id)['cancel_requested']) or stopping
            if cancelled:
                cleanup(token, os.getpid())
                break
            time.sleep(0.2)
        if child is not None:
            code = child.wait()
        cleanup(token, os.getpid())
    except (OSError, ValueError) as exc:
        error = str(exc)
        cleanup(token, os.getpid())
        if child is not None:
            child.wait(timeout=5)
    finally:
        for fd in (read_fd, write_fd):
            if fd is not None:
                os.close(fd)
    with transaction(pool) as db:
        current = get_job(db, job_id)
        if current['token'] != token:
            return
        cancelled = cancelled or bool(current['cancel_requested'])
        status = 'cancelled' if cancelled or stopping else ('complete' if code == 0 and error is None else 'failed')
        result = dict(status=status, exit_code=code, error=error, ended_at=time.time())
        atomic_json(receipt, result)
        db.execute('UPDATE jobs SET status=?,exit_code=?,error=? WHERE id=? AND token=?',
                   (status, code, error, job_id, token))


if __name__ == '__main__':
    run(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
