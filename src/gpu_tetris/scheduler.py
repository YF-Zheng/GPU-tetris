"""Single-owner scheduling loop with restart-safe worker supervision."""

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid

from .core import alive, allocated_gpus, exclusive, get_config, identity, require_schema, transaction
from .gpu import policies, query
from .filler import admission_views, prepare, recover, rearm_times
from .process import cleanup, detached, members
from .strategies import diagnose, order_jobs, select_gpus


def runtime(db, key, default=None):
    row = db.execute('SELECT value FROM runtime WHERE key=?', (key,)).fetchone()
    return json.loads(row['value']) if row else default


def put(db, key, value):
    db.execute('INSERT OR REPLACE INTO runtime VALUES (?,?)', (key, json.dumps(value)))


def reconcile(pool, db):
    for row in db.execute("SELECT * FROM jobs WHERE status IN ('running','starting')").fetchall():
        job = dict(row)
        token = job['token']
        receipt = pool / 'logs' / f'{job["id"]}.{token}.result.json'
        if alive(job['pid'], job['identity']):
            continue
        if receipt.exists():
            result = json.loads(receipt.read_text())
            db.execute('UPDATE jobs SET status=?,exit_code=?,error=? WHERE id=?',
                       (result['status'], result['exit_code'], result['error'], job['id']))
            continue
        child_file = pool / 'logs' / f'{job["id"]}.{token}.worker.json'
        child = json.loads(child_file.read_text()) if child_file.exists() else {}
        remaining = members(token)
        if alive(child.get('pid'), child.get('identity')) or remaining:
            if not job['cancel_requested']:
                db.execute('UPDATE jobs SET error=? WHERE id=?',
                           ('Worker supervisor lost; slot retained. Use cancel to clean up before retry.', job['id']))
                continue
            try:
                cleanup(token)
            except ValueError as exc:
                db.execute('UPDATE jobs SET error=? WHERE id=?', (str(exc), job['id']))
                continue
        db.execute('UPDATE jobs SET status=?,error=? WHERE id=?',
                   ('cancelled' if job['cancel_requested'] else 'failed',
                    'Interrupted attempt without exit receipt; inspect before retry', job['id']))


def tick(pool, history, last_launch, sample=None, now=None):
    with transaction(pool) as db:
        require_schema(db)
        config = get_config(db)
    started = time.time()
    try:
        gpus, apps = query(config['simulate'], config['simulate_gpu_count'],
                           query_apps=config['ownership_mode'] == 'strict') if sample is None else sample
        error = None
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        gpus, apps = [], []
        error = str(exc)
        history.clear()
    now = time.time() if now is None else now
    sampled_mode = config['ownership_mode']
    with transaction(pool) as db:
        config = get_config(db)
        if sample is None and config['ownership_mode'] != sampled_mode:
            gpus, apps = [], []
            error = 'GPU ownership mode changed during query; waiting for a fresh sample'
            history.clear()
        if runtime(db, 'policy_config') != config:
            history.clear()
            put(db, 'policy_config', config)
        last_launch.update(runtime(db, 'last_launch', {}))
        reconcile(pool, db)
        reclaimed = recover(db, history, last_launch, now)
        rearm = rearm_times(db)
        put(db, 'last_launch', last_launch)
        active = [dict(row) for row in db.execute("SELECT * FROM jobs WHERE status IN ('running','starting')")]
        views = policies(gpus, apps, active, history, last_launch, config, now)
        queued = [dict(row) for row in db.execute("SELECT * FROM jobs WHERE status IN ('queued','blocked') ORDER BY submitted,id")]
        demand = prepare(db, active, queued, views, config, now)
        reclaiming = set(reclaimed)
        for job in active:
            if json.loads(job['spec']).get('filler'):
                reclaiming.update(allocated_gpus(job))
        for gpu in views:
            if gpu['index'] in reclaiming:
                gpu['policy'] = 'FILLER / RECLAIM'
        for row in order_jobs(queued, config, now):
            current_config = get_config(db)
            if current_config != config:
                history.clear()
                break
            job = dict(db.execute('SELECT * FROM jobs WHERE id=?', (row['id'],)).fetchone())
            if job['status'] not in ('queued', 'blocked'):
                continue
            spec = json.loads(job['spec'])
            if spec.get('filler'):
                pending = [dict(r) for r in db.execute("SELECT * FROM jobs WHERE status IN ('queued','blocked')")]
                if any(not json.loads(r['spec']).get('filler') and not r['cancel_requested']
                       and r['id'] not in {q['id'] for q in queued} for r in pending):
                    break
                demand = prepare(db, active, pending, views, config, now)
            if job['cancel_requested']:
                db.execute('UPDATE jobs SET status=? WHERE id=?', ('cancelled', job['id']))
                continue
            dependencies = [db.execute('SELECT status FROM jobs WHERE id=?', (key,)).fetchone()
                            for key in spec['depends']]
            if any(dep is None or dep['status'] != 'complete' for dep in dependencies):
                db.execute('UPDATE jobs SET status=?,error=? WHERE id=?',
                           ('blocked', 'Waiting for successful dependencies', job['id']))
                if config['queue_policy'] == 'strict_fifo':
                    break
                continue
            db.execute('UPDATE jobs SET status=?,error=NULL WHERE id=?', ('queued', job['id']))
            candidates = admission_views(views, active, history, last_launch, config, now, demand | reclaimed, rearm) if spec.get('filler') else views
            selected = select_gpus(candidates, spec, config)
            if not selected:
                db.execute('UPDATE jobs SET error=? WHERE id=?',
                           (f'Waiting for {spec.get("gpu_count", 1)} eligible GPUs '
                            f'({"exclusive" if exclusive(spec) else "shared"}, '
                            f'{spec["min_free_mb"]} MiB per GPU): {diagnose(candidates, spec)}', job['id']))
                if config['queue_policy'] == 'strict_fifo':
                    break
                continue
            if sample is None and time.time() - started > config['poll_seconds']:
                break
            token = uuid.uuid4().hex
            attempt = job['attempt'] + 1
            log = pool / 'logs' / f'{job["id"]}.attempt{attempt}.log'
            ids = [gpu['index'] for gpu in selected]
            db.execute('UPDATE jobs SET status=?,gpu=?,gpu_ids=?,token=?,attempt=?,log=?,pid=NULL,identity=NULL,exit_code=NULL,error=NULL WHERE id=?',
                       ('starting', ids[0] if len(ids) == 1 else None, json.dumps(ids),
                        token, attempt, str(log), job['id']))
            put(db, 'binding:' + token, [gpu['uuid'] for gpu in selected])
            for gpu in selected:
                last_launch[str(gpu['index'])] = now
            put(db, 'last_launch', last_launch)
            db.commit()
            try:
                with log.open('a') as output:
                    worker = detached([sys.executable, '-m', 'gpu_tetris.worker',
                                       str(pool), job['id'], token],
                                      stdout=output, stderr=subprocess.STDOUT)
                worker_identity = identity(worker.pid)
                db.execute('BEGIN IMMEDIATE')
                db.execute('UPDATE jobs SET pid=?,identity=? WHERE id=? AND token=?',
                           (worker.pid, worker_identity, job['id'], token))
            except OSError as exc:
                if not db.in_transaction:
                    db.execute('BEGIN IMMEDIATE')
                db.execute('UPDATE jobs SET status=?,error=? WHERE id=?', ('failed', str(exc), job['id']))
            for gpu in selected:
                gpu['count'] += 1
                gpu['policy'] = ('EXCLUSIVE' if exclusive(spec) else
                                 'FULL' if gpu['count'] >= config['per_gpu_limit'] else 'OBSERVE / COOLDOWN')
                history[str(gpu['index'])] = []
                last_launch[str(gpu['index'])] = now
        put(db, 'snapshot', dict(updated_at=now, gpus=views, error=error, simulate=config['simulate']))


def serve(pool):
    with (pool / 'scheduler.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('A scheduler already owns this pool') from None
        stopping = False

        def stop(_signum, _frame):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        with transaction(pool) as db:
            put(db, 'daemon', dict(pid=os.getpid(), identity=identity(os.getpid())))
        history, last_launch = {}, {}
        try:
            while not stopping:
                tick(pool, history, last_launch)
                with transaction(pool) as db:
                    interval = get_config(db)['poll_seconds']
                deadline = time.monotonic() + interval
                while not stopping and time.monotonic() < deadline:
                    time.sleep(min(0.2, max(0, deadline - time.monotonic())))
        finally:
            with transaction(pool) as db:
                put(db, 'daemon', {})


def start(pool):
    with transaction(pool) as db:
        daemon = runtime(db, 'daemon', {})
        if alive(daemon.get('pid'), daemon.get('identity')):
            return daemon['pid']
    with (pool / 'scheduler.log').open('a') as output:
        child = detached([sys.executable, '-m', 'gpu_tetris', '--pool', str(pool), 'run'],
                         stdout=output, stderr=subprocess.STDOUT)
    for _ in range(100):
        with transaction(pool) as db:
            daemon = runtime(db, 'daemon', {})
        if alive(daemon.get('pid'), daemon.get('identity')):
            return daemon['pid']
        if child.poll() is not None:
            break
        time.sleep(0.05)
    raise ValueError(f'Scheduler failed to start; see {pool / "scheduler.log"}')


def stop(pool):
    with transaction(pool) as db:
        daemon = runtime(db, 'daemon', {})
        if alive(daemon.get('pid'), daemon.get('identity')):
            os.kill(daemon['pid'], signal.SIGTERM)
        else:
            return
    for _ in range(200):
        if not alive(daemon['pid'], daemon['identity']):
            return
        time.sleep(0.1)
    raise ValueError('Scheduler is still stopping; inspect scheduler.log')
