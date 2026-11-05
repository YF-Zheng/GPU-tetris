"""Lowest-tier filler admission and restart-safe cooperative reclamation."""

import json

from .core import alive, allocated_gpus
from .process import members
from .strategies import order_jobs, select_gpus


def is_filler(job):
    return json.loads(job['spec']).get('filler', False)


def mark_key(job_id):
    return 'filler_reclaim:' + job_id


def request_reclaim(db, job, reason):
    if job['cancel_requested']:
        return
    db.execute('INSERT OR REPLACE INTO runtime VALUES (?,?)',
               (mark_key(job['id']), json.dumps(dict(token=job['token'], reason=reason))))
    db.execute('UPDATE jobs SET cancel_requested=1,error=? WHERE id=?', (reason, job['id']))


REARM_KEY = 'filler_rearm'


def rearm_times(db):
    row = db.execute('SELECT value FROM runtime WHERE key=?', (REARM_KEY,)).fetchone()
    return json.loads(row['value']) if row else {}


def recover(db, history, last_launch, now):
    reclaimed = set()
    rearm = rearm_times(db)
    for row in db.execute("SELECT * FROM jobs WHERE status IN ('complete','failed','cancelled')").fetchall():
        job = dict(row)
        if not is_filler(job):
            continue
        key = mark_key(job['id'])
        marker = db.execute('SELECT value FROM runtime WHERE key=?', (key,)).fetchone()
        if marker is None:
            continue
        intent = json.loads(marker['value'])
        if intent['token'] != job['token']:
            db.execute('DELETE FROM runtime WHERE key=?', (key,))
            continue
        if alive(job['pid'], job['identity']) or members(job['token']):
            reclaimed.update(allocated_gpus(job))
            continue
        ids = allocated_gpus(job)
        reclaimed.update(ids)
        for index in ids:
            history[str(index)] = []
            last_launch[str(index)] = now
            rearm[str(index)] = now
        db.execute('UPDATE jobs SET status=?,cancel_requested=0,pid=NULL,identity=NULL,token=NULL,'
                   'gpu=NULL,gpu_ids=NULL,exit_code=NULL,error=? WHERE id=?',
                   ('queued', 'Filler reclaimed; waiting for spare capacity', job['id']))
        db.execute('DELETE FROM runtime WHERE key=?', (key,))
    # Persist so the re-arm hysteresis survives scheduler restarts.
    db.execute('INSERT OR REPLACE INTO runtime VALUES (?,?)', (REARM_KEY, json.dumps(rearm)))
    return reclaimed


def eligible_indices(spec, views, config):
    return {g['index'] for g in views
            if (config['allowed_gpus'] is None or g['index'] in config['allowed_gpus'])
            and (spec['allowed_gpus'] is None or g['index'] in spec['allowed_gpus'])}


def prepare(db, active, queued, views, config, now):
    demand = set()
    targets = set()
    fillers = {i for job in active if is_filler(job) for i in allocated_gpus(job)}
    planned = []
    for gpu in views:
        view = dict(gpu)
        if gpu['index'] in fillers:
            view['policy'] = 'FILLER WAIT'
        planned.append(view)
    for job in order_jobs(queued, config, now):
        if is_filler(job) or job['cancel_requested']:
            continue
        spec = json.loads(job['spec'])
        deps = [db.execute('SELECT status FROM jobs WHERE id=?', (key,)).fetchone() for key in spec['depends']]
        if any(dep is None or dep['status'] != 'complete' for dep in deps):
            if config['queue_policy'] == 'strict_fifo':
                break
            continue
        selected = select_gpus(planned, spec, config)
        if selected:
            for gpu in selected:
                gpu['policy'] = 'PLANNED NORMAL'
            continue
        indices = eligible_indices(spec, views, config)
        demand.update(indices)
        alternatives = []
        for gpu in planned:
            view = dict(gpu)
            if gpu['index'] in fillers:
                view.update(policy='READY', count=max(0, gpu['count'] - 1),
                            free_mb=max(gpu['free_mb'], spec['min_free_mb']))
            alternatives.append(view)
        selected = select_gpus(alternatives, spec, config)
        targets.update(g['index'] for g in selected if g['index'] in fillers)
        if config['queue_policy'] == 'strict_fifo':
            break
    for job in active:
        if not is_filler(job):
            continue
        ids = set(allocated_gpus(job))
        if not config['filler_enabled']:
            request_reclaim(db, job, 'Filler disabled; reclaim requested')
        elif not ids <= eligible_indices(json.loads(job['spec']), views, config):
            request_reclaim(db, job, 'Filler GPU disabled; reclaim requested')
        elif config['launches_enabled'] and ids & targets:
            request_reclaim(db, job, 'Normal task demand; filler reclaim requested')
    return demand


def admission_views(views, active, history, last_launch, config, now, excluded, rearm=None):
    rearm = rearm or {}
    occupied = {i for job in active if is_filler(job) for i in allocated_gpus(job)}
    threshold = config['filler_utilization_threshold']
    result = []
    for gpu in views:
        index = gpu['index']
        samples = history.get(str(index), [])
        stable = (samples and now - samples[0][0] >= config['observation_seconds']
                  and all(util < threshold for _, util in samples))
        resource_safe = (gpu['policy'] in ('READY', 'OBSERVE / COOLDOWN') or gpu['policy'].startswith('HOLD >='))
        ready = (config['filler_enabled'] and index not in occupied | excluded
                 and resource_safe and gpu['free_mb'] >= config['min_free_mb'] and stable
                 and now - last_launch.get(str(index), -1e30) >= config['launch_cooldown_seconds']
                 # After a reclaim, wait filler_rearm_seconds before re-admitting a filler on that GPU.
                 and now - rearm.get(str(index), -1e30) >= config['filler_rearm_seconds'])
        result.append(dict(gpu, policy='READY' if ready else 'FILLER WAIT'))
    return result
