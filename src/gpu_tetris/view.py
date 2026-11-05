"""Read-only queue snapshots and terminal rendering."""

from collections import Counter
import json
import time
import unicodedata

from .core import alive, allocated_gpus, connect, exclusive, get_config
from .scheduler import runtime


def safe(value):
    return ''.join(c if not unicodedata.category(c).startswith('C') else '?' for c in str(value))


def snapshot(pool):
    db = connect(pool)
    try:
        db.execute('BEGIN')
        config = get_config(db)
        daemon = runtime(db, 'daemon', {})
        data = runtime(db, 'snapshot', {})
        jobs = []
        for row in db.execute('SELECT * FROM jobs ORDER BY submitted,id'):
            job = dict(row)
            spec = json.loads(job.pop('spec'))
            job.pop('identity', None)
            job.pop('token', None)
            job['command'] = spec['command']
            job['cwd'] = spec['cwd']
            job['depends'] = spec['depends']
            job['gpu_ids'] = allocated_gpus(job)
            job['gpu_count'] = spec.get('gpu_count', 1)
            job['filler'] = spec.get('filler', False)
            job['priority'] = None if job['filler'] else spec.get('priority', 0)
            job['priority_tier'] = 'lowest' if job['filler'] else 'normal'
            job['sharing'] = 'exclusive' if exclusive(spec) else 'shared'
            jobs.append(job)
        return dict(schema_version=2, pool=str(pool), config=config, jobs=jobs,
                    counts=dict(Counter(j['status'] for j in jobs)),
                    daemon_running=alive(daemon.get('pid'), daemon.get('identity')), **data)
    finally:
        db.close()


def table(headers, rows):
    rows = [[safe(cell) for cell in row] for row in [headers, *rows]]
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    border = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
    lines = [border]
    for index, row in enumerate(rows):
        lines.append('| ' + ' | '.join(v.ljust(w) for v, w in zip(row, widths)) + ' |')
        if index == 0:
            lines.append(border)
    return '\n'.join([*lines, border])


def render(data, all_jobs=False, status=None):
    config = data['config']
    updated = data.get('updated_at')
    age = max(0, time.time() - updated) if isinstance(updated, (int, float)) else None
    stale = age is None or age > config['poll_seconds'] * 3 or not data['daemon_running']
    lines = [f'GPU TETRIS POOL  {safe(data["pool"])}',
             f'SCHEDULER {"RUNNING" if data["daemon_running"] else "STOPPED"}  '
             f'HEARTBEAT {str(round(age, 1)) + "s" if age is not None else "unknown"}  '
             + '  '.join(f'{k.upper()} {v}' for k, v in sorted(data['counts'].items()))]
    lines.append(f'QUEUE {safe(config["queue_policy"])}  PLACEMENT {safe(config["placement_policy"])}')
    if config['simulate']:
        lines.append('SIMULATION MODE: CPU processes only; CUDA_VISIBLE_DEVICES is empty.')
    if stale:
        lines.append('WARNING: STALE or missing scheduler heartbeat; resource policies are not current.')
    if data.get('error'):
        lines.append('GPU QUERY ERROR: ' + safe(data['error']))
    lines.append('GPU metrics and task counts: scheduler snapshot. READY means resource checks only.')
    lines.append(f'FILLER {"ENABLED" if config["filler_enabled"] else "DISABLED"}  '
                 f'ADMISSION <{config["filler_utilization_threshold"]}%')
    lines.append(table(['GPU', 'UTIL', 'FREE MiB', f'TASKS/{config["per_gpu_limit"]}', 'REFILL POLICY'],
                       [[g['index'], str(g['util']) + '%', g['free_mb'],
                         f'{g["count"]}/{config["per_gpu_limit"]}',
                         'UNKNOWN / STALE' if stale else g['policy']] for g in data.get('gpus', [])]))
    visible = [j for j in data['jobs'] if (status is None or j['status'] in status)
               and (all_jobs or status is not None or j['status'] != 'complete')]
    lines.append(table(['ID', 'NAME', 'TYPE', 'STATUS', 'GPUs', 'NEED', 'SHARING', 'PRI', 'PID', 'TRY', 'DETAIL'],
                       [[j['id'], j['name'], 'FILLER' if j['filler'] else 'TRAIN', j['status'], ','.join(map(str, j['gpu_ids'])) or '-',
                         j['gpu_count'], j['sharing'], 'LOWEST' if j['filler'] else j['priority'], j['pid'], j['attempt'],
                         j['error'] or ('cancel requested' if j['cancel_requested'] else '')] for j in visible]))
    return '\n'.join(lines)
