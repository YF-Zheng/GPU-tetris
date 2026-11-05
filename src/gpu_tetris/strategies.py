"""Deterministic queue ordering and all-or-nothing GPU placement."""

import json

from .core import exclusive


def order_jobs(jobs, config, now):
    policy = config['queue_policy']

    def normal_key(job):
        spec = json.loads(job['spec'])
        count = spec.get('gpu_count', 1)
        tie = (job['submitted'], job['id'])
        if policy == 'priority':
            age = max(0, now - job['submitted']) // config['priority_aging_seconds']
            return (-(spec.get('priority', 0) + age), *tie)
        if policy == 'small_first':
            return (count, *tie)
        if policy == 'large_first':
            return (-count, *tie)
        return tie

    def key(job):
        if json.loads(job['spec']).get('filler', False):
            return (1, job['submitted'], job['id'])
        return (0, *normal_key(job))

    return sorted(jobs, key=key)


def select_gpus(views, spec, config):
    candidates = [g for g in views if g['policy'] == 'READY'
                  and g['free_mb'] >= spec['min_free_mb']
                  and (spec['allowed_gpus'] is None or g['index'] in spec['allowed_gpus'])
                  and (not exclusive(spec) or g['count'] == 0)]
    policy = config['placement_policy']
    keys = {
        'balanced': lambda g: (g['count'], g['util'], -g['free_mb'], g['index']),
        'pack': lambda g: (-g['count'], g['free_mb'], g['util'], g['index']),
        'spread': lambda g: (g['count'], -g['free_mb'], g['util'], g['index']),
        'most_free': lambda g: (-g['free_mb'], g['count'], g['util'], g['index']),
    }
    count = spec.get('gpu_count', 1)
    selected = sorted(candidates, key=keys[policy])[:count]
    return sorted(selected, key=lambda g: g['index']) if len(selected) == count else []


def diagnose(views, spec):
    """Explain per GPU why a job cannot be placed (used in the task DETAIL column)."""
    parts = []
    for g in views:
        if spec['allowed_gpus'] is not None and g['index'] not in spec['allowed_gpus']:
            reason = 'NOT IN --gpus'
        elif g['policy'] != 'READY':
            reason = g['policy']
        elif g['free_mb'] < spec['min_free_mb']:
            reason = f'LOW MEMORY {g["free_mb"]}<{spec["min_free_mb"]} MiB'
        elif exclusive(spec) and g['count'] != 0:
            reason = f'NOT EMPTY ({g["count"]} tasks, exclusive needed)'
        else:
            reason = 'READY'
        parts.append(f'GPU{g["index"]}: {reason}')
    return '; '.join(parts) if parts else 'no GPUs reported'
