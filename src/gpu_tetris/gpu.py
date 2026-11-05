"""Bounded NVIDIA queries; no process environment scanning."""

import csv
import json
import subprocess

from .core import allocated_gpus, exclusive


def query(simulate=False, simulate_gpu_count=1, *, query_apps=True):
    if simulate:
        return [dict(index=i, uuid=f'SIMULATED-CPU-{i}', free_mb=1048576, util=0)
                for i in range(simulate_gpu_count)], []
    def run(fields, kind):
        result = subprocess.run(['nvidia-smi', f'--query-{kind}={fields}',
                                 '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=10, check=True)
        return [row for row in csv.reader(result.stdout.splitlines()) if row and any(v.strip() for v in row)]
    gpus = []
    for row in run('index,uuid,memory.free,utilization.gpu', 'gpu'):
        try:
            i, u, m, v = (value.strip() for value in row)
            gpu = dict(index=int(i), uuid=u, free_mb=int(m), util=int(v))
            if gpu['index'] < 0 or not u.startswith('GPU-') or gpu['free_mb'] < 0 or not 0 <= gpu['util'] <= 100:
                raise ValueError()
        except ValueError:
            raise ValueError(f'Invalid or unavailable NVIDIA GPU metrics: {row!r}') from None
        gpus.append(gpu)
    apps = []
    if query_apps:
        for row in run('pid,gpu_uuid', 'compute-apps'):
            try:
                pid, uid = (value.strip() for value in row)
                if int(pid) <= 0 or not uid.startswith(('GPU-', 'MIG-')):
                    raise ValueError()
                apps.append((int(pid), uid))
            except ValueError:
                raise ValueError(f'Invalid NVIDIA compute process data: {row!r}') from None
    if not gpus:
        raise ValueError('No NVIDIA GPUs detected')
    return gpus, apps


def process_group(pid):
    from pathlib import Path
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        return int(fields[2])
    except (OSError, ValueError, IndexError):
        return None


def policies(gpus, apps, active, history, last_launch, config, now):
    result = []
    known_groups = {row['pid'] for row in active if row['pid']}
    unknown = ({uid for pid, uid in apps if process_group(pid) not in known_groups}
               if config['ownership_mode'] == 'strict' else set())
    for gpu in gpus:
        index = gpu['index']
        key = str(index)
        samples = history.setdefault(key, [])
        if samples and (now <= samples[-1][0] or now - samples[-1][0] > config['poll_seconds'] * 2.5):
            samples.clear()
        samples.append([now, gpu['util']])
        cutoff = now - config['observation_seconds']
        while len(samples) > 1 and samples[1][0] <= cutoff:
            samples.pop(0)
        occupants = [row for row in active if index in allocated_gpus(row)]
        count = len(occupants)
        reserved = any(exclusive(json.loads(row.get('spec', '{}'))) for row in occupants)
        stable = (now - samples[0][0] >= config['observation_seconds']
                  and all(util < config['utilization_threshold'] for _, util in samples))
        if not config['launches_enabled']:
            policy = 'PAUSED'
        elif config['allowed_gpus'] is not None and index not in config['allowed_gpus']:
            policy = 'DISABLED'
        elif gpu['uuid'] in unknown:
            policy = 'UNKNOWN OWNER'
        elif reserved:
            policy = 'EXCLUSIVE'
        elif count >= config['per_gpu_limit']:
            policy = 'FULL'
        elif gpu['util'] >= config['utilization_threshold']:
            policy = f'HOLD >={config["utilization_threshold"]}%'
        elif gpu['free_mb'] < config['min_free_mb']:
            policy = 'LOW MEMORY'
        elif not stable or now - last_launch.get(key, -1e30) < config['launch_cooldown_seconds']:
            policy = 'OBSERVE / COOLDOWN'
        else:
            policy = 'READY'
        result.append(dict(gpu, count=count, policy=policy))
    return result
