"""Public terminal commands."""

import argparse
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from . import __version__
from .core import TERMINAL, get_config, get_job, initialize, pool_path, require_schema, submit, transaction, upgrade, validate_config
from .scheduler import serve, start, stop
from .view import render, snapshot


def gpu_list(text):
    try:
        values = [int(v) for v in text.split(',')]
        if not values or any(v < 0 for v in values):
            raise ValueError()
        return sorted(set(values))
    except ValueError:
        raise argparse.ArgumentTypeError('Expected comma-separated nonnegative GPU indices') from None


def parser():
    ap = argparse.ArgumentParser(description='Local persistent GPU queue. Linux, Python 3.10+, no Python dependencies.')
    ap.add_argument('--version', action='version', version=__version__)
    ap.add_argument('--pool', help='Pool directory (or TETRIS_POOL / legacy UNIFIED_GPU_POOL; default: XDG state directory)')
    subs = ap.add_subparsers(dest='action', required=True)
    for name in ('init', 'upgrade', 'start', 'run', 'stop', 'restart', 'status', 'submit', 'logs', 'cancel', 'retry', 'config', 'pause', 'resume', 'doctor', 'keepers'):
        sub = subs.add_parser(name)
        sub.add_argument('--pool', default=argparse.SUPPRESS)
        if name == 'init':
            sub.add_argument('--simulate', action='store_true', help='Explicit CPU-only simulation mode')
        elif name == 'submit':
            sub.add_argument('--name')
            sub.add_argument('--cwd')
            sub.add_argument('--env', action='append', default=[], metavar='KEY=VALUE')
            sub.add_argument('--depends', action='append', default=[], metavar='JOB_ID')
            sub.add_argument('--gpus', type=gpu_list, help='Allowed GPU indices, not a GPU count')
            sub.add_argument('--gpu-count', type=int, default=1, help='Number of GPUs allocated atomically')
            sub.add_argument('--priority', type=int, default=0, help='Higher runs first under priority policy')
            sub.add_argument('--filler', action='store_true', help='Preemptible single-GPU filler, always lowest tier')
            sub.add_argument('--keeper', metavar='PRESET', help='Built-in keeper preset (implies --filler); see: keepers. Extra options go after --')
            sub.add_argument('--keeper-python', metavar='PYTHON', help='Python with PyTorch installed, used to run the keeper (default: the interpreter running this tool)')
            sub.add_argument('--per-gpu', action='store_true', help='With --keeper: submit one keeper pinned to each GPU in --gpus')
            sub.add_argument('--sharing', choices=('exclusive', 'shared'), help='Default: multi-GPU exclusive, single-GPU shared')
            sub.add_argument('--min-free-mb', type=int, default=0, help='Minimum free MiB on EACH GPU')
            sub.add_argument('command', nargs=argparse.REMAINDER, help='-- PROGRAM ARG ... (no implicit shell)')
        elif name == 'status':
            sub.add_argument('--watch', nargs='?', const=5.0, type=float)
            sub.add_argument('--json', action='store_true')
            sub.add_argument('--all', action='store_true')
            sub.add_argument('--status', action='append', help='Filter displayed jobs by status')
            sub.add_argument('--no-live', action='store_true', help='Compatibility option; metrics always use scheduler snapshots')
        elif name in ('logs', 'cancel', 'retry'):
            sub.add_argument('job_id')
            if name == 'logs':
                sub.add_argument('--follow', '-f', action='store_true')
                sub.add_argument('--tail', type=int, default=50)
                sub.add_argument('--raw', action='store_true', help='Emit unsanitized task output (trusted logs only)')
        elif name == 'config':
            sub.add_argument('--set', action='append', default=[], metavar='KEY=JSON_VALUE')
    return ap


def display(pool, args):
    if args.watch is not None and (not math.isfinite(args.watch) or args.watch <= 0):
        raise ValueError('--watch must be finite and positive')
    if args.watch is not None and args.json:
        raise ValueError('--watch and --json cannot be combined')
    while True:
        data = snapshot(pool)
        if args.json:
            if args.status:
                data['jobs'] = [j for j in data['jobs'] if j['status'] in args.status]
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            if args.watch and sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            print(render(data, args.all, args.status), flush=True)
        if args.watch is None:
            return
        time.sleep(args.watch)


def logs(pool, args):
    from .view import safe
    if args.tail < 0:
        raise ValueError('--tail must be nonnegative')
    with transaction(pool) as db:
        job = get_job(db, args.job_id)
    if not job['log'] or not Path(job['log']).exists():
        print('No log yet; task has not started.')
        return
    with Path(job['log']).open(errors='replace') as stream:
        from collections import deque
        for line in deque(stream, maxlen=args.tail):
            print(line, end='') if args.raw else print(safe(line.rstrip('\n')))
        while args.follow:
            line = stream.readline()
            if line:
                print(line, end='', flush=True) if args.raw else print(safe(line.rstrip('\n')), flush=True)
                continue
            with transaction(pool) as db:
                if get_job(db, args.job_id)['status'] in TERMINAL:
                    break
            time.sleep(0.2)


def dispatch(args):
    pool = pool_path(args.pool)
    action = args.action
    if action == 'init':
        initialize(pool, args.simulate)
        print(f'Initialized {pool}')
    elif action == 'upgrade':
        upgrade(pool)
        print(f'Upgraded {pool}')
    elif action == 'submit':
        command = args.command
        if command[:1] == ['--']:
            command = command[1:]
        if args.keeper:
            from .keepers import PRESETS, build_parser as keeper_parser, __file__ as keepers_file_path
            keepers_file = lambda: keepers_file_path
            if args.keeper not in PRESETS:
                raise ValueError(f'Unknown keeper preset {args.keeper!r}; run: tetris keepers')
            keeper_parser().parse_args([args.keeper, *command])  # reject bad options before queuing
            with transaction(pool) as db:
                simulated = get_config(db)['simulate']
            interpreter = args.keeper_python or sys.executable
            if not simulated:
                # find_spec avoids importing torch (slow); catches the common "no torch here" mistake early.
                try:
                    probe = subprocess.run([interpreter, '-c', 'import importlib.util,sys;sys.exit(0 if importlib.util.find_spec("torch") else 3)'],
                                           capture_output=True, timeout=30)
                except OSError:
                    raise ValueError(f'Cannot run {interpreter}; pass a valid --keeper-python PATH') from None
                if probe.returncode != 0:
                    raise ValueError(f'PyTorch not found for {interpreter}; install it there or pass --keeper-python PATH')
            # keepers.py is stdlib-only and self-contained, so run it by path with any interpreter.
            command = [interpreter, keepers_file(), args.keeper, *command]
            args.filler = True
            args.name = args.name or f'keeper-{args.keeper}'
        elif args.per_gpu:
            raise ValueError('--per-gpu requires --keeper')
        env = {}
        for item in args.env:
            key, sep, value = item.partition('=')
            if not sep or not key or '\x00' in item:
                raise ValueError('--env requires KEY=VALUE without NUL characters')
            if key in ('CUDA_VISIBLE_DEVICES', 'TETRIS_JOB_ID', 'CUDA_DEVICE_ORDER',
                       'TETRIS_GPU_COUNT', 'TETRIS_ATTEMPT_TOKEN',
                       'UNIFIED_JOB_ID', 'UNIFIED_GPU_COUNT', 'UNIFIED_ATTEMPT_TOKEN'):
                raise ValueError(f'{key} is managed by the scheduler')
            env[key] = value
        targets = [[g] for g in args.gpus] if args.keeper and args.per_gpu and args.gpus else [args.gpus]
        for gpus in targets:
            name = f'{args.name}-gpu{gpus[0]}' if len(targets) > 1 else args.name
            print(submit(pool, command, name, args.cwd, env, args.depends, gpus, args.min_free_mb,
                         gpu_count=args.gpu_count, priority=args.priority, sharing=args.sharing, filler=args.filler))
    elif action == 'keepers':
        from .keepers import PRESETS
        for name, (summary, defaults) in PRESETS.items():
            opts = ' '.join(f'--{k.replace("_", "-")} {v}' for k, v in defaults.items())
            print(f'{name:10s} {summary}\n{"":10s} defaults: {opts}')
        print('\nExample: tetris submit --keeper floor --per-gpu --gpus 0,1,2,3 -- --util 40')
    elif action == 'start':
        print(f'Scheduler PID {start(pool)}')
    elif action == 'run':
        serve(pool)
    elif action == 'stop':
        stop(pool)
        print('Scheduler stopped. Running tasks are not stopped.')
    elif action == 'restart':
        # Hot update: pre-flight the freshly installed code in a new interpreter, so a
        # half-written install cannot take down a healthy scheduler.
        subprocess.run([sys.executable, '-m', 'gpu_tetris', '--version'], check=True,
                       capture_output=True, timeout=30)
        stop(pool)
        print(f'Scheduler PID {start(pool)} (running tasks were not touched)')
    elif action == 'status':
        display(pool, args)
    elif action == 'logs':
        logs(pool, args)
    elif action in ('cancel', 'retry'):
        with transaction(pool) as db:
            job = get_job(db, args.job_id)
            from .filler import mark_key
            db.execute('DELETE FROM runtime WHERE key=?', (mark_key(args.job_id),))
            if action == 'retry':
                if job['status'] not in ('failed', 'cancelled'):
                    raise ValueError('Only failed or cancelled tasks can be retried')
                require_schema(db)
                db.execute('UPDATE jobs SET status=?,cancel_requested=0,pid=NULL,identity=NULL,token=NULL,gpu=NULL,gpu_ids=NULL,error=NULL WHERE id=?',
                           ('queued', args.job_id))
            elif job['status'] not in TERMINAL:
                status = 'cancelled' if job['status'] in ('queued', 'blocked') else job['status']
                db.execute('UPDATE jobs SET cancel_requested=1,status=? WHERE id=?', (status, args.job_id))
        print(f'{action}: {args.job_id}')
    elif action in ('config', 'pause', 'resume'):
        with transaction(pool) as db:
            config = get_config(db)
            if action == 'config':
                for item in args.set:
                    key, sep, value = item.partition('=')
                    if not sep or key not in config:
                        raise ValueError('Unknown config key; use config to list keys')
                    if key == 'simulate':
                        raise ValueError('Simulation mode is fixed at init; use a separate pool')
                    config[key] = json.loads(value)
            else:
                config['launches_enabled'] = action == 'resume'
            validate_config(config)
            db.execute('UPDATE settings SET value=? WHERE key=?', (json.dumps(config), 'config'))
        print(json.dumps(config, indent=2))
    elif action == 'doctor':
        from .gpu import query
        with transaction(pool) as db:
            config = get_config(db)
        devices, _ = query(config['simulate'], config['simulate_gpu_count'])
        print(f'Python {sys.version.split()[0]} | pool writable: {os.access(pool, os.W_OK)} | GPUs: {len(devices)}')
        print('SIMULATION: CPU only' if config['simulate'] else 'NVIDIA query: OK')


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if sys.platform != 'linux':
            raise ValueError('Scheduling requires Linux (/proc and POSIX process groups)')
        dispatch(args)
        return 0
    except KeyboardInterrupt:
        return 0
    except BrokenPipeError:
        return 0
    except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


def smi_main():
    return main(['status', *sys.argv[1:]])
