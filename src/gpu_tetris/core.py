"""Transactional queue storage and Linux process identity helpers."""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid

DEFAULTS = {
    'per_gpu_limit': 2, 'utilization_threshold': 65,
    'observation_seconds': 60, 'poll_seconds': 10,
    'launch_cooldown_seconds': 60, 'min_free_mb': 16000,
    'allowed_gpus': None, 'launches_enabled': True, 'simulate': False,
    'queue_policy': 'fifo', 'placement_policy': 'balanced',
    'priority_aging_seconds': 300, 'simulate_gpu_count': 1,
    'filler_enabled': False, 'filler_utilization_threshold': 30,
    'filler_rearm_seconds': 1800,
}
QUEUE_POLICIES = ('fifo', 'strict_fifo', 'priority', 'small_first', 'large_first')
PLACEMENT_POLICIES = ('balanced', 'pack', 'spread', 'most_free')
DB_VERSION = 2
TERMINAL = {'complete', 'failed', 'cancelled'}


def pool_path(value=None):
    if value:
        return Path(value).expanduser().resolve()
    # TETRIS_POOL wins; UNIFIED_GPU_POOL is the pre-rename name, still honoured.
    for var in ('TETRIS_POOL', 'UNIFIED_GPU_POOL'):
        if os.environ.get(var):
            return Path(os.environ[var]).expanduser().resolve()
    root = Path(os.environ.get('XDG_STATE_HOME', str(Path.home() / '.local/state')))
    path = (root / 'gpu-tetris').resolve()
    legacy = (root / 'unified-gpu').resolve()
    # Keep using an existing pre-rename pool instead of silently starting an empty one.
    return legacy if not path.exists() and (legacy / 'queue.db').is_file() else path


def connect(pool):
    if not (pool / 'queue.db').is_file():
        raise ValueError('Pool not initialized. Run: tetris init')
    db = sqlite3.connect(f'{(pool / "queue.db").as_uri()}?mode=rw', uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    return db


@contextmanager
def transaction(pool):
    db = connect(pool)
    try:
        db.execute('BEGIN IMMEDIATE')
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def validate_config(config):
    for key in ('observation_seconds', 'launch_cooldown_seconds', 'min_free_mb', 'filler_rearm_seconds'):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be a finite nonnegative number')
    for key in ('per_gpu_limit', 'poll_seconds', 'utilization_threshold', 'priority_aging_seconds', 'simulate_gpu_count'):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be a finite positive number')
    if int(config['per_gpu_limit']) != config['per_gpu_limit']:
        raise ValueError('per_gpu_limit must be an integer')
    if config['utilization_threshold'] > 100:
        raise ValueError('utilization_threshold must not exceed 100')
    value = config['filler_utilization_threshold']
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 100:
        raise ValueError('filler_utilization_threshold must be in (0, 100]')
    for key in ('simulate', 'launches_enabled', 'filler_enabled'):
        if not isinstance(config[key], bool):
            raise ValueError(f'{key} must be a boolean')
    allowed = config['allowed_gpus']
    if allowed is not None and (not isinstance(allowed, list) or any(type(i) is not int or i < 0 for i in allowed)):
        raise ValueError('allowed_gpus must be null or an array of nonnegative integers')
    if type(config['simulate_gpu_count']) is not int or config['simulate_gpu_count'] < 1:
        raise ValueError('simulate_gpu_count must be a positive integer')
    if config['queue_policy'] not in QUEUE_POLICIES:
        raise ValueError(f'queue_policy must be one of {QUEUE_POLICIES}')
    if config['placement_policy'] not in PLACEMENT_POLICIES:
        raise ValueError(f'placement_policy must be one of {PLACEMENT_POLICIES}')
    return config


def allocated_gpus(job):
    value = job.get('gpu_ids')
    if value is not None:
        return json.loads(value) if isinstance(value, str) else value
    return [job['gpu']] if job.get('gpu') is not None else []


def exclusive(spec):
    return spec.get('sharing', 'exclusive' if spec.get('gpu_count', 1) > 1 else 'shared') == 'exclusive'


def require_schema(db):
    if db.execute('PRAGMA user_version').fetchone()[0] != DB_VERSION:
        raise ValueError('Pool schema requires upgrade. Stop scheduler, drain tasks, then run: tetris upgrade')


def migrate(db):
    version = db.execute('PRAGMA user_version').fetchone()[0]
    if version > DB_VERSION:
        raise ValueError('Pool was created by a newer version')
    if version == DB_VERSION:
        return
    daemon = db.execute('SELECT value FROM runtime WHERE key=?', ('daemon',)).fetchone()
    info = json.loads(daemon['value']) if daemon else {}
    if alive(info.get('pid'), info.get('identity')) or db.execute(
            "SELECT 1 FROM jobs WHERE status IN ('starting','running')").fetchone():
        raise ValueError('Stop scheduler and finish or cancel active tasks before upgrading')
    if 'gpu_ids' not in {row[1] for row in db.execute('PRAGMA table_info(jobs)')}:
        db.execute('ALTER TABLE jobs ADD COLUMN gpu_ids TEXT')
    for row in db.execute('SELECT id,gpu FROM jobs WHERE gpu IS NOT NULL AND gpu_ids IS NULL').fetchall():
        db.execute('UPDATE jobs SET gpu_ids=? WHERE id=?', (json.dumps([row['gpu']]), row['id']))
    db.execute(f'PRAGMA user_version={DB_VERSION}')


def upgrade(pool):
    import fcntl
    with (pool / 'scheduler.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Stop scheduler before upgrading') from None
        with transaction(pool) as db:
            if db.execute('PRAGMA user_version').fetchone()[0] < DB_VERSION:
                from .scheduler import reconcile, runtime
                daemon = runtime(db, 'daemon', {})
                if alive(daemon.get('pid'), daemon.get('identity')):
                    raise ValueError('Stop scheduler before upgrading')
                reconcile(pool, db)
            migrate(db)


def initialize(pool, simulate=False):
    pool.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (pool / 'state.json').exists() or (pool / 'sources').exists():
        raise ValueError('Legacy scheduler pool detected; use a separate directory')
    existing = (pool / 'queue.db').exists()
    db = sqlite3.connect(pool / 'queue.db', timeout=30)
    db.row_factory = sqlite3.Row
    try:
        if existing:
            require_schema(db)
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript('''
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, spec TEXT NOT NULL,
                status TEXT NOT NULL, submitted REAL NOT NULL, gpu INTEGER,
                pid INTEGER, identity TEXT, attempt INTEGER NOT NULL DEFAULT 0,
                token TEXT, log TEXT, exit_code INTEGER, error TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS runtime (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        ''')
        config = dict(DEFAULTS, simulate=simulate)
        db.execute('INSERT OR IGNORE INTO settings VALUES (?, ?)', ('config', json.dumps(config)))
        migrate(db)
        db.commit()
    finally:
        db.close()
    (pool / 'logs').mkdir(exist_ok=True, mode=0o700)


def get_config(db):
    row = db.execute('SELECT value FROM settings WHERE key=?', ('config',)).fetchone()
    return validate_config(dict(DEFAULTS, **json.loads(row['value'])))


def identity(pid):
    try:
        stat = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(') ', 1)[1].split()
        if stat[0] == 'Z':
            return None
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        return f'{boot}:{stat[19]}'
    except (OSError, ValueError, IndexError):
        return None


def alive(pid, token):
    return bool(pid and token and identity(pid) == token)


def atomic_json(path, value):
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value))
    temp.replace(path)


def submit(pool, command, name=None, cwd=None, env=None, depends=None, gpus=None, min_free_mb=0,
           *, gpu_count=1, priority=0, sharing=None, filler=False):
    if not command or any(not isinstance(arg, str) or '\x00' in arg for arg in command):
        raise ValueError('Command must be a nonempty argv array without NUL characters')
    cwd = str(Path(cwd or Path.cwd()).expanduser().resolve())
    if not Path(cwd).is_dir():
        raise ValueError(f'Working directory not found: {cwd}')
    if type(min_free_mb) is not int or min_free_mb < 0:
        raise ValueError('min_free_mb must be a nonnegative integer (per GPU)')
    if type(gpu_count) is not int or gpu_count < 1:
        raise ValueError('gpu_count must be a positive integer')
    if type(priority) is not int:
        raise ValueError('priority must be an integer')
    if sharing is None:
        sharing = 'exclusive' if gpu_count > 1 else 'shared'
    if sharing not in ('exclusive', 'shared'):
        raise ValueError('sharing must be exclusive or shared')
    if gpus is not None:
        if not isinstance(gpus, list) or any(type(i) is not int or i < 0 for i in gpus):
            raise ValueError('gpus must be an array of nonnegative integers')
        gpus = sorted(set(gpus))
        if len(gpus) < gpu_count:
            raise ValueError('GPU allowlist is smaller than gpu_count')
    if type(filler) is not bool:
        raise ValueError('filler must be a boolean')
    if filler and (gpu_count != 1 or sharing != 'shared' or priority != 0):
        raise ValueError('Filler requires one shared GPU and fixed lowest priority (omit --priority)')
    depends = depends or []
    spec = dict(command=command, cwd=cwd, env=env or {}, depends=depends,
                allowed_gpus=gpus, min_free_mb=min_free_mb, gpu_count=gpu_count,
                priority=None if filler else priority, sharing=sharing, filler=filler)
    job_id = uuid.uuid4().hex[:12]
    with transaction(pool) as db:
        require_schema(db)
        for dependency in depends:
            parent = db.execute('SELECT spec FROM jobs WHERE id=?', (dependency,)).fetchone()
            if parent is None:
                raise ValueError(f'Unknown dependency: {dependency}')
            if json.loads(parent['spec']).get('filler'):
                raise ValueError('Filler tasks cannot be dependencies')
        db.execute('INSERT INTO jobs (id,name,spec,status,submitted) VALUES (?,?,?,?,?)',
                   (job_id, name or Path(command[0]).name, json.dumps(spec), 'queued', time.time()))
    return job_id


def get_job(db, job_id):
    row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
    if row is None:
        raise ValueError(f'Unknown job: {job_id}')
    return dict(row)
