import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from gpu_tetris.cli import main
from gpu_tetris.core import connect, get_job, initialize, submit, transaction
from gpu_tetris.gpu import policies
from gpu_tetris.scheduler import start, stop, tick
from gpu_tetris.view import render, snapshot


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pool = Path(self.tmp.name) / 'pool'
        initialize(self.pool, simulate=True)

    def tearDown(self):
        stop(self.pool)
        with transaction(self.pool) as db:
            db.execute("UPDATE jobs SET cancel_requested=1 WHERE status IN ('running','starting')")
        deadline = time.time() + 8
        while time.time() < deadline:
            if not any(j['status'] in ('running', 'starting') for j in snapshot(self.pool)['jobs']):
                break
            time.sleep(0.1)
        self.tmp.cleanup()

    def cli(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(['--pool', str(self.pool), *args])

    def fast(self):
        self.assertEqual(self.cli('config', '--set', 'observation_seconds=0', '--set', 'poll_seconds=0.05',
                                  '--set', 'launch_cooldown_seconds=0'), 0)

    def job(self, code='print("hello queue")', **kwargs):
        return submit(self.pool, [sys.executable, '-u', '-c', code], **kwargs)

    def wait(self, job_id, wanted, timeout=12):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with transaction(self.pool) as db:
                job = get_job(db, job_id)
            if job['status'] in wanted:
                return job
            time.sleep(0.05)
        self.fail(f'Timeout: {job}')

    def test_submit_persist_and_validate(self):
        key = self.job()
        self.assertEqual(snapshot(self.pool)['jobs'][0]['id'], key)
        with self.assertRaises(ValueError):
            submit(self.pool, [])
        with self.assertRaises(ValueError):
            self.job(depends=['missing'])
        self.assertEqual(self.cli('submit', '--env', 'CUDA_VISIBLE_DEVICES=1', '--', 'echo', 'x'), 1)

    def test_lifecycle_exit_code_logs_dependency(self):
        self.fast()
        first = self.job('import time; time.sleep(.2); print("first")')
        second = self.job('print("second")', depends=[first])
        failed = self.job('raise SystemExit(7)')
        start(self.pool)
        self.wait(second, {'complete'})
        self.assertEqual(self.wait(failed, {'failed'})['exit_code'], 7)
        job = self.wait(first, {'complete'})
        self.assertIn('first', Path(job['log']).read_text())
        self.assertEqual(self.cli('logs', first, '--tail', '1'), 0)

    def test_restart_preserves_running_and_does_not_duplicate(self):
        self.fast()
        key = self.job('import time; print("once"); time.sleep(1.5)')
        start(self.pool)
        before = self.wait(key, {'running'})
        stop(self.pool)
        start(self.pool)
        after = self.wait(key, {'complete'})
        self.assertEqual(before['pid'], after['pid'])
        self.assertEqual(after['attempt'], 1)
        self.assertEqual(Path(after['log']).read_text().count('once'), 1)

    def test_cancel_running_and_retry(self):
        self.fast()
        key = self.job('import time; time.sleep(1)')
        start(self.pool)
        self.wait(key, {'running'})
        self.assertEqual(self.cli('cancel', key), 0)
        self.wait(key, {'cancelled'})
        self.assertEqual(self.cli('retry', key), 0)
        self.assertEqual(self.wait(key, {'complete'})['attempt'], 2)

    def test_cancel_queued_and_pause(self):
        self.fast()
        key = self.job()
        self.assertEqual(self.cli('pause'), 0)
        tick(self.pool, {}, {})
        self.assertEqual(snapshot(self.pool)['jobs'][0]['status'], 'queued')
        self.assertEqual(self.cli('cancel', key), 0)
        self.assertEqual(snapshot(self.pool)['jobs'][0]['status'], 'cancelled')

    def test_singleton_scheduler(self):
        self.fast()
        pid = start(self.pool)
        self.assertEqual(start(self.pool), pid)
        result = subprocess.run([sys.executable, '-m', 'gpu_tetris', '--pool', str(self.pool), 'run'],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 1)
        self.assertIn('already owns', result.stderr)

    def test_readonly_status_and_env_redaction(self):
        self.job(env={'SECRET': 'not-for-json'})
        db = connect(self.pool)
        before = db.execute('SELECT * FROM jobs').fetchall()
        data = snapshot(self.pool)
        self.assertNotIn('not-for-json', json.dumps(data))
        self.assertEqual(before, db.execute('SELECT * FROM jobs').fetchall())
        db.close()
        self.assertIn('STALE', render(data))
        self.assertIn('SIMULATION', render(data))

    def test_options_errors(self):
        for value in ('0', '-1', 'nan', 'inf'):
            self.assertEqual(self.cli('status', '--watch', value), 1)
        self.assertEqual(self.cli('status', '--watch', '--json'), 1)
        self.assertEqual(self.cli('config', '--set', 'per_gpu_limit=0'), 1)
        self.assertEqual(self.cli('config', '--set', 'simulate=false'), 1)
        self.assertEqual(self.cli('logs', 'missing'), 1)

    def test_gpu_failure_never_launches(self):
        self.job()
        history = {'0': [[1, 0]]}
        with patch('gpu_tetris.scheduler.query', side_effect=OSError('GPU unavailable')):
            tick(self.pool, history, {})
        self.assertEqual(history, {})
        self.assertEqual(snapshot(self.pool)['jobs'][0]['attempt'], 0)
        self.assertIn('unavailable', snapshot(self.pool)['error'])

    def test_worker_loss_can_cancel_and_retry(self):
        import signal
        self.fast()
        key = self.job('import time; print("ready"); time.sleep(10)')
        start(self.pool)
        job = self.wait(key, {'running'})
        os.kill(job['pid'], signal.SIGKILL)
        time.sleep(0.15)
        self.assertEqual(self.cli('cancel', key), 0)
        self.wait(key, {'cancelled'})
        from gpu_tetris.process import members
        self.assertEqual(members(job['token']), {})

    def test_detached_child_cancelled(self):
        self.fast()
        key = self.job('import os,time; os.setsid(); print("detached"); time.sleep(10)')
        start(self.pool)
        job = self.wait(key, {'running'})
        time.sleep(0.2)
        self.assertEqual(self.cli('cancel', key), 0)
        self.wait(key, {'cancelled'})
        from gpu_tetris.process import members
        self.assertEqual(members(job['token']), {})

    def test_pause_during_probe_prevents_launch(self):
        self.fast()
        self.job()
        def probe(_simulate, _count=1, **kwargs):
            self.assertEqual(self.cli('pause'), 0)
            return [dict(index=0, uuid='sim', free_mb=999999, util=0)], []
        with patch('gpu_tetris.scheduler.query', side_effect=probe):
            tick(self.pool, {}, {})
        self.assertEqual(snapshot(self.pool)['jobs'][0]['attempt'], 0)

    def test_ownership_change_during_probe_requires_fresh_sample(self):
        self.fast()
        key = self.job()
        def probe(_simulate, _count=1, **kwargs):
            self.assertFalse(kwargs['query_apps'])
            self.assertEqual(self.cli('config', '--set', 'ownership_mode="strict"'), 0)
            return [dict(index=0, uuid='GPU-test', free_mb=30000, util=0)], []
        with patch('gpu_tetris.scheduler.query', side_effect=probe):
            tick(self.pool, {}, {})
        data = snapshot(self.pool)
        self.assertEqual(next(j for j in data['jobs'] if j['id'] == key)['attempt'], 0)
        self.assertIn('fresh sample', data['error'])

    def test_cooldown_survives_restart(self):
        self.fast()
        self.assertEqual(self.cli('config', '--set', 'launch_cooldown_seconds=60'), 0)
        first = self.job()
        start(self.pool)
        self.wait(first, {'complete'})
        stop(self.pool)
        second = self.job()
        start(self.pool)
        time.sleep(0.2)
        with transaction(self.pool) as db:
            self.assertEqual(get_job(db, second)['attempt'], 0)

    def test_concurrent_submissions(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as executor:
            keys = list(executor.map(lambda _: self.job(), range(24)))
        self.assertEqual(len(set(keys)), 24)
        self.assertEqual(len(snapshot(self.pool)['jobs']), 24)

    def test_running_count_never_exceeds_cap(self):
        self.fast()
        keys = [self.job('import time; time.sleep(.5)') for _ in range(5)]
        start(self.pool)
        deadline = time.time() + 8
        peak = 0
        while time.time() < deadline:
            data = snapshot(self.pool)
            active = sum(j['status'] in ('running', 'starting') for j in data['jobs'])
            peak = max(peak, active)
            self.assertLessEqual(active, 2)
            if data['counts'].get('complete') == len(keys):
                break
            time.sleep(.02)
        self.assertEqual(snapshot(self.pool)['counts'].get('complete'), 5)
        self.assertEqual(peak, 2)

    def test_gate_eof_does_not_execute(self):
        marker = Path(self.tmp.name) / 'executed'
        read_fd, write_fd = os.pipe()
        child = subprocess.Popen([sys.executable, '-m', 'gpu_tetris.process', str(read_fd),
                                  sys.executable, '-c', f'open({str(marker)!r}, "w").close()'], pass_fds=(read_fd,))
        os.close(read_fd)
        os.close(write_fd)
        self.assertEqual(child.wait(timeout=5), 125)
        self.assertFalse(marker.exists())

    def test_blocked_dependency_does_not_stall_other_jobs(self):
        self.fast()
        first = self.job('raise SystemExit(1)')
        second = self.job(depends=[first])
        third = self.job()
        start(self.pool)
        self.wait(first, {'failed'})
        self.wait(second, {'blocked'})
        self.wait(third, {'complete'})

    def test_unknown_interrupted_attempt_not_retried(self):
        key = self.job()
        with transaction(self.pool) as db:
            db.execute('UPDATE jobs SET status=?,token=?,attempt=1 WHERE id=?', ('starting', 'lost', key))
        tick(self.pool, {}, {})
        job = snapshot(self.pool)['jobs'][0]
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['attempt'], 1)

    def test_legacy_directory_rejected(self):
        old = Path(self.tmp.name) / 'legacy'
        old.mkdir()
        (old / 'state.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Legacy'):
            initialize(old)
        self.assertFalse((old / 'queue.db').exists())


class PolicyTests(unittest.TestCase):
    def setUp(self):
        from gpu_tetris.core import DEFAULTS
        self.config = dict(DEFAULTS)
        self.gpu = dict(index=0, uuid='GPU-test', free_mb=30000, util=10)
        self.history = {}

    def policy(self, now, active=None, apps=None, last=None):
        return policies([self.gpu], apps or [], active or [], self.history, last or {}, self.config, now)[0]['policy']

    def test_full_continuous_window(self):
        for now in range(0, 60, 10):
            self.assertEqual(self.policy(now), 'OBSERVE / COOLDOWN')
        self.assertEqual(self.policy(60), 'READY')

    def test_threshold_boundary_and_gap_reset(self):
        self.gpu['util'] = 65
        self.assertEqual(self.policy(0), 'HOLD >=65%')
        self.gpu['util'] = 10
        self.assertEqual(self.policy(100), 'OBSERVE / COOLDOWN')

    def test_third_task_blocked(self):
        active = [dict(pid=123, gpu=0), dict(pid=124, gpu=0)]
        self.assertEqual(self.policy(0, active), 'FULL')

    def test_unknown_compute_process_blocked(self):
        self.config['ownership_mode'] = 'strict'
        self.assertEqual(self.policy(0, apps=[(999999999, 'GPU-test')]), 'UNKNOWN OWNER')

    def test_single_user_external_process_keeps_resource_limits(self):
        self.config.update(observation_seconds=0, launch_cooldown_seconds=0)
        apps = [(999999999, 'GPU-test')]
        self.assertEqual(self.policy(0, apps=apps), 'READY')
        self.gpu['util'] = 90
        self.assertEqual(self.policy(10, apps=apps), 'HOLD >=65%')
        self.gpu.update(util=0, free_mb=1)
        self.assertEqual(self.policy(20, apps=apps), 'LOW MEMORY')
        self.gpu['free_mb'] = 30000
        self.assertEqual(self.policy(30, [dict(pid=123, gpu=0), dict(pid=124, gpu=0)], apps), 'FULL')

    def test_strict_recognizes_managed_process_group(self):
        self.config.update(ownership_mode='strict', observation_seconds=0, launch_cooldown_seconds=0)
        with patch('gpu_tetris.gpu.process_group', return_value=123):
            self.assertEqual(self.policy(0, [dict(pid=123, gpu=0)], [(456, 'GPU-test')]), 'READY')

    def test_respects_configured_threshold_and_limit(self):
        self.config.update(per_gpu_limit=1, utilization_threshold=50)
        self.gpu['util'] = 50
        self.assertEqual(self.policy(0), 'HOLD >=50%')
        self.assertEqual(self.policy(10, [dict(pid=123, gpu=0)]), 'FULL')

    def test_memory_and_cooldown(self):
        self.config['observation_seconds'] = 0
        self.gpu['free_mb'] = 100
        self.assertEqual(self.policy(0), 'LOW MEMORY')
        self.gpu['free_mb'] = 30000
        self.assertEqual(self.policy(10, last={'0': 0}), 'OBSERVE / COOLDOWN')


if __name__ == '__main__':
    unittest.main()
