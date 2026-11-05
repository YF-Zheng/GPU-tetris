import json
import os
import signal
import time
import unittest
from unittest.mock import patch

import test_queue as support
from gpu_tetris.core import DEFAULTS, QUEUE_POLICIES, get_job, transaction, validate_config
from gpu_tetris.filler import mark_key, recover
from gpu_tetris.scheduler import tick
from gpu_tetris.strategies import order_jobs
from gpu_tetris.view import snapshot


class FillerTests(unittest.TestCase):
    setUp = support.QueueTests.setUp
    tearDown = support.QueueTests.tearDown
    cli = support.QueueTests.cli
    fast = support.QueueTests.fast
    job = support.QueueTests.job
    wait = support.QueueTests.wait

    def enable(self):
        self.fast()
        self.assertEqual(self.cli('config', '--set', 'filler_enabled=true', '--set', 'filler_rearm_seconds=0'), 0)

    def sample(self, util=0):
        return ([dict(index=0, uuid='SIMULATED-CPU-0', free_mb=100000, util=util)], [])

    def state(self, key):
        with transaction(self.pool) as db:
            return get_job(db, key)

    def start_filler(self):
        self.enable()
        key = self.job('import time; time.sleep(60)', filler=True)
        tick(self.pool, {}, {})
        return key, self.wait(key, {'running'})

    def test_config_and_submission_validation(self):
        for value in (0, 101, True, float('nan'), '30'):
            with self.assertRaises(ValueError):
                validate_config(dict(DEFAULTS, filler_utilization_threshold=value))
        for kwargs in (dict(priority=1), dict(gpu_count=2), dict(sharing='exclusive')):
            with self.assertRaises(ValueError):
                self.job(filler=True, **kwargs)
        self.assertEqual(self.cli('submit', '--filler', '--', 'echo', 'ok'), 0)
        data = snapshot(self.pool)['jobs'][0]
        self.assertTrue(data['filler'])
        self.assertIsNone(data['priority'])
        self.assertEqual(data['priority_tier'], 'lowest')

    def test_fixed_lowest_for_every_queue_policy(self):
        jobs = [dict(id='filler', submitted=0, spec=json.dumps(dict(filler=True, priority=None))),
                dict(id='normal', submitted=10000, spec=json.dumps(dict(priority=-10**30)))]
        for policy in QUEUE_POLICIES:
            self.assertEqual([j['id'] for j in order_jobs(jobs, dict(DEFAULTS, queue_policy=policy), 10000)],
                             ['normal', 'filler'])

    def test_disabled_by_default(self):
        self.fast()
        key = self.job(filler=True)
        tick(self.pool, {}, {})
        self.assertEqual(self.state(key)['attempt'], 0)

    def test_threshold_is_strict_and_configurable(self):
        self.enable()
        key = self.job(filler=True)
        tick(self.pool, {}, {}, sample=self.sample(30))
        self.assertEqual(self.state(key)['attempt'], 0)
        tick(self.pool, {}, {}, sample=self.sample(29))
        self.wait(key, {'complete'})

    def test_observation_window_and_gap(self):
        self.enable()
        self.cli('config', '--set', 'observation_seconds=10', '--set', 'poll_seconds=10')
        key = self.job(filler=True)
        history, launches = {}, {}
        tick(self.pool, history, launches, sample=self.sample(10), now=100)
        tick(self.pool, history, launches, sample=self.sample(35), now=110)
        tick(self.pool, history, launches, sample=self.sample(10), now=120)
        self.assertEqual(self.state(key)['attempt'], 0)
        tick(self.pool, history, launches, sample=self.sample(10), now=130)
        self.wait(key, {'complete'})

    def test_normal_launch_requires_fresh_samples_before_fill(self):
        self.enable()
        filler = self.job(filler=True)
        normal = self.job('import time; time.sleep(60)')
        tick(self.pool, {}, {})
        self.wait(normal, {'running'})
        self.assertEqual(self.state(filler)['attempt'], 0)
        tick(self.pool, {}, {})
        self.wait(filler, {'complete'})

    def test_one_filler_per_gpu(self):
        first, _ = self.start_filler()
        second = self.job(filler=True)
        tick(self.pool, {}, {})
        self.assertEqual(self.state(second)['attempt'], 0)
        self.assertEqual(self.state(first)['status'], 'running')

    def test_high_util_filler_yields_and_requeues(self):
        filler, old = self.start_filler()
        normal = self.job('import time; time.sleep(.5)', priority=-10**20)
        tick(self.pool, {}, {}, sample=self.sample(99))
        self.assertEqual(self.state(normal)['attempt'], 0)
        self.assertTrue(self.state(filler)['cancel_requested'])
        self.wait(filler, {'cancelled'})
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            tick(self.pool, {}, {})
            if self.state(normal)['status'] == 'complete':
                break
            time.sleep(.1)
        self.assertEqual(self.state(normal)['status'], 'complete')
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and self.state(filler)['attempt'] == old['attempt']:
            tick(self.pool, {}, {})
            time.sleep(.05)
        self.assertEqual(self.state(filler)['attempt'], old['attempt'] + 1)
        self.wait(filler, {'running'})

    def test_manual_cancel_during_reclaim_does_not_revive(self):
        filler, _ = self.start_filler()
        self.job()
        tick(self.pool, {}, {}, sample=self.sample(99))
        self.assertEqual(self.cli('cancel', filler), 0)
        self.wait(filler, {'cancelled'})
        tick(self.pool, {}, {})
        self.assertEqual(self.state(filler)['status'], 'cancelled')
        with transaction(self.pool) as db:
            self.assertIsNone(db.execute('SELECT value FROM runtime WHERE key=?', (mark_key(filler),)).fetchone())

    def test_disable_reclaims_but_pause_preserves(self):
        filler, _ = self.start_filler()
        self.cli('pause')
        tick(self.pool, {}, {})
        self.assertFalse(self.state(filler)['cancel_requested'])
        self.cli('config', '--set', 'filler_enabled=false')
        tick(self.pool, {}, {})
        self.assertTrue(self.state(filler)['cancel_requested'])
        self.wait(filler, {'cancelled'})
        for _ in range(8):
            tick(self.pool, {}, {})
            time.sleep(.05)
        self.assertEqual(self.state(filler)['status'], 'queued')
        self.assertEqual(self.state(filler)['attempt'], 1)

    def test_unknown_owner_still_blocks(self):
        self.enable()
        key = self.job(filler=True)
        cards, _ = self.sample()
        with patch('gpu_tetris.gpu.process_group', return_value=-1):
            tick(self.pool, {}, {}, sample=(cards, [(123, cards[0]['uuid'])]))
        self.assertEqual(self.state(key)['attempt'], 0)

    def test_recovery_retains_residual_process_slot(self):
        self.enable()
        key = self.job(filler=True)
        with transaction(self.pool) as db:
            db.execute('UPDATE jobs SET status=?,token=?,gpu_ids=? WHERE id=?',
                       ('cancelled', 'test-token', '[0]', key))
            db.execute('INSERT INTO runtime VALUES (?,?)',
                       (mark_key(key), json.dumps(dict(token='test-token'))))
            with patch('gpu_tetris.filler.members', return_value={123: 'stamp'}):
                self.assertEqual(recover(db, {}, {}, 100), {0})
            self.assertEqual(get_job(db, key)['status'], 'cancelled')
        tick(self.pool, {}, {})
        self.assertEqual(self.state(key)['status'], 'queued')

    def test_lost_supervisor_reclaim_survives_restart(self):
        filler, old = self.start_filler()
        os.kill(old['pid'], signal.SIGKILL)
        time.sleep(.1)
        normal = self.job()
        tick(self.pool, {}, {}, sample=self.sample(99))
        tick(self.pool, {}, {}, sample=self.sample(99))
        self.assertEqual(self.state(normal)['attempt'], 0)
        for _ in range(8):
            tick(self.pool, {}, {})
            if self.state(normal)['attempt']:
                break
            time.sleep(.1)
        self.wait(normal, {'complete'})
        self.assertEqual(self.state(filler)['attempt'], 1)

    def test_independent_threshold_above_normal_threshold(self):
        self.enable()
        self.cli('config', '--set', 'utilization_threshold=20', '--set', 'filler_utilization_threshold=30')
        key = self.job(filler=True)
        tick(self.pool, {}, {}, sample=self.sample(25))
        self.wait(key, {'complete'})

    def test_rearm_hysteresis_after_reclaim(self):
        from gpu_tetris.filler import admission_views, rearm_times
        self.enable()
        filler, _ = self.start_filler()
        self.job()
        tick(self.pool, {}, {}, sample=self.sample(99))
        self.wait(filler, {'cancelled'})
        tick(self.pool, {}, {})
        with transaction(self.pool) as db:
            stamp = rearm_times(db)['0']
        config = dict(DEFAULTS, filler_enabled=True, observation_seconds=0, launch_cooldown_seconds=0,
                      filler_rearm_seconds=1800)
        views = [dict(index=0, uuid='U', count=0, util=0, free_mb=10**6, policy='READY')]
        history = {'0': [[stamp, 0]]}
        # Normal 60 s observation must not re-admit a filler; only the 1800 s re-arm window does.
        for delay, wanted in ((60, 'FILLER WAIT'), (1799, 'FILLER WAIT'), (1800, 'READY')):
            got = admission_views(views, [], history, {}, config, stamp + delay, set(), {'0': stamp})
            self.assertEqual(got[0]['policy'], wanted, delay)

    def test_filler_cannot_be_dependency(self):
        filler = self.job(filler=True)
        with self.assertRaisesRegex(ValueError, 'cannot be dependencies'):
            self.job(depends=[filler])

    def test_free_card_does_not_reclaim_unrelated_filler(self):
        self.enable()
        self.cli('config', '--set', 'simulate_gpu_count=2')
        filler = self.job('import time; time.sleep(60)', filler=True, gpus=[1])
        tick(self.pool, {}, {})
        self.wait(filler, {'running'})
        normal = self.job()
        tick(self.pool, {}, {})
        self.wait(normal, {'complete'})
        self.assertFalse(self.state(filler)['cancel_requested'])
        self.assertEqual(self.state(normal)['gpu_ids'], '[0]')

    def test_new_normal_submission_between_claims_blocks_filler(self):
        from gpu_tetris.process import detached
        self.enable()
        self.cli('config', '--set', 'simulate_gpu_count=2')
        first = self.job('import time; time.sleep(60)', filler=True, gpus=[0])
        second = self.job(filler=True, gpus=[1])
        submitted = []
        def launch(*args, **kwargs):
            submitted.append(self.job(gpus=[1]))
            return detached(*args, **kwargs)
        with patch('gpu_tetris.scheduler.detached', side_effect=launch):
            tick(self.pool, {}, {})
        self.wait(first, {'running'})
        self.assertEqual(self.state(second)['attempt'], 0)
        self.assertEqual(len(submitted), 1)
        tick(self.pool, {}, {})
        self.wait(submitted[0], {'complete'})

    def test_multigpu_normal_reclaims_complete_filler_group(self):
        self.enable()
        self.cli('config', '--set', 'simulate_gpu_count=2')
        fillers = [self.job('import time; time.sleep(60)', filler=True, gpus=[i]) for i in range(2)]
        tick(self.pool, {}, {})
        for key in fillers:
            self.wait(key, {'running'})
        normal = self.job(gpu_count=2)
        tick(self.pool, {}, {})
        self.assertEqual(self.state(normal)['attempt'], 0)
        for key in fillers:
            self.assertTrue(self.state(key)['cancel_requested'])
            self.wait(key, {'cancelled'})
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            tick(self.pool, {}, {})
            if self.state(normal)['attempt']:
                break
            time.sleep(.05)
        self.wait(normal, {'complete'})
        self.assertEqual(json.loads(self.state(normal)['gpu_ids']), [0, 1])

    def test_cancel_during_worker_final_cleanup(self):
        from gpu_tetris.process import cleanup
        from gpu_tetris.worker import run
        self.enable()
        key = self.job(filler=True)
        with patch('gpu_tetris.scheduler.detached', side_effect=OSError('test claim')):
            tick(self.pool, {}, {})
        job = self.state(key)
        with transaction(self.pool) as db:
            db.execute('UPDATE jobs SET status=? WHERE id=?', ('starting', key))
        def cancel_then_clean(*args, **kwargs):
            self.assertEqual(self.cli('cancel', key), 0)
            return cleanup(*args, **kwargs)
        with patch('gpu_tetris.worker.signal.signal'), patch('gpu_tetris.worker.cleanup', side_effect=cancel_then_clean):
            run(self.pool, key, job['token'])
        self.assertEqual(self.state(key)['status'], 'cancelled')
        receipt = self.pool / 'logs' / f'{key}.{job["token"]}.result.json'
        self.assertEqual(json.loads(receipt.read_text())['status'], 'cancelled')

    def test_natural_exit_and_failure_not_restarted(self):
        self.enable()
        for code, wanted in [('pass', 'complete'), ('raise SystemExit(2)', 'failed')]:
            key = self.job(code, filler=True)
            tick(self.pool, {}, {})
            self.wait(key, {wanted})
            tick(self.pool, {}, {})
            self.assertEqual(self.state(key)['attempt'], 1)
            self.assertEqual(self.state(key)['status'], wanted)


if __name__ == '__main__':
    unittest.main()
