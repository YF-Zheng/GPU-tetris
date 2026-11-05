import json
import os
import signal
import sys
import time
import unittest
from unittest.mock import patch

import test_queue as support
from gpu_tetris.core import DEFAULTS, allocated_gpus, get_job, identity, submit, transaction, upgrade
from gpu_tetris.gpu import policies
from gpu_tetris.scheduler import start, stop, tick
from gpu_tetris.strategies import order_jobs, select_gpus
from gpu_tetris.view import render, snapshot


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(DEFAULTS)
        self.spec = dict(gpu_count=2, allowed_gpus=None, min_free_mb=16000, sharing='shared')
        self.views = [dict(index=0, uuid='A', count=1, util=10, free_mb=20000, policy='READY'),
                      dict(index=1, uuid='B', count=0, util=20, free_mb=40000, policy='READY'),
                      dict(index=2, uuid='C', count=0, util=5, free_mb=30000, policy='READY')]

    def selected(self, policy, count=1):
        self.config['placement_policy'] = policy
        self.spec['gpu_count'] = count
        return [g['index'] for g in select_gpus(self.views, self.spec, self.config)]

    def test_all_placement_policies(self):
        for policy, expected in [('balanced', [2]), ('pack', [0]), ('spread', [1]), ('most_free', [1])]:
            with self.subTest(policy=policy):
                self.assertEqual(self.selected(policy), expected)
        self.assertEqual(self.selected('most_free', 2), [1, 2])

    def test_diagnose_reports_each_gpu(self):
        from gpu_tetris.strategies import diagnose
        views = [dict(index=0, count=2, util=10, free_mb=40000, policy='FULL'),
                 dict(index=1, count=0, util=10, free_mb=100, policy='READY'),
                 dict(index=2, count=1, util=10, free_mb=40000, policy='READY'),
                 dict(index=3, count=0, util=10, free_mb=40000, policy='READY'),
                 dict(index=4, count=0, util=10, free_mb=40000, policy='READY')]
        spec = dict(gpu_count=2, allowed_gpus=[0, 1, 2, 3], min_free_mb=24000, sharing='exclusive')
        text = diagnose(views, spec)
        for part in ('GPU0: FULL', 'GPU1: LOW MEMORY 100<24000 MiB', 'GPU2: NOT EMPTY',
                     'GPU3: READY', 'GPU4: NOT IN --gpus'):
            self.assertIn(part, text)

    def test_insufficient_is_all_or_nothing(self):
        self.spec['gpu_count'] = 4
        self.assertEqual(select_gpus(self.views, self.spec, self.config), [])

    def test_exclusive_needs_empty_cards(self):
        self.spec.update(gpu_count=3, sharing='exclusive')
        self.assertEqual(select_gpus(self.views, self.spec, self.config), [])
        self.spec['gpu_count'] = 2
        self.assertEqual([g['index'] for g in select_gpus(self.views, self.spec, self.config)], [1, 2])

    def test_every_card_must_pass_memory_allowlist_and_safety(self):
        for changes in [dict(min_free_mb=35000), dict(allowed_gpus=[0]), dict(gpu_count=3)]:
            with self.subTest(changes=changes):
                spec = dict(self.spec, **changes)
                self.views[0]['policy'] = 'UNKNOWN OWNER'
                self.assertEqual(select_gpus(self.views, spec, self.config), [])

    def test_queue_ordering_and_aging(self):
        jobs = [dict(id='old', submitted=0, spec=json.dumps(dict(gpu_count=4, priority=0))),
                dict(id='new', submitted=1000, spec=json.dumps(dict(gpu_count=1, priority=2)))]
        for policy, expected in [('fifo', ['old', 'new']), ('strict_fifo', ['old', 'new']),
                                 ('small_first', ['new', 'old']), ('large_first', ['old', 'new'])]:
            self.config['queue_policy'] = policy
            self.assertEqual([j['id'] for j in order_jobs(jobs, self.config, 1000)], expected)
        self.config['queue_policy'] = 'priority'
        self.assertEqual(order_jobs(jobs, self.config, 1000)[0]['id'], 'old')
        self.config['priority_aging_seconds'] = 10000
        self.assertEqual(order_jobs(jobs, self.config, 1000)[0]['id'], 'new')

    def test_multicard_count_and_exclusive(self):
        active = [dict(pid=123, gpu=None, gpu_ids='[0,1]',
                       spec=json.dumps(dict(gpu_count=2, sharing='exclusive')))]
        config = dict(self.config, observation_seconds=0)
        views = policies(self.views, [], active, {}, {}, config, 10)
        self.assertEqual([g['count'] for g in views], [1, 1, 0])
        self.assertEqual([g['policy'] for g in views], ['EXCLUSIVE', 'EXCLUSIVE', 'READY'])
        active[0]['spec'] = json.dumps(dict(gpu_count=2, sharing='shared'))
        views = policies(self.views, [], active, {}, {}, config, 10)
        self.assertEqual([g['policy'] for g in views], ['READY', 'READY', 'READY'])
        self.assertEqual(allocated_gpus(dict(gpu=1)), [1])


class MultiQueueTests(unittest.TestCase):
    setUp = support.QueueTests.setUp
    tearDown = support.QueueTests.tearDown
    cli = support.QueueTests.cli
    fast = support.QueueTests.fast
    job = support.QueueTests.job
    wait = support.QueueTests.wait

    def cards(self, count=4):
        self.fast()
        self.assertEqual(self.cli('config', '--set', f'simulate_gpu_count={count}'), 0)

    def test_multi_lifecycle_binding_and_json(self):
        self.cards()
        key = self.job('import os; print(os.environ["TETRIS_GPU_COUNT"]); print(repr(os.environ["CUDA_VISIBLE_DEVICES"]))',
                       gpu_count=2, gpus=[1, 3])
        start(self.pool)
        job = self.wait(key, {'complete'})
        self.assertIsNone(job['gpu'])
        self.assertEqual(allocated_gpus(job), [1, 3])
        with transaction(self.pool) as db:
            binding = json.loads(db.execute('SELECT value FROM runtime WHERE key=?',
                                           ('binding:' + job['token'],)).fetchone()[0])
        self.assertEqual(binding, ['SIMULATED-CPU-1', 'SIMULATED-CPU-3'])
        self.assertIn("2\n''", support.Path(job['log']).read_text())
        data = snapshot(self.pool)
        self.assertEqual(data['schema_version'], 2)
        self.assertEqual(data['jobs'][0]['gpu_ids'], [1, 3])
        self.assertEqual(data['jobs'][0]['sharing'], 'exclusive')
        self.assertIn('1,3', render(data, all_jobs=True))

    def test_insufficient_cards_never_claim_partial(self):
        self.cards(1)
        key = self.job(gpu_count=2)
        tick(self.pool, {}, {})
        with transaction(self.pool) as db:
            job = get_job(db, key)
            self.assertEqual(job['attempt'], 0)
            self.assertIsNone(job['token'])
            self.assertEqual(allocated_gpus(job), [])
        self.assertIn('Waiting for 2', job['error'])
        self.assertIn('GPU0:', job['error'])

    def test_strict_fifo_vs_backfill(self):
        self.cards(1)
        head = self.job(gpu_count=2)
        tail = self.job()
        self.assertEqual(self.cli('config', '--set', 'queue_policy="strict_fifo"'), 0)
        tick(self.pool, {}, {})
        with transaction(self.pool) as db:
            self.assertEqual(get_job(db, tail)['attempt'], 0)
        self.assertEqual(self.cli('config', '--set', 'queue_policy="fifo"'), 0)
        tick(self.pool, {}, {})
        self.wait(tail, {'complete'})
        with transaction(self.pool) as db:
            self.assertEqual(get_job(db, head)['attempt'], 0)

    def test_priority_used_by_dispatch(self):
        self.cards(1)
        low = self.job('import time; time.sleep(.3)', priority=-1)
        high = self.job('import time; time.sleep(.3)', priority=10)
        self.assertEqual(self.cli('config', '--set', 'queue_policy="priority"'), 0)
        tick(self.pool, {}, {})
        self.wait(high, {'complete'})
        with transaction(self.pool) as db:
            self.assertEqual(get_job(db, low)['attempt'], 0)

    def test_exclusive_reserves_entire_group_and_releases(self):
        self.cards(2)
        head = self.job('import time; time.sleep(.5)', gpu_count=2)
        tail = self.job()
        start(self.pool)
        self.wait(head, {'running'})
        time.sleep(.1)
        with transaction(self.pool) as db:
            self.assertEqual(get_job(db, tail)['attempt'], 0)
        self.wait(head, {'complete'})
        self.wait(tail, {'complete'})

    def test_shared_multicard_respects_each_card_cap(self):
        self.cards(2)
        keys = [self.job('import time; time.sleep(1)', gpu_count=2, sharing='shared') for _ in range(3)]
        start(self.pool)
        self.wait(keys[1], {'running'})
        with transaction(self.pool) as db:
            self.assertEqual(get_job(db, keys[2])['attempt'], 0)
        for key in keys:
            self.wait(key, {'complete'})

    def test_restart_cancel_and_retry_group(self):
        self.cards(2)
        key = self.job('import time; time.sleep(1)', gpu_count=2)
        start(self.pool)
        before = self.wait(key, {'running'})
        stop(self.pool)
        start(self.pool)
        self.assertEqual(self.cli('cancel', key), 0)
        self.wait(key, {'cancelled'})
        stop(self.pool)
        self.assertEqual(self.cli('retry', key), 0)
        with transaction(self.pool) as db:
            self.assertEqual(allocated_gpus(get_job(db, key)), [])
        start(self.pool)
        after = self.wait(key, {'complete'})
        self.assertEqual(after['attempt'], 2)
        self.assertEqual(allocated_gpus(before), allocated_gpus(after))

    def test_worker_loss_retains_all_cards_until_cancel(self):
        self.cards(2)
        key = self.job('import time; time.sleep(10)', gpu_count=2)
        start(self.pool)
        job = self.wait(key, {'running'})
        time.sleep(.15)
        os.kill(job['pid'], signal.SIGKILL)
        tail = self.job()
        time.sleep(.2)
        with transaction(self.pool) as db:
            self.assertEqual(allocated_gpus(get_job(db, key)), [0, 1])
            self.assertEqual(get_job(db, tail)['attempt'], 0)
        self.assertEqual(self.cli('cancel', key), 0)
        self.wait(key, {'cancelled'})
        self.wait(tail, {'complete'})

    def test_invalid_resources_and_policies(self):
        for kwargs in [dict(gpu_count=0), dict(gpu_count=True), dict(gpu_count=1.5),
                       dict(gpu_count=2, gpus=[0]), dict(sharing='other'), dict(priority=True)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.job(**kwargs)
        for setting in ['queue_policy="bad"', 'placement_policy="bad"',
                        'priority_aging_seconds=0', 'simulate_gpu_count=1.5']:
            self.assertEqual(self.cli('config', '--set', setting), 1)
        self.assertEqual(self.cli('submit', '--gpu-count', '2', '--gpus', '0', '--', 'true'), 1)

    def legacy(self):
        key = self.job()
        with transaction(self.pool) as db:
            db.execute('ALTER TABLE jobs DROP COLUMN gpu_ids')
            db.execute('PRAGMA user_version=0')
            db.execute('UPDATE jobs SET gpu=1 WHERE id=?', (key,))
        return key

    def test_explicit_idempotent_upgrade_and_readonly_old_snapshot(self):
        key = self.legacy()
        data = snapshot(self.pool)
        self.assertEqual(data['jobs'][0]['gpu_ids'], [1])
        with transaction(self.pool) as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 0)
        with self.assertRaisesRegex(ValueError, 'upgrade'):
            submit(self.pool, [sys.executable, '-c', 'pass'])
        upgrade(self.pool)
        upgrade(self.pool)
        with transaction(self.pool) as db:
            self.assertEqual(allocated_gpus(get_job(db, key)), [1])
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 2)

    def test_upgrade_rejects_active_legacy_task(self):
        key = self.legacy()
        with transaction(self.pool) as db:
            db.execute('UPDATE jobs SET status=?,pid=?,identity=? WHERE id=?',
                       ('running', os.getpid(), identity(os.getpid()), key))
        with self.assertRaisesRegex(ValueError, 'active tasks'):
            upgrade(self.pool)
        with transaction(self.pool) as db:
            db.execute('UPDATE jobs SET status=? WHERE id=?', ('failed', key))

    def test_upgrade_recovers_dead_legacy_attempt(self):
        key = self.legacy()
        with transaction(self.pool) as db:
            db.execute('UPDATE jobs SET status=?,token=? WHERE id=?', ('running', 'dead-token', key))
        upgrade(self.pool)
        with transaction(self.pool) as db:
            self.assertEqual(get_job(db, key)['status'], 'failed')
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 2)

    def test_strict_fifo_blocks_on_dependencies(self):
        self.cards(1)
        parent = self.job()
        with transaction(self.pool) as db:
            db.execute('UPDATE jobs SET status=? WHERE id=?', ('failed', parent))
        head = self.job(depends=[parent])
        tail = self.job()
        self.assertEqual(self.cli('config', '--set', 'queue_policy="strict_fifo"'), 0)
        tick(self.pool, {}, {})
        with transaction(self.pool) as db:
            self.assertEqual(get_job(db, head)['status'], 'blocked')
            self.assertEqual(get_job(db, tail)['attempt'], 0)

    def test_config_change_resets_observation_window(self):
        self.cards(2)
        self.assertEqual(self.cli('config', '--set', 'observation_seconds=10', '--set', 'poll_seconds=10'), 0)
        history = {}
        tick(self.pool, history, {}, now=100)
        tick(self.pool, history, {}, now=110)
        self.assertEqual(snapshot(self.pool)['gpus'][0]['policy'], 'READY')
        self.assertEqual(self.cli('config', '--set', 'utilization_threshold=50'), 0)
        tick(self.pool, history, {}, now=120)
        self.assertEqual(snapshot(self.pool)['gpus'][0]['policy'], 'OBSERVE / COOLDOWN')

    def test_spawn_failure_releases_whole_group(self):
        self.cards(2)
        key = self.job(gpu_count=2)
        with patch('gpu_tetris.scheduler.detached', side_effect=OSError('spawn failed')):
            tick(self.pool, {}, {})
        with transaction(self.pool) as db:
            self.assertEqual(get_job(db, key)['status'], 'failed')
        tail = self.job(gpu_count=2)
        tick(self.pool, {}, {})
        self.wait(tail, {'complete'})

    def test_old_single_uuid_binding(self):
        self.cards(1)
        with transaction(self.pool) as db:
            config = dict(DEFAULTS, observation_seconds=0, launch_cooldown_seconds=0)
            db.execute('UPDATE settings SET value=? WHERE key=?', (json.dumps(config), 'config'))
        key = self.job('import os; print(os.environ["CUDA_VISIBLE_DEVICES"])')
        from gpu_tetris.scheduler import put
        def legacy_put(db, name, value):
            put(db, name, value[0] if name.startswith('binding:') else value)
        sample = ([dict(index=0, uuid='GPU-old', free_mb=40000, util=0)], [])
        with patch('gpu_tetris.scheduler.put', side_effect=legacy_put):
            tick(self.pool, {}, {}, sample=sample)
        job = self.wait(key, {'complete'})
        self.assertIn('GPU-old', support.Path(job['log']).read_text())

    def test_real_mode_uuid_binding_without_gpu_execution(self):
        self.cards(2)
        with transaction(self.pool) as db:
            config = dict(DEFAULTS, observation_seconds=0, launch_cooldown_seconds=0)
            db.execute('UPDATE settings SET value=? WHERE key=?', (json.dumps(config), 'config'))
        key = self.job('import os; print(os.environ["CUDA_VISIBLE_DEVICES"])', gpu_count=2)
        sample = ([dict(index=i, uuid=f'GPU-test-{i}', free_mb=40000, util=0) for i in range(2)], [])
        tick(self.pool, {}, {}, sample=sample)
        job = self.wait(key, {'complete'})
        self.assertIn('GPU-test-0,GPU-test-1', support.Path(job['log']).read_text())


if __name__ == '__main__':
    unittest.main()
