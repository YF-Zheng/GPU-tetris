import os
from pathlib import Path
import unittest
from unittest.mock import patch

import test_queue as support
from gpu_tetris.core import get_job, pool_path, transaction
from gpu_tetris.scheduler import runtime, start, tick


class CompatTests(unittest.TestCase):
    setUp = support.QueueTests.setUp
    tearDown = support.QueueTests.tearDown
    cli = support.QueueTests.cli
    fast = support.QueueTests.fast
    job = support.QueueTests.job
    wait = support.QueueTests.wait

    def test_legacy_pool_env_is_still_honoured(self):
        with patch.dict(os.environ, {'UNIFIED_GPU_POOL': '/tmp/legacy-pool'}):
            os.environ.pop('TETRIS_POOL', None)
            self.assertEqual(pool_path(), Path('/tmp/legacy-pool').resolve())
            os.environ['TETRIS_POOL'] = '/tmp/new-pool'
            self.assertEqual(pool_path(), Path('/tmp/new-pool').resolve())

    def test_legacy_task_env_aliases(self):
        self.fast()
        key = self.job('import os; print(os.environ["TETRIS_JOB_ID"], os.environ["UNIFIED_JOB_ID"], '
                       'os.environ["TETRIS_GPU_COUNT"], os.environ["UNIFIED_GPU_COUNT"])')
        tick(self.pool, {}, {})
        job = self.wait(key, {'complete'})
        self.assertEqual(Path(job['log']).read_text().split(), [key, key, '1', '1'])

    def test_restart_does_not_touch_running_tasks(self):
        self.fast()
        key = self.job('import time; time.sleep(30)')
        start(self.pool)
        before = self.wait(key, {'running'})
        with transaction(self.pool) as db:
            old_daemon = runtime(db, 'daemon', {})
        self.assertEqual(self.cli('restart'), 0)
        with transaction(self.pool) as db:
            new_daemon = runtime(db, 'daemon', {})
            after = get_job(db, key)
        self.assertNotEqual(old_daemon['pid'], new_daemon['pid'])
        self.assertEqual((after['status'], after['pid'], after['attempt']),
                         ('running', before['pid'], before['attempt']))


if __name__ == '__main__':
    unittest.main()
