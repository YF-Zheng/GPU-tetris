import subprocess
import sys
import time
import unittest

import test_queue as support
from gpu_tetris import keepers
from gpu_tetris.view import snapshot
from gpu_tetris.scheduler import tick


class FakeBackend:
    name = 'fake'

    def __init__(self):
        self.busy = 0.0
        self.now = 0.0

    def burn(self, seconds):
        self.busy += seconds
        self.now += seconds


def parse(*argv):
    return keepers.resolve(keepers.build_parser().parse_args(list(argv)))


class KeeperLogicTests(unittest.TestCase):
    def test_parse_mem(self):
        gib = 1 << 30
        self.assertEqual(keepers.parse_mem('4G', 0), 4 * gib)
        self.assertEqual(keepers.parse_mem('512M', 0), 512 << 20)
        self.assertEqual(keepers.parse_mem('2048', 0), 2048 << 20)
        self.assertEqual(keepers.parse_mem('0', 0), 0)
        self.assertEqual(keepers.parse_mem('50%', 10 * gib), 5 * gib)
        for bad in ('-1', '0%', '99%', 'abc', 'nan', 'inf', '1e309', '1e308G'):
            with self.assertRaises(ValueError):
                keepers.parse_mem(bad, gib)

    def test_defaults_and_validation(self):
        self.assertEqual(parse('steady').util, 30.0)
        self.assertEqual(parse('steady', '--util', '45').util, 45.0)
        self.assertEqual(parse('burn').sm, 100.0)
        for argv in (['steady', '--util', '150'], ['wave', '--lo', '80', '--hi', '20'], ['floor', '--util', '0'],
                     ['steady', '--sm', '0'], ['heartbeat', '--period', '0'], ['steady', '--mem', 'x'],
                     ['heartbeat', '--burst', '31'], ['steady', '--report', 'nan'],
                     ['steady', '--report', '-1'], ['steady', '--max-seconds', 'inf'],
                     ['steady', '--max-seconds', '-1'], ['steady', '--util', 'nan']):
            with self.assertRaises(ValueError):
                parse(*argv)

    def test_open_loop_plans(self):
        self.assertAlmostEqual(keepers.plan_for(parse('steady', '--util', '30'))(5, None), 0.3)
        self.assertAlmostEqual(keepers.plan_for(parse('burn'))(5, None), 1.0)
        beat = keepers.plan_for(parse('heartbeat', '--burst', '2', '--period', '60'))
        self.assertEqual((beat(1, None), beat(2.5, None), beat(61, None)), (1.0, 0.0, 1.0))
        wave = keepers.plan_for(parse('wave', '--lo', '10', '--hi', '50', '--period', '100'))
        self.assertAlmostEqual(wave(0, None), 0.1)
        self.assertAlmostEqual(wave(50, None), 0.5)

    def test_duty_cycle_accuracy(self):
        backend = FakeBackend()
        def sleep(seconds):
            backend.now += seconds
        keepers.run_loop(backend, keepers.plan_for(parse('steady', '--util', '25')), 0.2,
                         clock=lambda: backend.now, sleep=sleep, max_seconds=100)
        self.assertAlmostEqual(backend.busy / backend.now, 0.25, delta=0.01)

    def simulate_floor(self, others, target=40.0, seconds=300):
        """Plant model: measured util = min(100, others + keeper duty*100)."""
        ctl = keepers.FloorController(target, interval=2.0)
        history = []
        for t in range(seconds):
            obs = min(100.0, others(t) + ctl.duty * 100)
            history.append((ctl(t, obs), obs))
        return ctl, history

    def test_floor_fills_idle_card_up_to_target(self):
        ctl, history = self.simulate_floor(lambda t: 0.0)
        self.assertAlmostEqual(history[-1][1], 40.0, delta=3.0)

    def test_floor_backs_off_when_others_are_busy(self):
        ctl, history = self.simulate_floor(lambda t: 0.0 if t < 100 else 85.0)
        self.assertAlmostEqual(history[99][1], 40.0, delta=3.0)
        self.assertAlmostEqual(history[-1][0], ctl.lo, delta=0.01)  # nearly out of the way
        self.assertGreaterEqual(history[-1][1], 85.0)              # never lowers the total

    def test_floor_without_measurement_holds_duty(self):
        ctl = keepers.FloorController(40.0)
        before = ctl(0, None)
        self.assertEqual(ctl(10, None), before)

    def test_stop_flag_ends_loop(self):
        backend = FakeBackend()
        calls = []
        keepers.run_loop(backend, lambda t, o: 0.0, 0.1, clock=lambda: backend.now,
                         sleep=lambda s: calls.append(s), stop=lambda: len(calls) >= 3)
        self.assertEqual(len(calls), 3)


class KeeperCliTests(unittest.TestCase):
    setUp = support.QueueTests.setUp
    tearDown = support.QueueTests.tearDown
    cli = support.QueueTests.cli
    fast = support.QueueTests.fast
    wait = support.QueueTests.wait

    def test_submit_keeper_per_gpu(self):
        self.fast()
        self.assertEqual(self.cli('config', '--set', 'filler_enabled=true', '--set', 'filler_rearm_seconds=0',
                                  '--set', 'simulate_gpu_count=2'), 0)
        self.assertEqual(self.cli('submit', '--keeper', 'steady', '--per-gpu', '--gpus', '0,1', '--', '--util', '20',
                                  '--max-seconds', '1'), 0)
        jobs = snapshot(self.pool)['jobs']
        self.assertEqual(sorted(j['name'] for j in jobs), ['keeper-steady-gpu0', 'keeper-steady-gpu1'])
        self.assertTrue(all(j['filler'] for j in jobs))
        self.assertEqual(sorted(j['command'][2] for j in jobs), ['steady', 'steady'])

    def test_bad_keeper_arguments_rejected(self):
        self.assertEqual(self.cli('submit', '--keeper', 'nope', '--'), 1)
        self.assertEqual(self.cli('submit', '--keeper', 'steady', '--', '--util', '999'), 1)
        self.assertEqual(self.cli('submit', '--per-gpu', '--', 'echo'), 1)
        self.assertEqual(self.cli('submit', '--keeper', 'steady', '--per-gpu'), 1)
        self.assertEqual(self.cli('submit', '--keeper-python', sys.executable, '--', 'echo'), 1)
        for options in (['--mem', 'inf'], ['--report', 'nan'], ['--max-seconds', '-1']):
            self.assertEqual(self.cli('submit', '--keeper', 'steady', '--', *options), 1)
        self.assertEqual(self.cli('submit', '--keeper', 'heartbeat', '--', '--burst', '40'), 1)
        self.assertEqual(snapshot(self.pool)['jobs'], [])

    def test_none_backend_does_not_require_torch(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from gpu_tetris.core import initialize
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp) / 'real'
            initialize(pool)
            with patch('gpu_tetris.cli.subprocess.run', side_effect=AssertionError('must not probe torch')):
                self.assertEqual(support.main(['--pool', str(pool), 'submit', '--keeper', 'steady',
                                               '--', '--backend', 'none', '--max-seconds', '1']), 0)
            self.assertEqual(len(snapshot(pool)['jobs']), 1)

    def test_standalone_probe_defaults_to_first_gpu(self):
        from unittest.mock import patch
        import os
        with patch.dict(os.environ, {}, clear=True), \
                patch('gpu_tetris.keepers.ctypes.CDLL', side_effect=OSError), \
                patch('gpu_tetris.keepers.subprocess.run') as run:
            run.return_value.stdout = '42\n'
            self.assertEqual(keepers.util_probe()(), 42)
            self.assertEqual(run.call_args.args[0][-2:], ['-i', '0'])

    def test_keeper_runs_in_simulation_pool_and_exits_on_cancel(self):
        self.fast()
        self.cli('config', '--set', 'filler_enabled=true')
        self.assertEqual(self.cli('submit', '--keeper', 'floor', '--', '--util', '40', '--report', '1'), 0)
        key = snapshot(self.pool)['jobs'][0]['id']
        tick(self.pool, {}, {})
        job = self.wait(key, {'running'})
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and 'preset=floor' not in support.Path(job['log']).read_text():
            time.sleep(.1)
        self.assertIn('preset=floor backend=none', support.Path(job['log']).read_text())
        self.assertEqual(self.cli('cancel', key), 0)
        self.wait(key, {'cancelled'})

    def test_real_gpu_pool_requires_torch_in_interpreter(self):
        # A non-simulated pool must refuse a keeper whose interpreter cannot import torch.
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        from gpu_tetris.core import initialize
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / 'real'
            initialize(real, simulate=False)
            with contextlib.redirect_stderr(io.StringIO()) as err, contextlib.redirect_stdout(io.StringIO()):
                code = support.main(['--pool', str(real), 'submit', '--keeper', 'steady',
                                     '--keeper-python', sys.executable, '--', '--util', '10'])
            probe = subprocess.run([sys.executable, '-c', 'import importlib.util;raise SystemExit(0 if importlib.util.find_spec("torch") else 3)'])
            self.assertEqual(code, 0 if probe.returncode == 0 else 1)
            if probe.returncode != 0:
                self.assertIn('PyTorch not found', err.getvalue())

    def test_standalone_cli_exit_codes(self):
        run = lambda *a: subprocess.run([sys.executable, '-m', 'gpu_tetris.keepers', *a], capture_output=True, text=True, timeout=30)
        self.assertEqual(run('steady', '--backend', 'none', '--max-seconds', '0.3', '--mem', '1M').returncode, 0)
        self.assertEqual(run('steady', '--util', '200').returncode, 2)


if __name__ == '__main__':
    unittest.main()
