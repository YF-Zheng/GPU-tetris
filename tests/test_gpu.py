import subprocess
import unittest
from unittest.mock import patch

from gpu_tetris.core import DEFAULTS, validate_config
from gpu_tetris.gpu import query


class GpuQueryTests(unittest.TestCase):
    def test_single_user_skips_unavailable_process_query(self):
        def run(argv, **kwargs):
            if '--query-compute-apps=pid,gpu_uuid' in argv:
                raise subprocess.CalledProcessError(1, argv)
            return subprocess.CompletedProcess(argv, 0, '\n0, GPU-test, 30000, 10\n\n', '')
        with patch('gpu_tetris.gpu.subprocess.run', side_effect=run) as probe:
            cards, apps = query(query_apps=False)
            self.assertEqual(cards[0]['free_mb'], 30000)
            self.assertEqual(apps, [])
            self.assertEqual(probe.call_count, 1)
            with self.assertRaises(subprocess.CalledProcessError):
                query(query_apps=True)

    def test_unavailable_metrics_are_not_treated_as_idle(self):
        for output in ('0, GPU-test, 30000, N/A\n', '0, GPU-test, N/A, 0\n',
                       '0, GPU-test, 30000\n', '0, GPU-test, 30000, 101\n'):
            with self.subTest(output=output), patch('gpu_tetris.gpu.subprocess.run') as run:
                run.return_value.stdout = output
                with self.assertRaisesRegex(ValueError, 'GPU metrics'):
                    query(query_apps=False)

    def test_empty_gpu_query_is_an_error(self):
        with patch('gpu_tetris.gpu.subprocess.run') as run:
            run.return_value.stdout = '\n'
            with self.assertRaisesRegex(ValueError, 'No NVIDIA GPUs'):
                query(query_apps=False)

    def test_strict_process_csv(self):
        with patch('gpu_tetris.gpu.subprocess.run') as run:
            run.side_effect = [subprocess.CompletedProcess([], 0, '0, GPU-test, 30000, 0\n'),
                               subprocess.CompletedProcess([], 0, '\n123, GPU-test\n')]
            self.assertEqual(query()[1], [(123, 'GPU-test')])

    def test_ownership_mode_validation(self):
        for mode in ('single_user', 'strict'):
            validate_config(dict(DEFAULTS, ownership_mode=mode))
        for mode in ('ignore', None, False):
            with self.assertRaisesRegex(ValueError, 'ownership_mode'):
                validate_config(dict(DEFAULTS, ownership_mode=mode))
