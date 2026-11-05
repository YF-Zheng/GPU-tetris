"""Built-in keeper (filler) programs that generate GPU *compute* utilization.

Needs PyTorch to run on a real GPU:
    python -m gpu_tetris.keepers PRESET [options]
or  tetris submit --keeper PRESET --gpus 0 -- [options]

What "utilization" means here: nvidia-smi's GPU-Util is the share of time in which at least
one kernel is running. So two independent knobs are exposed:
  --util  how much of the time a kernel is in flight (the number nvidia-smi shows)
  --sm    how many SMs that kernel occupies, in % of the card (how hard it competes with
          other work on the same card; a small value barely slows real training)

Requires PyTorch (CUDA build) in the interpreter that runs it; use --keeper-python to point
at one. With an empty CUDA_VISIBLE_DEVICES (simulation pools) the "none" backend only sleeps,
so CI and demos need no GPU or torch. A keeper must be interruptible at any time: SIGTERM frees all
device memory and exits 0.
"""

import argparse
import ctypes
import math
import os
import signal
import subprocess
import sys
import time

CHUNK = 256 << 20

# name -> (one-line summary, defaults)
PRESETS = {
    'steady': ('Fixed compute load: GPU-Util held near --util (open loop)',
               dict(util=30.0, sm=25.0, mem='512M')),
    'floor': ('Closed loop: keep TOTAL GPU-Util at or above --util; backs off when others are busy',
              dict(util=40.0, sm=25.0, mem='512M')),
    'wave': ('Load that rises and falls like a sine wave between --lo and --hi',
             dict(lo=20.0, hi=60.0, period=600.0, sm=25.0, mem='512M')),
    'heartbeat': ('Short full-rate bursts: --burst seconds busy at the start of every --period',
                  dict(burst=3.0, period=30.0, sm=50.0, mem='512M')),
    'burn': ('Continuous 100% load on every SM (maximum occupancy)',
             dict(util=100.0, sm=100.0, mem='512M')),
}


# ---------------------------------------------------------------- pure helpers
def parse_mem(text, free_bytes):
    """'4G', '512M', '2048' (MiB), '60%' (of currently free memory) -> bytes."""
    s = str(text).strip().upper().replace('IB', '').replace('B', '')
    if s.endswith('%'):
        frac = float(s[:-1]) / 100
        if not 0 < frac <= 0.95:
            raise ValueError('--mem percentage must be in (0, 95]')
        return int(free_bytes * frac)
    scale = 1 << 20
    if s.endswith('G'):
        scale, s = 1 << 30, s[:-1]
    elif s.endswith('M'):
        s = s[:-1]
    value = float(s)
    if value < 0:
        raise ValueError('--mem must be nonnegative')
    return int(value * scale)


class FloorController:
    """Integral controller on the measured total GPU-Util.

    Raises the keeper's duty while util < target and lowers it when util >= target, so the
    keeper fills the gap and gets out of the way when real work already keeps the card busy.
    """

    def __init__(self, target, interval=2.0, gain=0.6, lo=0.02, hi=0.95, start=0.3):
        self.target, self.interval, self.gain = target, interval, gain
        self.lo, self.hi, self.duty = lo, hi, start
        self.next_at = 0.0

    def __call__(self, t, obs):
        if obs is None:  # no measurement: hold the current duty (open loop)
            return self.duty
        if t >= self.next_at:  # one correction per reading interval
            self.next_at = t + self.interval
            self.duty = min(self.hi, max(self.lo, self.duty + self.gain * (self.target - obs) / 100.0))
        return self.duty


def plan_for(o):
    """Return f(elapsed_seconds, observed_util_percent_or_None) -> duty cycle in [0, 1]."""
    name = o.preset
    if name in ('steady', 'burn'):
        return lambda t, obs: o.util / 100.0
    if name == 'floor':
        return FloorController(o.util, o.interval)
    if name == 'wave':
        mid, amp = (o.lo + o.hi) / 200.0, (o.hi - o.lo) / 200.0
        return lambda t, obs: mid - amp * math.cos(2 * math.pi * t / o.period)
    if name == 'heartbeat':
        return lambda t, obs: 1.0 if (t % o.period) < o.burst else 0.0
    raise ValueError(f'unknown preset {name}')


def run_loop(backend, plan, slice_s, clock=time.monotonic, sleep=time.sleep,
             stop=lambda: False, observe=None, report=None, max_seconds=0):
    """Duty-cycle driver: each short slice is busy for duty*slice, then idle for the rest."""
    start = clock()
    last_report = start
    while not stop():
        now = clock() - start
        if max_seconds and now >= max_seconds:
            break
        obs = observe() if observe else None
        duty = min(1.0, max(0.0, plan(now, obs)))
        began = clock()
        if duty > 0:
            backend.burn(slice_s * duty)
        rest = slice_s - (clock() - began)
        if rest > 0:
            sleep(rest)
        if report and clock() - last_report >= report[0]:
            last_report = clock()
            report[1](now, duty, obs)


# ------------------------------------------------------------------- backends
class NoneBackend:
    name = 'none'

    def mem_info(self):
        return 96 << 30, 96 << 30

    def alloc(self, nbytes):
        pass

    def burn(self, seconds):
        time.sleep(seconds)

    def close(self):
        pass


class TorchBackend:
    """cuBLAS matmuls; the matrix size is chosen so roughly sm% of the SMs stay busy."""
    name = 'torch'

    def __init__(self, sm_pct):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('torch.cuda is not available')
        self.t, self.dev, self.held = torch, torch.device('cuda:0'), []
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        blocks = max(1.0, sms * sm_pct / 100.0)
        self.n = max(128, int(128 * math.sqrt(blocks)))  # ~one 128x128 output tile per SM
        self.a = torch.randn(self.n, self.n, device=self.dev)
        self.b = torch.randn(self.n, self.n, device=self.dev)

    def mem_info(self):
        return self.t.cuda.mem_get_info(0)

    def alloc(self, nbytes):
        block = self.t.empty(nbytes, dtype=self.t.uint8, device=self.dev)
        block.zero_()
        self.held.append(block)

    def burn(self, seconds):
        end = time.perf_counter() + seconds
        while True:
            for _ in range(8):
                self.a @ self.b
            self.t.cuda.synchronize()
            if time.perf_counter() >= end:
                break

    def close(self):
        self.held.clear()
        self.t.cuda.empty_cache()


def make_backend(choice, sm_pct):
    """PyTorch is required for real GPUs; "none" (auto under an empty CUDA_VISIBLE_DEVICES) only sleeps."""
    if choice == 'none' or (choice == 'auto' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''):
        return NoneBackend()
    try:
        return TorchBackend(sm_pct)
    except ImportError:
        raise RuntimeError('PyTorch is required for keeper presets (pip install torch in this interpreter, '
                           'or pass --keeper-python PATH). Use --backend none for a dry run.') from None


# ------------------------------------------------------- utilization measurement
class _Rates(ctypes.Structure):
    _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]


def util_probe(interval=2.0):
    """Cached reader of the whole card's GPU-Util (NVML, then nvidia-smi). None if unavailable."""
    target = (os.environ.get('CUDA_VISIBLE_DEVICES') or '').split(',')[0].strip()
    state = dict(at=-1e9, value=None)
    nvml = handle = None
    try:
        nvml = ctypes.CDLL('libnvidia-ml.so.1')
        if nvml.nvmlInit_v2() == 0 and target:
            handle = ctypes.c_void_p()
            if target[:4] in ('GPU-', 'MIG-'):
                code = nvml.nvmlDeviceGetHandleByUUID(target.encode(), ctypes.byref(handle))
            else:
                code = nvml.nvmlDeviceGetHandleByIndex_v2(int(target), ctypes.byref(handle))
            if code != 0:
                handle = None
    except (OSError, ValueError, AttributeError):
        nvml = handle = None

    def read():
        if handle is not None:
            rates = _Rates()
            if nvml.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(rates)) == 0:
                return float(rates.gpu)
        if target:
            try:
                out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits',
                                      '-i', target], capture_output=True, text=True, timeout=5, check=True).stdout
                return float(out.split()[0])
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
        return None

    def probe():
        now = time.monotonic()
        if now - state['at'] >= interval:
            state['at'] = now
            value = read()
            state['value'] = value
        return state['value']
    return probe


# ------------------------------------------------------------------------ main
def build_parser():
    ap = argparse.ArgumentParser(prog='keepers', description='Built-in GPU compute-load keeper presets.')
    ap.add_argument('preset', choices=sorted(PRESETS))
    ap.add_argument('--util', type=float, help='steady/floor/burn: target GPU-Util in %% (floor: minimum TOTAL util)')
    ap.add_argument('--lo', type=float, help='wave: minimum util %%')
    ap.add_argument('--hi', type=float, help='wave: maximum util %%')
    ap.add_argument('--period', type=float, help='wave/heartbeat: seconds per cycle')
    ap.add_argument('--burst', type=float, help='heartbeat: busy seconds at the start of each period')
    ap.add_argument('--sm', type=float, help='%% of the SMs the load occupies (small = light touch on shared cards)')
    ap.add_argument('--mem', help='Memory to hold: 512M, 4G, 2048 (MiB), 60%% of free, 0 = none')
    ap.add_argument('--backend', choices=('auto', 'torch', 'none'), default='auto')
    ap.add_argument('--slice', type=float, default=0.2, help='control slice in seconds (keep below 1 s)')
    ap.add_argument('--interval', type=float, default=2.0, help='floor: seconds between utilization readings')
    ap.add_argument('--report', type=float, default=60.0, help='status line interval in seconds (0 = off)')
    ap.add_argument('--max-seconds', dest='max_seconds', type=float, default=0.0,
                    help='exit after this many seconds (0 = until terminated); for trials')
    return ap


def resolve(args):
    for key, value in PRESETS[args.preset][1].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    for key in ('util', 'lo', 'hi'):
        value = getattr(args, key, None)
        if value is not None and not 0 <= value <= 100:
            raise ValueError(f'--{key} must be between 0 and 100')
    if args.sm is not None and not 0 < args.sm <= 100:
        raise ValueError('--sm must be in (0, 100]')
    if args.preset == 'wave' and args.lo > args.hi:
        raise ValueError('--lo must not exceed --hi')
    if args.preset == 'floor' and args.util <= 0:
        raise ValueError('--util must be positive for floor')
    for key in ('period', 'burst', 'slice', 'interval'):
        value = getattr(args, key, None)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f'--{key} must be positive')
    parse_mem(args.mem, 1 << 30)  # validate early
    return args


def main(argv=None):
    try:
        args = resolve(build_parser().parse_args(argv))
        backend = make_backend(args.backend, args.sm)
    except (ValueError, RuntimeError) as exc:
        print(f'[keeper] error: {exc}', file=sys.stderr, flush=True)
        return 2
    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append(1))
    signal.signal(signal.SIGINT, lambda *_: stopping.append(1))
    try:
        free, _total = backend.mem_info()
        target = parse_mem(args.mem, free)
        held = 0
        while held < target and not stopping:
            size = min(CHUNK, target - held)
            try:
                backend.alloc(size)
            except Exception as exc:  # torch raises its own OutOfMemoryError type
                print(f'[keeper] allocation stopped at {held >> 20} MiB: {exc}', flush=True)
                break
            held += size
        if target > 0 and held == 0:
            print('[keeper] error: could not allocate any memory', file=sys.stderr, flush=True)
            return 1
        measured = util_probe(args.interval) if backend.name != 'none' else None
        if args.preset == 'floor' and measured is not None and measured() is None:
            print('[keeper] warning: cannot read GPU utilization; floor falls back to open loop', flush=True)
        print(f'[keeper] preset={args.preset} backend={backend.name} sm={args.sm:g}% held={held >> 20} MiB', flush=True)

        def report(t, duty, obs):
            seen = '' if obs is None else f' gpu_util={obs:.0f}%'
            print(f'[keeper] t={t:.0f}s keeper_duty={duty * 100:.0f}%{seen}', flush=True)

        run_loop(backend, plan_for(args), args.slice, stop=lambda: bool(stopping),
                 observe=measured, report=(args.report, report) if args.report else None,
                 max_seconds=args.max_seconds)
        return 0
    finally:
        backend.close()


if __name__ == '__main__':
    raise SystemExit(main())
