# 变更记录

## 0.3.0（Alpha，未发布）

- **改名为 GPU Tetris**：包名 `gpu-tetris`，Python 包 `gpu_tetris`，命令 `tetris` / `tetris-smi`，环境变量 `TETRIS_*`。
- **兼容旧名字**：仍读取 `UNIFIED_GPU_POOL`；自动沿用已存在的 `~/.local/state/unified-gpu` 池；任务内保留 `UNIFIED_*` 别名（已弃用）；取消/清理能识别旧版 worker 启动的任务。
- 新增内置 Keeper 预设（`steady`/`floor`/`wave`/`heartbeat`/`burn`），产生**计算利用率**：`--util` 控制 GPU-Util，`--sm` 控制占用的 SM 比例；`floor` 为闭环，读取整卡利用率只补缺口。新增 `tetris keepers` 与 `submit --keeper [--per-gpu]`。需要 PyTorch（CUDA 版）；提交时会检查所选解释器有无 torch。尚未在真机验证。
- 新增 `tetris restart`：热更新调度服务（预检新版本、不影响运行中的任务）。
- 新增 `filler_rearm_seconds`（默认 1800）：占卡任务被调度器回收后，同一 GPU 必须再等待该时长才允许补入新的占卡任务，并持久化于池内，重启调度器后依然有效。此前占卡准入只使用 `observation_seconds`（60 秒）。
- 排队原因包含逐卡诊断，例如 `GPU0: HOLD >=65%; GPU1: FULL; GPU2: LOW MEMORY 100<24000 MiB`。
- 新增 `LICENSE`（MIT）、`docs/real-gpu-validation.md`、`docs/benchmark.md`。

## 已知限制 / 路线图

- 占卡回收的预测仍较粗糙：假定移除占卡后满足普通任务的显存需求，可能产生无效回收。
- 没有基于历史的显存预测；`--min-free-mb` 由用户手工声明。
- 多卡任务在持续有小任务提交时可能饥饿；可用 `strict_fifo` / `large_first` 缓解，预留式回填留待后续版本。
- 真实 NVIDIA GPU 的端到端验证见 `docs/real-gpu-validation.md`。
