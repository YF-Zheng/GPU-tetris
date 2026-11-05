# 真实 GPU 集成测试清单（H20 / 任意 NVIDIA 多卡机器）

CI 与 68 项单元测试使用 CPU 模拟池，**不覆盖真实 GPU**。公开宣称“已在真机验证”前，请在独立 pool 上完成下列项目并记录结果（驱动、CUDA、GPU 型号、tetris-smi 版本、结果）。

建议环境：`export TETRIS_POOL=~/.local/state/gpu-tetris-e2e`，`tetris init`，`tetris doctor` 通过，且没有其他训练在测试卡上运行。

| # | 场景 | 步骤 | 期望 |
| - | --- | --- | --- |
| 1 | 查询与 UUID 绑定 | 提交 `python -c "import os;print(os.environ['CUDA_VISIBLE_DEVICES'])"` | 输出为 `GPU-...` UUID，且与 `nvidia-smi -L` 对应 |
| 2 | 单卡共置 | 同一卡上先后提交两个真实训练（显存足够） | 第二个在 60 秒连续低利用率观察窗口后启动；每卡不超过 `per_gpu_limit` |
| 3 | 高利用率拒绝 | 一个训练使 GPU-Util >= 65% 时提交第二个 | 状态显示 `HOLD >=65%`，任务保持排队 |
| 4 | 低显存拒绝 | 用 `--min-free-mb` 设置超过空闲显存的值 | 状态 `LOW MEMORY`，`DETAIL` 给出逐卡原因 |
| 5 | 多卡原子分配 | `--gpu-count 4` 且仅 3 卡空闲 | 不占用任何卡，`attempt` 不增加 |
| 6 | 多卡 DDP | `torchrun --standalone --nproc-per-node=N` | N 个 rank 看到正确的逻辑卡 0..N-1，退出码正确 |
| 7 | 未知进程 | 手动在某卡上运行不受管 CUDA 进程 | 该卡 `UNKNOWN OWNER`，不派发 |
| 8 | 容器/PID 命名空间 | 在容器内运行时检查 `nvidia-smi --query-compute-apps` 的 PID 能否映射 | 记录是否出现误报 `UNKNOWN OWNER` |
| 9 | 调度器重启 | 训练中 `tetris stop` 再 `start` | 训练不被重启，状态保持 `running` |
| 10 | Worker 丢失 | `kill -9` worker（非训练进程） | 槽位保留；`cancel` 后清理子进程 |
| 11 | OOM / 非零退出 | 提交必然 OOM 的任务 | 状态 `failed`，退出码被记录，GPU 槽位释放，不自动重跑 |
| 12 | Keeper 回收 | 启用 filler，普通任务提交到 filler 所在卡 | filler 退出并重新排队；普通任务启动；之后至少等待 `filler_rearm_seconds` 才再补入 |
| 13 | NVIDIA 查询失败 | 临时让 `nvidia-smi` 不可用（PATH 覆盖） | 调度器不派发，状态显示 `GPU QUERY ERROR` |
| 14 | 取消 | 取消正在使用 GPU 的训练 | 进程组被清理，`nvidia-smi` 中无残留，状态 `cancelled` |
| 15 | Keeper steady | `tetris submit --keeper steady --gpus 0 -- --util 30 --max-seconds 60` | `nvidia-smi` 的 GPU-Util 稳定在约 30%，显存约 512 MiB，结束后释放 |
| 16 | 缺少 PyTorch | 用没有 torch 的解释器：`--keeper-python /usr/bin/python3` | 提交时直接报错 `PyTorch not found`，任务不入队 |
| 17 | Keeper floor 闭环 | `--keeper floor -- --util 40`，另开一个训练把卡跑到 80% | 空闲时总利用率约 40%；训练开始后 keeper 占空比降到最低，总利用率不被拉低 |
| 18 | Keeper 的 `--sm` 影响 | 在有真实训练的卡上分别用 `--sm 10` 和 `--sm 100` | `--sm 10` 对训练吞吐影响明显小于 `--sm 100`，记录两者的训练速度 |
| 19 | Keeper 让路 | 开启 filler，提交需要该卡的普通任务 | Keeper 在数秒内退出并释放显存，普通任务启动 |

通过后可将结果表附在 Release 说明里，并把 README 的“实卡验证”状态更新为具体硬件与日期。
