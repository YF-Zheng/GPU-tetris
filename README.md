# GPU Tetris

独立的单机 GPU 任务队列与终端监控工具。无需训练框架、Conda、数据库服务或 Python 第三方运行时依赖。

**当前版本：0.3.0 Alpha。支持 Linux、Python 3.10+、NVIDIA GPU。** 适合单用户工作站或服务器，不是多租户集群调度器。

## 安装

推荐在克隆仓库后用 pipx 安装：

```sh
pipx install .
pipx ensurepath
```

重新打开终端后执行 `tetris --help`。也可使用虚拟环境，避免 PATH 问题：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/tetris --help
```

以下示例假定命令已加入 PATH。虚拟环境用户也可使用 `.venv/bin/tetris`。
`tetris-smi` 是 `tetris status` 的独立短入口。

## GPU 快速开始

```sh
tetris init
tetris doctor
tetris start
tetris submit --name experiment-1 --cwd /path/to/project -- python3 train.py --epochs 100
tetris-smi --watch 2
```

`submit` 返回任务 ID，后续用该 ID 查看日志或操作任务。命令按 argv 直接执行，不隐式启动 shell；管道等 shell 功能必须明确使用 `bash -lc '...'`。只提交可信命令。

默认调度规则：

- 每张卡最多 2 个共享任务；不指定卡数时申请 1 张，支持显式多卡申请。
- 每 10 秒采样；利用率连续 60 秒低于 65% 才补入一项。
- 补入后至少冷却 60 秒，并重新积累观察窗口。
- 空闲显存至少 16000 MiB，可按任务进一步提高要求。
- CUDA 使用 GPU UUID 绑定，避免设备枚举顺序差异。
- 默认 `ownership_mode="single_user"`：按整卡利用率与空闲显存补位，不要求 GPU 进程属于本队列；GPU 指标查询失败时不派发。
- 按提交时间顺序选择符合条件的任务；阻塞任务不阻挡其余任务。

资源监控不是显存预留，也不是对其他程序的强制限制。`READY` 仅表示队列的资源条件满足，不保证某个具体任务马上派发。

## 无 GPU 演示

使用单独的模拟池，执行真实 CPU 子进程，不是伪造 GPU 训练：

```sh
export TETRIS_POOL="$HOME/.local/state/gpu-tetris-demo"
tetris init --simulate
tetris config --set observation_seconds=0 --set launch_cooldown_seconds=0 --set poll_seconds=1
tetris start
tetris submit --name hello -- python3 -c 'print("hello queue")'
tetris-smi --watch 1
tetris stop
```

模拟模式设置 `CUDA_VISIBLE_DEVICES` 为空，只用于测试与演示。模式在初始化时确定，不能在同一池中切换。不要把模拟池用于 GPU 训练。

## 常用操作

| 命令 | 功能 |
| --- | --- |
| `tetris init` | 初始化独立队列 |
| `tetris start` | 启动后台调度服务，重复调用不会启动第二个服务 |
| `tetris run` | 前台运行，方便 systemd 等进程管理器托管 |
| `tetris stop` | 停止调度服务，已运行任务继续执行 |
| `tetris restart` | 热更新：重启调度服务，运行中的任务不受影响 |
| `tetris submit -- PROGRAM ARG...` | 独立提交任务，返回 ID |
| `tetris-smi --watch 2` | 每 2 秒显示调度快照，Ctrl+C 退出 |
| `tetris-smi --all` | 包含已完成任务 |
| `tetris-smi --status failed` | 只显示失败任务，可重复指定状态 |
| `tetris-smi --json` | 输出版本化 JSON，不包含任务环境变量 |
| `tetris logs JOB_ID -f --tail 50` | 跟随当前尝试日志 |
| `tetris cancel JOB_ID` | 取消等待任务，或请求 worker 终止运行任务 |
| `tetris retry JOB_ID` | 手动重试失败或已取消任务，增加尝试编号 |
| `tetris pause` / `tetris resume` | 暂停／恢复新任务派发，不停止运行任务 |
| `tetris config` | 查看调度配置 |
| `tetris doctor` | 检查池与 NVIDIA 查询 |

任务状态：`queued`、`blocked`、`starting`、`running`、`complete`、`failed`、`cancelled`。
完成由主命令退出码 0 确定；非零退出码视为失败。普通任务没有隐式自动重试；占卡仅在调度器回收后自动重新排队。
取消是异步请求，运行中任务只有清理结束后才变为 `cancelled`。正常 worker 即使调度器已停止也能响应取消；worker 异常丢失时需启动调度器完成清理。

### 指定资源、环境和依赖

```sh
tetris submit --name train --gpus 0,1 --min-free-mb 24000 --env OMP_NUM_THREADS=4 -- python3 train.py
tetris submit --name evaluate --depends JOB_ID -- python3 evaluate.py
tetris config --set per_gpu_limit=1 --set utilization_threshold=50 --set 'allowed_gpus=[0,1]'
```

依赖必须是已提交的完整任务 ID；所有父任务成功后才运行。失败或取消的父任务令子任务保持阻塞，可在处理父任务后重试。不支持优先级抢占。

## 可组合调度策略

队列排序使用 `queue_policy`，选卡使用 `placement_policy`；两者独立，不绕过利用率、观察窗口、冷却、显存、未知消费者和每卡上限检查。

| 队列策略 | 行为与取舍 |
| --- | --- |
| `fifo`（默认） | 按提交时间尝试，资源不足或依赖阻塞时允许后续任务回填；不是基于运行时间预估的 EASY backfill |
| `strict_fifo` | 队首资源不足或依赖阻塞时不再派发后续任务，可减轻大任务饥饿，但会造成队首阻塞 |
| `priority` | 有效优先级高者先运行，同分按提交时间；有效值为 `priority + floor(等待秒数 / priority_aging_seconds)`，默认每 300 秒加 1 |
| `small_first` | 申请卡数少者优先，不是预计运行时短者优先 |
| `large_first` | 申请卡数多者优先，便于尽早处理大卡组 |

| 放置策略 | 排序依据 |
| --- | --- |
| `balanced`（默认） | 槽位少 → 利用率低 → 空闲显存多 |
| `pack` | 槽位多 → 满足要求的空闲显存较少，集中共享任务，尽量留下整卡 |
| `spread` | 槽位少 → 空闲显存多 → 利用率低 |
| `most_free` | 空闲显存多 → 槽位少 → 利用率低 |

```sh
tetris config --set 'queue_policy="priority"' --set 'placement_policy="pack"'
tetris config --set priority_aging_seconds=300
tetris submit --priority 10 -- python3 urgent_train.py
```

改配置影响后续派发并重置观察窗口，不抢占在跑任务。`priority` 仅在优先级策略下影响排序；老化减轻饥饿，但不保证任意负载下的有界等待。`small_first`、`large_first` 和回填 FIFO 都可能让某些任务长时间等待。

## 最低优先级占卡任务

显式使用 `--filler` 登记可信占卡命令。调度器不自带 GPU 压力程序，不猜测或接管旧 keeper；程序及其训练框架依赖由用户提供，命令应使用任务内逻辑 GPU 0，并保持前台运行。

```sh
tetris config --set filler_enabled=true --set filler_utilization_threshold=30
tetris submit --filler --name filler-0 --gpus 0 -- /你的环境/bin/python /你的程序/gpu_keeper.py
tetris submit --filler --name filler-1 --gpus 1 -- /你的环境/bin/python /你的程序/gpu_keeper.py
```

- 占卡是写死的最低层级，所有队列策略下都后于普通任务，不参与优先级老化，不允许调整 `--priority`；仅支持单卡 `shared`，每卡最多一个。不能将占卡任务作为其他任务的依赖。
- 先尝试普通任务，再判断余量。训练新启动后会清空对应卡的观察历史，至少等下一次采样；配置的观察窗口和冷却同时生效，不能用启动前的旧低利用率立即补占卡。
- 占卡被调度器回收后，同一张卡需再等待 `filler_rearm_seconds`（默认 1800 秒，即 30 分钟）才允许补入新的占卡任务；该滞回独立于普通任务的 `observation_seconds`，持久化在池内，重启调度器后依然生效。可用 `tetris config --set filler_rearm_seconds=600` 调整。
- 观察窗口内 GPU 利用率必须始终**严格低于** `filler_utilization_threshold`（默认 30%）。此阈值独立于普通任务的 `utilization_threshold`；显存、每卡总任务数、独占、未知消费者和卡白名单限制仍然生效。
- 支持与共享训练共卡，但不绕过每卡上限。普通任务无法使用现有空卡时，按其申请卡数请求相关占卡任务退出；占卡造成的高利用率或低显存不会阻止发出回收请求。清理完成后重置窗口和冷却，并等待新采样，才允许派发普通任务。
- 待满足的普通任务需求覆盖范围内不新补占卡。严格 FIFO 的队首阻塞同样阻止后续占卡；这比最大化填充更保守。已运行的占卡不会仅因达到准入阈值自行退出，阈值是准入条件，不是实时利用率控制器。
- 仅调度器回收的占卡会重新排队，保留 attempt 日志，后续有余量再从头启动。人工取消、正常退出和异常失败都不会自动重启；占卡程序应可随时中断且无必须保留的计算状态。
- `filler_enabled` 默认 `false`；关闭后，正在运行的受管占卡会被异步回收，普通训练不受影响。`pause` 和 `stop` 仍只停止新派发，不停止任何在跑任务；配置关闭需由运行中的调度器处理。
- 尚未归属本池的旧训练或旧 keeper 仍是未知消费者，阻止对应卡派发。此功能不会修改旧池配置、终止旧进程或自动切换调度器。

状态表新增 `TYPE=FILLER`、`PRI=LOWEST`，JSON 保持 v2 并新增 `filler`、`priority_tier`；占卡的 `priority=null`。0.2.0 池无须 schema 升级，但不要在运行中覆盖安装或混用两个版本的调度器；停止服务后更新再重启，现有 worker 不被停止。

## 排队原因诊断

任务无法派发时，`DETAIL` 列会逐卡给出原因，例如：

```text
Waiting for 2 eligible GPUs (exclusive, 24000 MiB per GPU): GPU0: HOLD >=65%; GPU1: FULL; GPU2: LOW MEMORY 100<24000 MiB; GPU3: READY
```

`READY` 表示该卡对这个任务满足条件；候选数量不足时任务整体不占卡。

### 内置 Keeper 预设（计算利用率）

内置预设产生的是**计算负载**，目标是 `nvidia-smi` 里的 GPU-Util；显存只占很少一点（默认 512M，可用 `--mem` 调整，`--mem 0` 不占）。**需要 PyTorch（CUDA 版）**：用 cuBLAS 矩阵乘产生负载。调度器本身仍然零依赖，只有 Keeper 进程需要 torch。

```sh
tetris keepers                                    # 列出预设和默认参数
tetris config --set filler_enabled=true
tetris submit --keeper floor --per-gpu --gpus 0,1,2,3 -- --util 40
```

`--keeper` 自动按 `--filler` 提交；`--per-gpu` 必须同时明确指定 `--gpus`，为其中每张卡各登记一个；`--` 之后是预设自己的参数。`--keeper-python /path/to/python` 指定装有 PyTorch 的解释器（默认用运行 `tetris` 的那个）。提交前会校验参数取值；真实 GPU 模式下还会检查该解释器能否找到 torch，找不到就直接报错，不会入队。`--backend none` 是纯 CPU 试跑，不需要 torch。

两个独立的旋钮：

- `--util`：**有多少时间**有内核在跑，也就是 `nvidia-smi` 显示的 GPU-Util。
- `--sm`：负载占用**多少比例的 SM**（默认 25%，由矩阵大小估算，不是精确值）。数值小，则与同卡的真实训练抢得少；`burn` 预设为 100。

| 预设 | 行为 | 默认 |
| --- | --- | --- |
| `steady` | 开环：GPU-Util 固定在 `--util` 附近 | util 30，sm 25 |
| `floor` | **闭环**：读取整卡利用率，只补到 `--util` 为止；别人忙起来就自动让出，总利用率不会被它拉低 | util 40，sm 25 |
| `wave` | 在 `--lo` 到 `--hi` 之间按正弦起伏，周期 `--period` 秒 | 20–60%，600 s |
| `heartbeat` | 每 `--period` 秒的开头满速跑 `--burst` 秒，其余时间空闲 | 3 s / 30 s |
| `burn` | 持续 100% 满载、占满所有 SM | util 100，sm 100 |

其他参数：`--mem`、`--backend auto|torch|none`、`--interval`（floor 的读数间隔，默认 2 秒）、`--slice`、`--report`、`--max-seconds`（试跑用）。收到 SIGTERM 时释放显存并以 0 退出。

- **`floor` 的读数**：优先用 NVML（`libnvidia-ml`），不可用时调 `nvidia-smi`；读不到时退化为保持当前占空比，并在日志里警告。
- **利用率的含义**：GPU-Util 只表示“采样周期内是否有内核在跑”，不等于算力被充分使用。`--sm` 小的内核也可能把 GPU-Util 顶到很高。
- **与补入判断的关系**：占卡自己制造的负载会计入整卡利用率，而 `filler_utilization_threshold`（默认 30）决定能否补入新的占卡。若把 `--util` 设得高于该阈值，同卡上的其他占卡就不会再被补入；这正是每卡一个占卡的预期。
- **尚未在真实 GPU 上验证。**请先用 `--max-seconds 30` 在空闲卡上试跑，对照 `nvidia-smi` 的 GPU-Util 和 SM 占用。
- 模拟池（CPU 模拟）里 Keeper 只睡眠，不需要 torch。
- 请勿用占卡规避共享集群的使用规定。

## 单任务多卡

```sh
tetris submit --name ddp-4 --gpu-count 4 --gpus 0,1,2,3,4,5,6,7 --min-free-mb 24000 -- torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py
tetris submit --name shared-2 --gpu-count 2 --sharing shared -- python3 multi_gpu_program.py
```

- `--gpus` 仍是候选卡白名单，**不表示使用全部列出的卡**；`--gpu-count` 决定实际卡数。白名单少于申请数时提交直接报错。
- 多卡默认 `exclusive`，每张卡必须没有本池任务，运行期间阻止本池其他任务进入；单卡默认 `shared`。也可显式指定 `--sharing exclusive` 或 `shared`。
- 所有卡必须同时满足条件，才在同一数据库事务中认领整组。缺卡时零预留、零 attempt 增量，不会持有部分卡等待；硬件卡数不足的请求保持排队并显示原因。
- `--min-free-mb` 为**每卡**空闲显存门槛，不是卡组显存总和；全局显存门槛同时生效。
- 分配后按宿主 GPU 索引升序生成有序 UUID 列表，写入 `CUDA_VISIBLE_DEVICES`。例如物理卡 2、5 在任务内映射为逻辑卡 0、1；不要在训练命令里再次硬编码物理索引。
- 一个任务仍只有一个 supervisor 和一个退出码。调度器不会自动把普通训练脚本改造成 DDP；进程数由你的 `torchrun`、框架或程序自行启动，并与 `--gpu-count` 保持一致。使用 `--standalone` 避免并发任务固定 rendezvous 端口冲突。
- 取消、失联保留槽位、服务恢复和重试都作用于整组卡。训练主命令应等待所有 rank 退出，并在任一 rank 失败时返回非零。
- `exclusive` 只是**本池**的逻辑独占，不是驱动锁或 cgroup 隔离；不支持 NVLink/NUMA 拓扑感知、MIG 配额、多节点训练、弹性缩扩容或硬显存预留。

无 GPU 多卡演示：在独立模拟池设置 `tetris config --set simulate_gpu_count=4`，再提交 `--gpu-count 2` 的 CPU 命令。任务内 `TETRIS_GPU_COUNT=2`，但 `CUDA_VISIBLE_DEVICES` 仍为空。

环境覆盖只影响任务。`CUDA_VISIBLE_DEVICES`、`CUDA_DEVICE_ORDER`、`TETRIS_JOB_ID`、`TETRIS_GPU_COUNT`、`TETRIS_ATTEMPT_TOKEN`（及已弃用的 `UNIFIED_*` 别名）由调度器管理。
任务使用提交时的绝对工作目录；请使用可用的 Python 解释器路径，或在启动服务前激活训练环境。后台服务继承启动时环境；运行期间更改当前终端环境不会改变既有服务。

## 更新与迁移

### 热更新（不中断训练）

调度服务与 worker 是独立进程；`stop` 不会结束训练。更新步骤：

```sh
pipx upgrade gpu-tetris      # 或 .venv/bin/python -m pip install -U .
tetris restart               # 先预检新版本能否启动，再重启调度服务
```

- 重启只在一轮调度结束后发生；运行中的任务、日志和队列保持不变。重启期间只是暂停派发新任务。
- 旧版 worker 会继续用旧代码监督任务直到任务结束；新旧版本通过数据库和退出凭据文件通信，次要版本之间保持兼容。
- 更新前建议备份池目录。若发布说明提示“需要 `tetris upgrade`”（数据库结构变化），则需排空任务后再升级，不能热更新。
- 新版本带来新的默认配置项时，观察窗口会被重置一次（最多多等一个观察窗口），不影响运行中的任务。

### 从 unified-gpu-smi（0.3.0 之前的名字）迁移

项目已改名为 GPU Tetris：包名 `gpu-tetris`，命令 `tetris` 和 `tetris-smi`，环境变量前缀 `TETRIS_`。

1. `unified-gpu stop`（旧命令；也可直接用新版 `tetris stop`）。运行中的任务不会被停止。
2. 安装新包：`pipx install gpu-tetris`，并卸载旧包。
3. 池无需迁移：若没有设置 `--pool`/环境变量，且存在旧默认目录 `~/.local/state/unified-gpu`，新版会继续使用它；新建的池在 `~/.local/state/gpu-tetris`。
4. `tetris start`。
5. 兼容：`UNIFIED_GPU_POOL` 仍然有效（`TETRIS_POOL` 优先）；任务内同时提供 `TETRIS_*` 与旧的 `UNIFIED_*` 环境变量（旧名已弃用，后续版本将移除）。

迁移后旧版 worker 还在运行的任务，取消和清理仍能被新版识别。

## 数据位置与迁移

数据目录选择顺序：

1. `--pool /path/to/pool`（可在子命令前或后指定）
2. 环境变量 `TETRIS_POOL`
3. `$XDG_STATE_HOME/gpu-tetris`，未设置时为 `~/.local/state/gpu-tetris`

源码与队列数据完全分离。池内包含 SQLite 数据库、每次尝试日志、退出凭据与调度锁；不要放到公开仓库、NFS 或不可信共享目录。SQLite 事务保护并发提交，单实例文件锁保护派发。只读状态查询不会创建新池。

### 从 0.1.0 升级

先使用旧版暂停新派发，等待任务完成或取消并清理，再停止调度器，备份整个池；安装 0.2.0 后执行 `tetris upgrade`，最后 `tetris start`。不要在运行中的 scheduler/worker 下原地覆盖安装，也不要用旧版程序重新打开升级后的池。

升级持有本池调度锁，核验进程身份并恢复中断记录：确认已死亡且无残留进程的尝试标记失败，不自动重跑；仍有活进程时拒绝升级。失联子进程可先 `cancel JOB_ID` 再 `upgrade` 清理；正常 worker 仍存活时先等待取消完成。旧 spec 缺省为单卡，历史 `gpu` 回填为单元素 `gpu_ids`。升级可重复执行；只读 status 不迁移数据库。

JSON 输出升级为 `schema_version=2`：使用 `gpu_ids` 数组读取完整分配；单卡继续提供整数 `gpu`，多卡时 `gpu=null`，请求数量由 `gpu_count` 表示。外部消费者需适配 v2。

软件可在其他机器重新安装。**迁移源码不等于迁移运行中的进程或训练数据。** 迁移池应先等待任务结束并停止服务，备份整个池，确保 SQLite WAL 一并保留，并调整绝对工作目录及训练文件；不能跨机器恢复旧 PID。当前不提供一键路径重写或旧实验池导入。

本软件拒绝在旧版 `state.json/sources` 实验池中初始化，不接管或更改旧调度器。不同池之间没有统一派发锁；不要在同一组 GPU 上同时运行多个池，也不要与其他调度器混用。

## 恢复与取消边界

- worker 与调度服务独立；服务停止、SSH 退出或服务重启不会主动结束训练。
- 进程身份结合 Linux boot ID、PID 启动时钟校验，避免仅按复用 PID 判断。
- 用户命令通过管道握手放行，先持久化身份，再执行命令。
- 缺少退出凭据的尝试不自动重跑；worker 丢失但子进程还活着时保留槽位，可用 `cancel` 清理后再重试。
- 取消会清理进程组及带同一 attempt 标识的同用户子进程，必要时升级到 SIGKILL。默认只在清理结束后报告终态。
- **这不是 cgroup 沙箱**。任务必须保持在同一用户/PID 命名空间内，不得删除 `TETRIS_ATTEMPT_TOKEN`、清空继承环境后脱离进程组，或通过容器/特权辅助程序逃离监督。此类任务需另用 cgroup/systemd/集群调度器管理，不能保证取消清理完整。
- 默认单用户模式不做 GPU 进程归属检查，兼容自己在队列外启动的训练以及无法映射宿主 PID 的环境；仍检查整卡显存、利用率、观察窗口及本队列任务数。队列外进程不计入 `TASKS`，也不会被队列取消；单用户模式不提供多用户资源隔离。
- 需要保守阻塞非队列 GPU 进程时可执行 `tetris config --set 'ownership_mode="strict"'`。严格模式要求查询计算进程成功，并通过进程组匹配队列 supervisor；无法映射宿主 PID、脱离进程组的任务或外部进程可能显示 `UNKNOWN OWNER`。切回默认模式：`tetris config --set 'ownership_mode="single_user"'`。旧池缺少此配置时也采用单用户默认值。
- `STALE` 表示调度器未运行或心跳过期，与 `UNKNOWN OWNER` 不同；先用 `tetris start` 启动或检查 `scheduler.log`。GPU 指标为 `N/A` 或查询失败仍停止派发，避免把未知指标当成空闲。
- 不提供 exactly-once 外部副作用保证。异常任务重试前检查已写入文件、外部服务及 checkpoint。

## 安全与发布

仅用于同一用户执行可信任务，无网络监听、无远程执行 API、无鉴权边界。不要以 root 运行不可信任务，不要将数据库或日志暴露给其他用户。
任务参数、工作目录、日志可能包含敏感数据；JSON 隐藏 env 不代表整体脱敏。尽量避免通过命令行传递密钥。
默认清理终端输出中的控制字符，`logs --raw` 仅用于可信日志。

仓库中只有源码和合成测试，没有训练数据或固定机器路径。`.gitignore` 默认排除队列数据、日志、环境和构建产物。
本项目以 MIT 许可证发布，见 `LICENSE`。

## 验证状态与路线图

- 自动化测试使用 CPU 模拟池，**尚未包含真实 NVIDIA GPU 的端到端验证**；真机检查清单见 [docs/real-gpu-validation.md](docs/real-gpu-validation.md)。
- 动态共置相对独占 FIFO 的吞吐对比方案见 [docs/benchmark.md](docs/benchmark.md)（结果待补充）。
- 已知限制：占卡回收的显存预测较粗糙、无历史显存预测、持续小任务下多卡任务可能饥饿。详见 [CHANGELOG.md](CHANGELOG.md)。

## 开发与验证

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -W error::ResourceWarning -m unittest discover -s tests -v
.venv/bin/python -m pip wheel --no-deps . -w dist
```

集成测试全部使用临时 CPU 模拟池，不使用真实 GPU。测试覆盖资源策略、持久化、真实退出码、依赖、服务恢复、取消重试、worker 异常丢失、启动握手、未知消费者和查询失败。
CI 测试 Python 3.10–3.13；本地验证环境与实际执行结果以交付说明为准。
