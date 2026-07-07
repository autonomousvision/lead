# LEAD 在 H20 容器节点上的踩坑总结（环境配置 → 训练正常起）

> 适用环境:快手 GPU 容器节点 `aiplatform-wlf2-ge71-92`
> 8× **NVIDIA H20**(96G),224 核,数据盘为**共享 CephFS** `/mmu_mllm_hdd_3`,容器内 root。
> 仓库:`https://github.com/kesai-labs/lead`,路径 `/mmu_mllm_hdd_3/liuzihan08/vla/lead`。
>
> 一句话结论:**训练这条链路能在 H20 上跑通**(本文记录了所有坑);**CARLA 闭环评测/数据采集这条链路在本节点跑不了**(GPU 图形栈缺失,见 §3)。

---

## 0. 环境变量(每次新 shell 都要,建议写进 ~/.bashrc)

仓库的 `scripts/main.sh` **只设了 NavSim/Py123D 的变量,没设 `CARLA_ROOT`**,需要自己补:

```bash
export LEAD_PROJECT_ROOT=/mmu_mllm_hdd_3/liuzihan08/vla/lead
export CARLA_ROOT=$LEAD_PROJECT_ROOT/3rd_party/CARLA_0915
source $LEAD_PROJECT_ROOT/scripts/main.sh
conda activate lead     # unzip / parallel / 训练依赖都在这个 env
```

---

## 1. 依赖安装:`uv sync` 网络超时

**现象**:`uv sync --active --extra dev` 下载 `ray` 等大 wheel 时 `Failed to download ... network timeout (UV_HTTP_TIMEOUT: 30s)`。

**解法**:加大超时(可配国内镜像),`uv sync` 断点可续:
```bash
UV_HTTP_TIMEOUT=600 uv sync --active --extra dev
# 网差再加镜像:
UV_HTTP_TIMEOUT=600 UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync --active --extra dev
```

---

## 2. CARLA 安装:`setup_carla.sh` 报 `tar: Unexpected inconsistency when making directory`

**现象**:`bash scripts/setup_carla.sh` 解压时一堆 `Unexpected inconsistency` 最后 `Exiting with failure`。

**根因**:CARLA **早就解压好了**,在**已存在的目录上重复解压**,CephFS 上 `tar` 往已存在目录 `mkdir` 会冒这个错。**不是磁盘满、不是包坏**。

**解法**:别重跑 `setup_carla.sh`。确认已装好即可(关键看 Town12/Town13 在不在):
```bash
ls 3rd_party/CARLA_0915/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping   # 主二进制
ls -d 3rd_party/CARLA_0915/CarlaUE4/Content/Carla/Maps/Town12* Town13*    # AdditionalMaps 导入成功的标志
```
（可删两个大 tar.gz 省 ~15G:`CARLA_0915/CARLA_0915.tar.gz`、`CARLA_0915/Import/AdditionalMaps_0.9.15.tar.gz`）

---

## 3. CARLA 启动:两个坑 + 一个硬限制 ⚠️

### 3a. `CARLA_ROOT` 未设
`scripts/start_carla.sh` 用 `$CARLA_ROOT/CarlaUE4.sh`,没设就变成 `/CarlaUE4.sh: No such file`。→ 见 §0 设 `CARLA_ROOT`。

### 3b. `Refusing to run with the root privileges.`
CARLA 底层 Unreal Engine **拒绝以 root 运行**。解法:建非 root 用户跑 CARLA(它只监听 2000 端口,评测进程仍可用 root 连):
```bash
useradd -m -u 2000 -s /bin/bash carla
chown -R carla:carla $CARLA_ROOT          # 让 carla 能写 Saved/Logs/Intermediate 等(否则崩在早期)
runuser -u carla -- bash -c "cd $LEAD_PROJECT_ROOT && export HOME=/home/carla CARLA_ROOT=$CARLA_ROOT && setsid bash scripts/start_carla.sh >/tmp/carla.log 2>&1 < /dev/null"
```
注意:Shipping 版**默认不输出日志**,排查要加 `-log`;`xdg-user-dir: Permission denied` 是无害警告。

### 3c. ⚠️ 硬限制:本节点缺 GPU 图形/Vulkan 栈,CARLA 跑不起来
即使解决了 root 问题,CARLA 仍会**早期崩溃**。根因(诊断结论):
- 容器以 `NVIDIA_DRIVER_CAPABILITIES=compute,utility` 启动,**只挂了计算驱动,没挂 graphics**
- 磁盘上**没有 NVIDIA 图形库**(`libGLX_nvidia` 等)、**没有 Vulkan ICD**(`/usr/share/vulkan/icd.d/` 空)
- H20 属数据中心卡,README 自己也把同类 A100 的 **Inference 标成 ✗**

**结论**:**CARLA 闭环评测(Bench2Drive/Longest6/Town13)和 expert 数据采集(2.2 Option B)在本节点无法进行。** 解决需要平台侧用 `NVIDIA_DRIVER_CAPABILITIES=all`(含 graphics,display)重启容器,或换 workstation 卡(L40S/RTX 等)。**训练不受影响**(训练只用 compute)。

---

## 4. 数据下载与解压(README 2.2 Option A)

数据:HuggingFace `ln2697/lead`,**9715 个 zip,约 502G**,解压到 `data/carla_leaderboard2/data/`。

### 4a. 并行解压的 mkdir 竞争
`unzip_routes.sh` 用 `parallel -P 64`,多个进程同时建同一个**类别目录**时在 CephFS 上报 `checkdir error: cannot create ... File exists`,会**跳过**该 zip。
**解法**:先串行预建所有类别目录,再解压(消除竞争):
```bash
for d in data/carla_leaderboard2/zip/*/; do mkdir -p "data/carla_leaderboard2/data/$(basename "$d")"; done
bash scripts/unzip_routes.sh
```

### 4b. 假性「缺 998 个 route」——别用 results.json 判完整性
有 **997 个 route 的 zip 本身不含 `results.json`**(数据集特性,多为较大的 route),加载器 `route_filtering.route_failed()` 会**优雅跳过**它们(打日志,不报错)。
**教训**:判断解压是否完整要数**二级目录数**(= route 数),不要数 `results.json`:
```bash
find data/carla_leaderboard2/data -name '*.zip' ... # 错
find data/carla_leaderboard2/data -maxdepth 2 -mindepth 2 -type d | wc -l   # 对,应 = 9715
```
实际可训练 route ≈ **8718**(9715 − 997),属正常。

### 4c. 别反复 kill 并行 unzip
在 CephFS 上大规模并行 `unzip` 容易**卡死/留下僵尸进程**,而且**被 kill 时正在写的文件会被截断**(→ §5 的坏 laz 来源)。只解压缺失项时用 Python `zipfile` 或低并行更稳。

---

## 5. build_cache:坏 laz 导致崩溃 + cache 位置

**现象**:`python scripts/build_cache.py` 跑到某帧崩 `lazrs.LazrsError: IoError: failed to fill whole buffer`。
**根因**:个别 LiDAR `.laz` 文件**被截断**(来自 §4c 被 kill 的并行解压),不是数据集本身坏。

**解法(已打补丁)**:`carla_dataset.py` 的 `__getitem__` 加了 try/except,读传感器失败就**记 `[corrupt-skip]` 日志 + 随机换样本**,build_cache 和训练都不再被单个坏帧搞崩。

**cache 信息**:
- 位置:**`data/carla_leaderboard2/cache/`**,约 **83G**
- 结构:`cache/<场景>/<route>/<参数指纹串>/{normal,perturbated}/<frame>.pkl`(每样本一个 lzma 压缩 pkl)
- **建一次,pretrain/posttrain 共用**,放共享盘别的机器也能读
- **改了图像尺寸/BEV范围/depth等预处理配置**才需重建(指纹会变)
- 中途断了**别直接重跑 `build_cache.py`**(它 `force_rebuild=True` 会全重做);要么直接开训练(`force_rebuild=False` 会懒加载补缺),要么把 build_cache 改成 `force_rebuild_data_cache=False` 续建

---

## 6. 训练启动:四连坑

### 6a. `torchrun ... can't open file '.../python3'`
`pretrain_ddp.sh`/`posttrain_ddp.sh` 末尾是 `python3 lead/training/train.py`,但 torchrun 默认会自动加 python 前缀。**已补 `--no-python`**。

### 6b. `RuntimeError: context has already been set`
`training_utils.set_start_method()` 里 `mp.set_start_method("fork")` 没加 force。torchrun worker 里上下文已设过 → 报错。**已改 `force=True`**。

### 6c. `KeyError: 'USER'`
`config_training.py` 里 `f"/tmp/{os.environ['USER']}"`,容器无 `USER` 变量 → 崩。**已改 `os.environ.get('USER','lead')`**(session cache 落 `/tmp/lead`)。

### 6d. wandb
后台跑没法交互登录会崩 `No API key configured`。要用 wandb:`wandb login <KEY>`(存 ~/.netrc,一次即可);不想用:`export WANDB_MODE=offline`(本地)或 `disabled`。
- 项目页:https://wandb.ai/models-kuaishou/lead_pretrain
- 拿当前 run 链接:`grep -a "View run at" pretrain.log | tail -1`

### 6e. cuDNN SDPA 警告被当成错误 ⚠️(H20 特有)
**现象**:训练第 1 步 `backward()` 崩,只打印 `UserWarning: cuDNN SDPA backward got grad_output.strides() != output.strides()...`。
**根因**:`train.py` 顶部 `warnings.filterwarnings("error")` 把**所有警告升级成异常**;H20 上 cuDNN 注意力反向会触发这条**无害警告**(它会自动 materialize 修正),于是被升级成错误炸掉。作者的卡不触发,所以没事。
**解法(已打补丁)**:在 `train.py` 忽略列表里加了这条警告。**以后若再撞到别的无害 UserWarning 被升级成错误,同样加进忽略列表即可。**

---

## 7. 性能调优(GPU 喂不饱)

### 7a. dataloader worker 太少(主因)
`num_workers = int(assigned_cpu_cores / 卡数) * 1`,而 `assigned_cpu_cores` 无 SLURM 时**默认 8** → 4 卡时**每卡仅 2 个 worker**,CephFS 上严重喂不饱 GPU(显存空、利用率低)。
**解法(已打补丁)**:`config_training.py` 把默认值改成 `min(os.cpu_count(), 96)` → 每卡 24 worker。效果:**~2 it/s → ~4 it/s(翻倍)**。
- ⚠️ worker 不是越多越好:太多(如 96 个进程)在**共享 CephFS / 共享节点**上会互相抢 IO,反而变慢、且 it/s 波动大。按实际 util 调,够用即可。

### 7b. batch / epoch(谨慎改)
- **当前**:全局 `batch_size=64`,4 卡每卡 16;`epochs=31`(`carla_leaderboard_mode` 固定);每 epoch 17040 步。
- **加 batch**:数据 IO 瓶颈时**不会加速**(等数据更久);且偏离 paper 配方(`lr=3e-4`/31ep 是按 bsz64 调的),要复现 95 分就别动,真要加得相应缩放 LR。必须能被卡数整除,经 `LEAD_TRAINING_CONFIG="batch_size=128"` 覆盖。
- **减 epoch**:线性加速但**欠训掉分**,只适合验流程。
- 指定卡:`CUDA_VISIBLE_DEVICES=4,5,6,7`(物理序号,逗号无空格)。

---

## 8. 本次对仓库的代码改动清单(`git diff`,均为本节点环境特有的小补丁,可 `git checkout` 还原)

| 文件 | 改动 | 为什么 |
| :-- | :-- | :-- |
| `scripts/pretrain_ddp.sh` | torchrun 加 `--no-python` | §6a |
| `scripts/posttrain_ddp.sh` | torchrun 加 `--no-python` | §6a |
| `lead/training/training_utils.py` | `set_start_method(..., force=True)` | §6b |
| `lead/training/config_training.py` | `USER` 默认值;`assigned_cpu_cores` 默认改 `min(cpu_count,96)` | §6c / §7a |
| `lead/training/train.py` | 忽略 cuDNN SDPA backward 警告 | §6e |
| `lead/data_loader/carla_dataset.py` | 传感器读取容错跳过坏帧 | §5 |

---

## 9. 命令速查

```bash
# 环境
export LEAD_PROJECT_ROOT=/mmu_mllm_hdd_3/liuzihan08/vla/lead
export CARLA_ROOT=$LEAD_PROJECT_ROOT/3rd_party/CARLA_0915
source $LEAD_PROJECT_ROOT/scripts/main.sh && conda activate lead

# 建 cache(一次性,~83G,长任务)
python scripts/build_cache.py

# 四卡 pretrain(31 epoch,~2.4h/epoch)
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash scripts/pretrain_ddp.sh > pretrain.log 2>&1 &

# 监控
tail -f pretrain.log
tr '\r' '\n' < pretrain.log | grep -aoE "[0-9]+/17040 \[[^]]*\]" | tail -3   # 速度
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -l 2   # 利用率

# 停训练
pkill -f torchrun; pkill -f "train.py"

# posttrain(用 pretrain 第30个epoch权重)
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash scripts/posttrain_ddp.sh > posttrain.log 2>&1 &
```

---

## 10. 仍未解决 / 待办

- **CARLA 闭环评测 & expert 采数据**:受限于本节点 GPU 图形栈缺失(§3c),需平台侧开 `NVIDIA_DRIVER_CAPABILITIES=all` 或换图形卡。
- **wandb API key**:之前在终端明文贴过,建议吊销重建。
- **坏 laz 帧**:已用容错跳过;若数量大可考虑用 Python zipfile 针对性重解压受影响 route。
