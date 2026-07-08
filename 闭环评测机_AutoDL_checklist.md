# 闭环评测机 checklist:AutoDL 上跑 CARLA（headless Vulkan）

> 背景:CARLA 需要 Vulkan 图形渲染。**纯算力容器(恒源云/H20 等)没有任何图形库(EGL/GLX),跑不了**。
> AutoDL 的实例**带 `libEGL_nvidia`**(headless EGL),手动建 Vulkan ICD 指向它即可 → CARLA `-RenderOffScreen` 能跑。
> 参考:[AutoDL Vulkan 文档](https://www.autodl.com/docs/vulkan/);非 root / CARLA 坑见 `踩坑总结_LEAD_H20.md` §3。

---

## 0. 上机第一件事:go / no-go 判定(装任何东西之前)

```bash
ldconfig -p | grep -i libEGL_nvidia
```
- **有输出**(如 `/lib/x86_64-linux-gnu/libEGL_nvidia.so.0`)→ 继续。
- **空** → 这台也不行,换实例/镜像。

## 1. 配 Vulkan ICD(指向 libEGL,不是 GLX)

```bash
mkdir -p /etc/vulkan/icd.d
cat > /etc/vulkan/icd.d/nvidia_icd.json <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "/lib/x86_64-linux-gnu/libEGL_nvidia.so.0",
        "api_version" : "1.3.0"
    }
}
EOF
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
echo 'export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json' >> ~/.bashrc
```

## 2. 装 vulkan 工具并验证(关键关卡)

```bash
apt update && apt install -y vulkan-tools libvulkan1 libsm6 libegl1
vulkaninfo --summary
```
→ **`vulkaninfo` 认出 "NVIDIA GeForce RTX 4090" = Vulkan 通了,CARLA 90% 有戏**。认不出就先别往下走。

## 3. 代码 + 环境

```bash
# 代码:clone 你的分支(不是官方 main —— 官方没有 intent/control 代码)
export HF_ENDPOINT=https://hf-mirror.com
git clone -b visual-intent-p2 https://github.com/<你的用户名>/lead.git
cd lead
echo "export LEAD_PROJECT_ROOT=$(pwd)"                 >> ~/.bashrc
echo "export CARLA_ROOT=$(pwd)/3rd_party/CARLA_0915"   >> ~/.bashrc
echo "export HF_ENDPOINT=https://hf-mirror.com"        >> ~/.bashrc
echo "source $(pwd)/scripts/main.sh"                   >> ~/.bashrc
source ~/.bashrc

# 环境
conda create -n lead python=3.10 -y && conda activate lead
conda install -c conda-forge ffmpeg parallel tree gcc zip unzip git-lfs uv
UV_HTTP_TIMEOUT=600 uv sync --active --extra dev

# CARLA
bash scripts/setup_carla.sh
```

## 4. 起 CARLA(⚠️ 非 root + 带 VK_ICD_FILENAMES)

UE 拒绝 root。若你是 root(容器常见)建非 root 用户:
```bash
useradd -m -u 2000 -s /bin/bash carla
chown -R carla:carla $CARLA_ROOT
runuser -u carla -- bash -c "cd $LEAD_PROJECT_ROOT && \
  export HOME=/home/carla CARLA_ROOT=$CARLA_ROOT \
  VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json && \
  setsid bash scripts/start_carla.sh >/tmp/carla.log 2>&1 < /dev/null"
sleep 30
pgrep -af CarlaUE4-Linux-Shipping; ss -ltnp | grep 2000
```
→ 2000 端口监听 = CARLA 起来了。崩了看 `/tmp/carla.log` 和 `$CARLA_ROOT/CarlaUE4/Saved/Logs/CarlaUE4.log` 里的 Vulkan 报错。

## 5. 先用官方 checkpoint 验证闭环(排查机器/CARLA,与模型无关)

```bash
mkdir -p outputs/checkpoints/tfv6_resnet34
wget https://hf-mirror.com/ln2697/tfv6/resolve/main/tfv6_resnet34/config.json     -O outputs/checkpoints/tfv6_resnet34/config.json
wget https://hf-mirror.com/ln2697/tfv6/resolve/main/tfv6_resnet34/model_0030_0.pth -O outputs/checkpoints/tfv6_resnet34/model_0030_0.pth
conda activate lead
python -m lead --checkpoint outputs/checkpoints/tfv6_resnet34 \
  --routes data/benchmark_routes/bench2drive/23687.xml --bench2drive
```
→ 出 `outputs/local_evaluation/23687/`(视频 + metric_info.json)= 全链路通。

## 6. 换成你的 P2 模型跑闭环

把 H20 上的 `outputs/local_training/control_p2_full/{config.json,model_0030.pth}` 传过来,然后:
```bash
python -m lead --checkpoint <你的模型目录> \
  --routes data/benchmark_routes/bench2drive/23687.xml --bench2drive
```
之后用 `slurm/evaluation/` 或批量脚本跑完整 Bench2Drive / Longest6 出 Driving Score。

---

## 关键判定表

| 检查 | 通过标志 |
| :-- | :-- |
| 有图形库 | `ldconfig -p \| grep libEGL_nvidia` 有输出 |
| Vulkan 通 | `vulkaninfo --summary` 认出 4090 |
| CARLA 起来 | 2000 端口 LISTEN |
| 闭环通 | `outputs/local_evaluation/<route>/metric_info.json` 生成 |

任一步空/崩,先解决它再往下,别跳。

---

## 6. 批量多卡评测(220 条 Bench2Drive)的坑 —— `scripts/eval_bench2drive_local.sh`

单机多卡批量跑完整 benchmark 时踩过、已在 driver 里根治的 4 个坑:

1. **端口间距**:CARLA 每实例要 3 个连续端口(world / +1 secondary+streaming / +2)。多卡时 world-port 用 `2000+gpu*10`(间距 10),别用 `gpu*2`——会重叠导致 msgpack `bad_cast`。
2. **常驻 CARLA 会崩,且崩了污染后面所有 route**:一个 CARLA 连跑几十条 route 必然内存泄漏/段错误,之后该卡剩余 route 全 `Failed to connect to CARLA server`。→ **每条 route 起一个全新 CARLA,跑完 SIGTERM→SIGKILL 并等 world-port 释放**(官方 leaderboard 本就这么做)。
3. **Traffic Manager 端口 TIME_WAIT bind error**:CARLA 的 TM RPC socket 没设 `SO_REUSEADDR`,每条 route 复用同一 TM 端口时,上一条关闭的 socket 卡在 ~60s TIME_WAIT → 下一条 `trying to create rpc server for traffic manager; but ... bind error`。→ **每条 route 轮转 TM 端口**,每卡一段 400 宽的带(base 31000),再加**按 wall-clock 的随机起始偏移**,这样连"快速重启撞上一轮 TIME_WAIT"也避开。
4. **孤儿 python 评测进程**:`pkill` 只杀了 driver bash + CarlaUE4,**没杀正在跑的 `python -m lead`/`leaderboard_evaluator` 子进程**——它们变孤儿继续占 TM 端口 + 各吃一块 GPU 显存。→ 清理时必须一并:
   ```bash
   pkill -9 -f eval_bench2drive_local.sh
   pkill -9 -f leaderboard_evaluator
   pkill -9 -f "lead --checkpoint"
   pkill -9 -f CarlaUE4
   ```

**两个模型同时评测**:输出按模型隔离(driver 传 `--output-dir outputs/local_evaluation_<TAG>/<id>`),否则两模型评同一批 route id 会写同一目录、互相"已完成跳过"污染。用法:
```bash
nohup bash scripts/eval_bench2drive_local.sh outputs/checkpoints/my_p2   "0 1" my_p2   > /tmp/drive_my_p2.log   2>&1 &
nohup bash scripts/eval_bench2drive_local.sh outputs/checkpoints/my_lead "3 4" my_lead > /tmp/drive_my_lead.log 2>&1 &
```
跑完 `slurm/evaluation/merge_route_json.py -f outputs/b2d_<TAG>` 出 Driving Score。注意:**失败/撞车的 route 也会写 `checkpoint_endpoint.json`**(记 0 分结果),所以 skip 逻辑会把"已失败"当"已完成"跳过——真要重跑某条得先删它的输出目录。
