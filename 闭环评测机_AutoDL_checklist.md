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
