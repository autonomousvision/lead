"""P4-A2 冒烟测试:在 Qwen3-VL-4B-Instruct 上验证"取图像 token 的 LLM hidden states"。

只验证 3 件事,不做任何训练/缓存:
  1. 模型能否在 qwenvl 环境加载(transformers>=4.57, Qwen3VL 架构)。
  2. 单次 prefill 前向(不生成)能否 output_hidden_states,并定位图像 token 位置。
  3. 图像 token 数与 image_grid_thw 是否自洽,能否 reshape 成 (h', w') 网格。

用法(在 qwenvl 环境):
  python scripts/p4/smoke_vlm_hidden.py \
      --model /mmu_mllm_hdd_3/liuzihan08/vla/models/Qwen3-VL-4B-Instruct \
      [--image some.jpg]   # 不给则用一张随机图
"""

import argparse

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/mmu_mllm_hdd_3/liuzihan08/vla/models/Qwen3-VL-4B-Instruct",
    )
    parser.add_argument("--image", default=None, help="可选:真实图片路径,不给则用随机图")
    parser.add_argument("--command", default="在路口直行,注意前方车辆", help="导航命令文本")
    args = parser.parse_args()

    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[1/4] transformers 版本检查 ...")
    import transformers

    print(f"      transformers=={transformers.__version__}")

    print(f"[2/4] 加载 processor + model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    # 图像 token id(用于在序列里定位图像 token 的位置)
    image_token_id = getattr(model.config, "image_token_id", None)
    print(f"      image_token_id = {image_token_id}")

    # 准备输入图像
    if args.image:
        image = Image.open(args.image).convert("RGB")
        print(f"[3/4] 用真实图片: {args.image}  size={image.size}")
    else:
        # 造一张 1280x720 的随机图(接近 CARLA 前视分辨率量级)
        arr = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
        image = Image.fromarray(arr)
        print(f"[3/4] 用随机图  size={image.size}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.command},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda:0")

    input_ids = inputs["input_ids"]
    grid_thw = inputs.get("image_grid_thw")
    print(f"      input_ids shape = {tuple(input_ids.shape)}")
    print(f"      image_grid_thw  = {grid_thw.tolist() if grid_thw is not None else None}")

    print(f"[4/4] 单次 prefill 前向(output_hidden_states=True, 不生成) ...")
    with torch.no_grad():
        out = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )

    hs = out.hidden_states  # tuple: (embed, layer1, ..., layerN)
    last = hs[-1]  # (1, seq_len, D)
    n_layers = len(hs)
    D = last.shape[-1]
    print(f"      hidden_states 层数(含embed) = {n_layers}")
    print(f"      last hidden shape = {tuple(last.shape)}  (D={D})")

    # 定位图像 token 位置
    img_mask = input_ids[0] == image_token_id
    n_img_tokens = int(img_mask.sum().item())
    print(f"      序列里图像 token 数 = {n_img_tokens}")

    if grid_thw is not None:
        t, h, w = grid_thw[0].tolist()
        merge = getattr(processor.image_processor, "merge_size", None)
        if merge is None:
            merge = getattr(model.config.vision_config, "spatial_merge_size", 2)
        h2, w2 = h // merge, w // merge
        print(f"      grid_thw=(t={t},h={h},w={w})  merge_size={merge}")
        print(f"      => 期望网格 (h'={h2}, w'={w2}) = {h2 * w2} tokens")
        print(f"      与实际图像 token 数是否一致: {h2 * w2 == n_img_tokens}")

        # 关键验证:抽出图像 token 的 hidden states,reshape 成网格
        img_hidden = last[0][img_mask]  # (n_img_tokens, D)
        print(f"      抽出的图像 hidden = {tuple(img_hidden.shape)}")
        if h2 * w2 == n_img_tokens:
            grid = img_hidden.reshape(h2, w2, D)
            print(f"      reshape 成网格成功: {tuple(grid.shape)}  ✓")
        else:
            print(f"      ⚠️ token 数不整除,reshape 失败,需检查 merge/grid 逻辑")

    print("\n===== A2 可行性结论 =====")
    print(f"  加载: OK")
    print(f"  取图像token hidden states: OK, 每token维度 D={D}")
    print(f"  缓存估算: {n_img_tokens} tokens x {D} x 2byte(bf16) "
          f"= {n_img_tokens * D * 2 / 1024:.1f} KB/帧")


if __name__ == "__main__":
    main()
