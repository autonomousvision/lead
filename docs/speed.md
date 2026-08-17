# Training speed

This documentation provides some information on how to configure the training pipeline for
faster research iterations.

Times are per training stage on 4 × H100 80GB. Pretrain times marked with ~ were
not measured and are estimated from the post-train time (the post-train stage
runs roughly an hour slower). Driving scores are means ± std over 3 training
seeds.

<div align="center">

| Configuration                 | Pretrain (h) | Post-train (h) | Total (h) | Bench2Drive | Longest6 v2 |
| :---------------------------- | :----------: | :------------: | :-------: | :---------: | :---------: |
| Baseline (batch 64, bilinear) |    ~20.5     |      21.5      |   ~42.0   | 93.6 ± 1.0  | 54.3 ± 4.4  |
| Logits up-sampling            |     15.7     |      16.0      |   31.7    | 93.5 ± 1.9  | 52.7 ± 3.9  |
| Logits + nearest up-sampling  |     14.3     |      16.4      |   30.7    | 93.1 ± 1.1  | 53.3 ± 3.0  |
| Batch size 96                 |    ~18.6     |      19.6      |   ~38.2   | 92.9 ± 1.9  | 51.7 ± 2.3  |
| Batch size 128                |    ~16.9     |      17.9      |   ~34.8   | 92.1 ± 1.7  | 49.4 ± 3.0  |
| Logits + nearest + batch 96   |     11.4     |      12.5      |   23.9    | 93.8 ± 0.3  | 52.3 ± 3.6  |

</div>

These numbers were obtained with the dataset from release
[v1.4.0](https://github.com/kesai-labs/lead/releases/tag/v1.4.0). Driving
scores may shift with future dataset releases; the relative speed of the
configurations should carry over.

## Compilation

`training.optimization.use_torch_compile=true` (default) compiles the model with
`torch.compile`. Compilation requires `LEAD_RUNTIME_TYPE_CHECKING=false` (the
training scripts set this), because beartype and Dynamo cannot run together.

`training.optimization.torch_compile_mode=max-autotune` spends several minutes
before the first step searching for faster kernels. Use it for full training
runs. For short runs, use `default` or disable compilation.

## Logits up-sampling

`policy.transfuser.upsample_perspective_logits=true` makes the perspective
decoders up-sample the class logits instead of the feature map. The logits have
far fewer channels, so the decoder's last block runs at lower cost. The output
resolution is unchanged.

## Nearest up-sampling

`policy.transfuser.upsample_mode=nearest` uses nearest-neighbor instead of
bilinear up-sampling. Nearest reads one input element per output element instead
of four and is cheaper to backpropagate.

## Batch size

`training.optimization.batch_size` is the global batch size, split across GPUs
(default 64). Raising it to e.g. 96 or 128 improves GPU utilization and
throughput, at the cost of more GPU memory per step and a slight reduction in
driving quality.

## Frozen backbone

`policy.transfuser.freeze_backbone=true` freezes the pretrained backbone during
post-training, so only the decoders are updated. Since the frozen backbone gets
no gradients from them, the auxiliary heads (semantic, depth, BEV semantic, box
detection) can be disabled as well. On top of the fastest configuration above
(logits + nearest + batch 96), this cuts the post-train stage to 9.5 h:

<div align="center">

| Configuration               | Pretrain (h) | Post-train (h) | Total (h) | Longest6 v2 |
| :-------------------------- | :----------: | :------------: | :-------: | :---------: |
| Logits + nearest + batch 96 |     11.4     |      12.5      |   23.9    | 52.3 ± 3.6  |
| + frozen backbone           |     11.4     |      9.5       |   20.9    | 44.1 ± 6.1  |

</div>

We do not recommend this option: the driving score drops sharply and becomes
too noisy across seeds (std 6.1 vs 3.6) to draw reliable conclusions from.

## No depth head

`policy.transfuser.use_depth=false` removes the depth prediction head and its
decoder, saving roughly 4 h per training stage over the baseline:

<div align="center">

| Configuration                 | Pretrain (h) | Post-train (h) | Total (h) | Bench2Drive | Longest6 v2 |
| :---------------------------- | :----------: | :------------: | :-------: | :---------: | :---------: |
| Baseline (batch 64, bilinear) |    ~20.5     |      21.5      |   ~42.0   | 93.6 ± 1.0  | 54.3 ± 4.4  |
| No depth head                 |     17.1     |      17.4      |   34.4    | 93.6 ± 0.9  | 50.4 ± 5.1  |

</div>
