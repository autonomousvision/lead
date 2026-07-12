"""P4 step 1 (lead env): build a manifest of front-camera frames for VLM feature extraction.

Reuses CARLAData only to get the EXACT training frame list (ds.images). For each (subsampled)
frame it records the raw strip-image path + the front-camera crop fraction, keyed by
(scenario, route, frame) -- the same key the training dataloader derives, so the VLM feature
cache lines up 1:1 with training samples.

By default it writes ONLY a manifest (no image files) -- fast, no small-file explosion on the
shared FS. The qwenvl-side extractor reads `src` directly and crops the front third itself.
Use --save-png to also dump cropped front PNGs for eyeballing the crop.

Two-env bridge: runs in `lead` env (needs the lead dataset). Manifest is consumed by
scripts/p4/extract_vlm_features.py in the `qwenvl` env.

Run (lead env, CPU is enough):
    # smoke: verify crop is correct (writes 20 PNGs to eyeball)
    CUDA_VISIBLE_DEVICES="" python scripts/p4/dump_front_images.py \
        --manifest data/p4/manifest_smoke.jsonl --n 20 --save-png --out data/p4/front_smoke
    # full P4 subsample: 1 of every 10 frames (~97k), manifest only
    CUDA_VISIBLE_DEVICES="" python scripts/p4/dump_front_images.py \
        --manifest data/p4/manifest.jsonl --stride 10
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/p4/manifest.jsonl")
    ap.add_argument("--n", type=int, default=0, help="cap on #frames to output; 0 = no cap")
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="subsample: take 1 of every `stride` frames (adjacent frames are near-duplicate; "
        "stride>1 decorrelates and cuts cost). e.g. --stride 10 over 968k -> ~97k frames",
    )
    ap.add_argument(
        "--front-frac",
        type=float,
        nargs=2,
        default=(1.0 / 3.0, 2.0 / 3.0),
        help="horizontal [start,end] fraction of the front camera within the 3-cam strip",
    )
    ap.add_argument("--save-png", action="store_true", help="also dump cropped front PNGs (eyeball)")
    ap.add_argument("--out", default="data/p4/front", help="dir for PNGs when --save-png")
    ap.add_argument("--densify-junction-dist", type=float, default=0.0,
                    help="P5: frames within this many metres of a junction are sampled at "
                         "--stride-near instead of --stride (0 = uniform stride, P4 behaviour)")
    ap.add_argument("--stride-near", type=int, default=2, help="finer stride near junctions")
    args = ap.parse_args()

    from lead.data_loader.carla_dataset import CARLAData
    from lead.training.config_training import TrainingConfig
    from lead.common import common_utils

    # CARLA high-level navigation command (integer -> word) for conditioning the VLM.
    CMD = {1: "LEFT", 2: "RIGHT", 3: "STRAIGHT", 4: "LANEFOLLOW",
           5: "CHANGELANELEFT", 6: "CHANGELANERIGHT"}

    mdir = os.path.dirname(args.manifest)
    if mdir:
        os.makedirs(mdir, exist_ok=True)
    if args.save_png:
        os.makedirs(args.out, exist_ok=True)
        import cv2

    config = TrainingConfig()
    config.use_planning_decoder = True

    ds = CARLAData(root=config.carla_data, config=config)
    paths = ds.images
    metas = ds.metas
    cmd_key = f"next_commands_{config.tp_pop_distance}"
    n_total = len(paths)
    base_indices = list(range(0, n_total, max(1, args.stride)))
    if args.n > 0:
        base_indices = base_indices[: args.n]
    f0, f1 = args.front_frac
    print(f"dataset frames: {n_total}, base stride: {args.stride}")
    print(f"front-cam horizontal fraction: {args.front_frac}  save_png={args.save_png}")
    print(f"command key: {cmd_key}  densify<{args.densify_junction_dist}m @stride {args.stride_near}")

    def read_meta(idx: int) -> tuple[str, float]:
        """Read the nav command word and distance-to-junction for a frame."""
        meta = common_utils.read_pickle(str(metas[idx], encoding="utf-8"))
        cmd = CMD.get(int(meta[cmd_key][0]), "LANEFOLLOW")
        dist = float(meta.get("dist_to_junction", 1e9))
        return cmd, dist

    # Read meta only for base (strided) frames; densify around near-junction ones by
    # filling their neighbours at the finer stride (extras inherit the nearby command,
    # which is unused by the command-agnostic P5 prompt anyway).
    index_cmd: dict[int, str] = {}
    for pos, i in enumerate(base_indices):
        try:
            cmd, dist = read_meta(i)
        except Exception as exc:  # noqa: BLE001
            if pos == 0:
                print(f"  [warn] meta read failed ({exc}); defaulting LANEFOLLOW / far")
            cmd, dist = "LANEFOLLOW", 1e9
        index_cmd.setdefault(i, cmd)
        if args.densify_junction_dist > 0 and dist < args.densify_junction_dist:
            for j in range(max(0, i - args.stride), min(n_total, i + args.stride + 1),
                           max(1, args.stride_near)):
                index_cmd.setdefault(j, cmd)

    final_indices = sorted(index_cmd)
    print(f"dumping: {len(final_indices)} frames "
          f"({len(base_indices)} base + {len(final_indices) - len(base_indices)} junction-densified)")

    written = skipped = bad = 0
    seen_keys = set()
    with open(args.manifest, "w") as mf:
        for i in final_indices:
            p = str(paths[i], encoding="utf-8")
            parts = p.split("/")
            scenario, route, frame = parts[-4], parts[-3], parts[-1].split(".")[0]
            key = f"{scenario}__{route}__{frame}"
            if key in seen_keys:
                continue
            seen_keys.add(key)

            entry = {
                "key": key,
                "scenario": scenario,
                "route": route,
                "frame": frame,
                "src": p,
                "front_frac": [f0, f1],
                "command": index_cmd[i],
            }

            if args.save_png:
                out_png = os.path.join(args.out, key + ".png")
                if not os.path.exists(out_png):
                    img = cv2.imread(p)  # BGR, expect (384, 1152, 3)
                    if img is None:
                        bad += 1
                        if bad <= 5:
                            print(f"  [bad] cannot read {p}")
                        continue
                    W = img.shape[1]
                    cv2.imwrite(out_png, img[:, int(W * f0):int(W * f1)])
                entry["png"] = out_png

            mf.write(json.dumps(entry) + "\n")
            written += 1
            if written <= 3 or written % 20000 == 0:
                print(f"[{written}] {key}  src={p}")

    print(f"done: written={written} bad={bad}")
    print(f"manifest -> {args.manifest}  ({written} frames)")


if __name__ == "__main__":
    main()
