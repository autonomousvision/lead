#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import carla


def normalize_map_name(name: str) -> str:
    return name.split("/")[-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Spawn all blueprints matching a CARLA filter, measure actor bounding-box "
            "extents, and save results to a file."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--timeout", type=float, default=10.0, help="Client timeout (s)")
    parser.add_argument(
        "--town",
        default=None,
        help="Target town name (e.g. Town03). If set and different, load this town first.",
    )
    parser.add_argument(
        "--blueprint-filter",
        required=True,
        help="Blueprint filter to resolve (e.g. vehicle.*, walker.pedestrian.*, static.prop.*).",
    )
    parser.add_argument(
        "--output",
        default="out/blueprint_extents.json",
        help="Output file path (.json or .csv).",
    )
    parser.add_argument("--spawn-x", type=float, default=0.0, help="Fixed spawn x")
    parser.add_argument("--spawn-y", type=float, default=0.0, help="Fixed spawn y")
    parser.add_argument("--spawn-z", type=float, default=100.0, help="Fixed spawn z")
    parser.add_argument("--spawn-pitch", type=float, default=0.0, help="Fixed spawn pitch")
    parser.add_argument("--spawn-yaw", type=float, default=0.0, help="Fixed spawn yaw")
    parser.add_argument("--spawn-roll", type=float, default=0.0, help="Fixed spawn roll")
    parser.add_argument(
        "--settle-ticks",
        type=int,
        default=1,
        help="World ticks to wait after spawning before measuring extents.",
    )
    return parser


def write_json(output_path: Path, rows: list[dict], metadata: dict) -> None:
    payload = {
        "metadata": metadata,
        "count": len(rows),
        "actors": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "blueprint_id",
        "spawned",
        "error",
        "actor_id",
        "extent_x",
        "extent_y",
        "extent_z",
        "size_x",
        "size_y",
        "size_z",
        "spawn_x",
        "spawn_y",
        "spawn_z",
        "spawn_pitch",
        "spawn_yaw",
        "spawn_roll",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = build_parser().parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = client.get_world()

    if args.town:
        target_town = normalize_map_name(args.town)
        current_town = normalize_map_name(world.get_map().name)
        if target_town != current_town:
            print(f"Loading town: {target_town} (current: {current_town})")
            world = client.load_world(target_town)

    spawn_transform = carla.Transform(
        location=carla.Location(x=args.spawn_x, y=args.spawn_y, z=args.spawn_z),
        rotation=carla.Rotation(
            pitch=args.spawn_pitch,
            yaw=args.spawn_yaw,
            roll=args.spawn_roll,
        ),
    )

    bp_lib = world.get_blueprint_library()
    blueprints = sorted(bp_lib.filter(args.blueprint_filter), key=lambda bp: bp.id)
    if not blueprints:
        raise RuntimeError(f"No blueprint matches '{args.blueprint_filter}'.")

    print(f"Resolved {len(blueprints)} blueprint(s) for filter '{args.blueprint_filter}'.")

    rows: list[dict] = []

    for i, bp in enumerate(blueprints, start=1):
        actor = None
        spawn_error = ""
        actor = world.try_spawn_actor(bp, spawn_transform)

        if actor is None:
            spawn_error = "failed_to_spawn_at_fixed_transform"
            print(f"[{i}/{len(blueprints)}] {bp.id}: {spawn_error}")
            rows.append(
                {
                    "blueprint_id": bp.id,
                    "spawned": False,
                    "error": spawn_error,
                    "actor_id": "",
                    "extent_x": "",
                    "extent_y": "",
                    "extent_z": "",
                    "size_x": "",
                    "size_y": "",
                    "size_z": "",
                    "spawn_x": "",
                    "spawn_y": "",
                    "spawn_z": "",
                    "spawn_pitch": "",
                    "spawn_yaw": "",
                    "spawn_roll": "",
                }
            )
            continue

        try:
            for _ in range(max(0, args.settle_ticks)):
                world.wait_for_tick()

            extent = actor.bounding_box.extent
            size_x = extent.x * 2.0
            size_y = extent.y * 2.0
            size_z = extent.z * 2.0

            row = {
                "blueprint_id": bp.id,
                "spawned": True,
                "error": "",
                "actor_id": actor.id,
                "extent_x": extent.x,
                "extent_y": extent.y,
                "extent_z": extent.z,
                "size_x": size_x,
                "size_y": size_y,
                "size_z": size_z,
                "spawn_x": spawn_transform.location.x,
                "spawn_y": spawn_transform.location.y,
                "spawn_z": spawn_transform.location.z,
                "spawn_pitch": spawn_transform.rotation.pitch,
                "spawn_yaw": spawn_transform.rotation.yaw,
                "spawn_roll": spawn_transform.rotation.roll,
            }
            rows.append(row)

            print(
                f"[{i}/{len(blueprints)}] {bp.id}: "
                f"extent=({extent.x:.3f}, {extent.y:.3f}, {extent.z:.3f}) "
                f"size=({size_x:.3f}, {size_y:.3f}, {size_z:.3f})"
            )
        finally:
            actor.destroy()

    output_path = Path(args.output)
    suffix = output_path.suffix.lower()

    metadata = {
        "host": args.host,
        "port": args.port,
        "town": normalize_map_name(world.get_map().name),
        "blueprint_filter": args.blueprint_filter,
        "spawn_transform": {
            "x": args.spawn_x,
            "y": args.spawn_y,
            "z": args.spawn_z,
            "pitch": args.spawn_pitch,
            "yaw": args.spawn_yaw,
            "roll": args.spawn_roll,
        },
        "settle_ticks": args.settle_ticks,
    }

    if suffix == ".csv":
        write_csv(output_path, rows)
    else:
        write_json(output_path, rows, metadata)

    success_count = sum(1 for row in rows if row["spawned"])
    print(
        f"Saved {len(rows)} rows to {output_path} "
        f"(spawned={success_count}, failed={len(rows) - success_count})."
    )


if __name__ == "__main__":
    main()
