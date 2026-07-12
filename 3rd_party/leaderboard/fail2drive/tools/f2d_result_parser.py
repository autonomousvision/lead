#!/usr/bin/env python3
"""Parse Fail2Drive result folders into the dataframe used by the notebook."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

IGNORED_INFRACTIONS_FOR_SUCCESS = {"min_speed_infractions", "outside_route_lanes"}
ROUTE_FILE_REGEX = re.compile(r"^(Base|Generalization)_(.+)_(\d{4})\.xml$")
RESULT_FILE_REGEX = re.compile(r"^(\d{4})_res\.json$")
# V2 format: Split_Scenario_Index.json (e.g. Base_BadParking_0.json)
RESULT_FILE_REGEX_V2 = re.compile(r"^(Base|Generalization)_(.+)_(\d+)\.json$")
# Map result-file scenario names to canonical names (merge variants into one scenario).
SCENARIO_ALIASES: dict[str, str] = {
    "BadParkingTwoWays": "BadParking",
}
SCENARIO_CLASSES = {
    "Robustness": [
        "RightConstruction",
        "OppositeConstruction",
        "ImageOnObject",
        "PassableObstacles",
        "PedestrianCrowd",
    ],
    "Visual-lat": [
        "BadParking",
        "ConstructionPermutations",
        "CustomObstacles",
    ],
    "Visual-lon": [
        "ObscuredStop",
        "HardBrake",
        "RightOfWay",
        "Animals",
        "PedestrianOtherBlocker",
    ],
    "Behavior": [
        "ConstructionPedestrian",
        "Wall",
        "FullyBlocked",
        "PedestriansOnRoad",
    ],
}
SCENARIO_PAPER_ORDER = [
    "Wall",
    "FullyBlocked",
    "PedestriansOnRoad",
    "ConstructionPedestrian",
    "BadParking",
    "ConstructionPermutations",
    "CustomObstacles",
    "ObscuredStop",
    "HardBrake",
    "RightOfWay",
    "Animals",
    "PedestrianOtherBlocker",
    "RightConstruction",
    "OppositeConstruction",
    "ImageOnObject",
    "PassableObstacles",
    "PedestrianCrowd",
]
CLASS_PAPER_ORDER = ["Visual-lon", "Visual-lat", "Behavior", "Robustness"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Required:
    parser.add_argument("results_root", help="Method results root, e.g. /path/to/method_eval")
    # Optional:
    parser.add_argument(
        "--method",
        default="Method_TODO",
        help="Method name written into the output dataframe (default: Method_TODO).",
    )
    parser.add_argument(
        "--route-dir",
        default=str(Path(__file__).resolve().parents[1] / "fail2drive_split"),
        help="Directory containing <Split>_<Scenario>_<id>.xml route files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any seed is missing expected results or contains unmapped result files.",
    )
    parser.add_argument(
        "--b2d-score",
        type=float,
        default=None,
        help="B2D score used as the second column in the LaTeX main-table row.",
    )
    return parser.parse_args()


def build_route_mapping(route_dir: Path) -> dict[int, tuple[str, str]]:
    mapping: dict[int, tuple[str, str]] = {}
    for xml_path in sorted(route_dir.glob("*.xml")):
        match = ROUTE_FILE_REGEX.match(xml_path.name)
        if not match:
            continue
        split, scenario, route_idx = match.groups()
        mapping[int(route_idx)] = (split, scenario.replace("TwoWays", ""))

    if not mapping:
        raise RuntimeError(f"No route XML mapping found in {route_dir}")
    return mapping


def iter_seed_dirs(results_root: Path) -> Iterable[Path]:
    for path in sorted(results_root.iterdir()):
        if path.is_dir():
            yield path


def resolve_result_dir(seed_dir: Path) -> Optional[Path]:
    for name in ("res", "results", "eval"):
        candidate = seed_dir / name
        if candidate.exists() and candidate.is_dir():
            return candidate
    # V2 layout: seed_dir contains a timestamp subdir which holds results/ or eval/.
    for child in sorted(seed_dir.iterdir()):
        if not child.is_dir():
            continue
        for name in ("res", "results", "eval"):
            candidate = child / name
            if candidate.exists() and candidate.is_dir():
                return candidate
    return None


def route_idx_from_filename(json_path: Path) -> Optional[int]:
    match = RESULT_FILE_REGEX.match(json_path.name)
    if not match:
        return None
    return int(match.group(1))


def is_success(record: dict) -> bool:
    infractions = record.get("infractions", {}) or {}
    for infraction_name, entries in infractions.items():
        if infraction_name in IGNORED_INFRACTIONS_FOR_SUCCESS:
            continue
        if entries:
            return False
    return True


def load_rows(
    method: str,
    results_root: Path,
    route_map: dict[int, tuple[str, str]],
 ) -> tuple[pd.DataFrame, list[str], bool]:
    rows: list[dict] = []
    warnings: list[str] = []
    strict_failure = False
    expected_route_ids = set(route_map.keys())

    for seed_dir in iter_seed_dirs(results_root):
        result_dir = resolve_result_dir(seed_dir)
        if result_dir is None:
            warning = f"WARNING: seed {seed_dir.name} has no res/ or results/ directory."
            warnings.append(warning)
            strict_failure = True
            continue

        json_files = sorted(result_dir.glob("*.json"))
        seen_route_ids: set[int] = set()
        unmapped_files: list[str] = []

        for json_path in json_files:
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                warnings.append(f"WARNING: failed to parse {json_path}")
                strict_failure = True
                continue

            records = data.get("_checkpoint", {}).get("records") or []
            if not records:
                warnings.append(f"WARNING: no records found in {json_path}")
                strict_failure = True
                continue

            record = records[0]
            if record.get("status") == "Failed - Agent couldn't be set up":
                warnings.append(f"WARNING: agent setup failed in {json_path}")
                strict_failure = True
                continue

            # Try v1 format (NNNN_res.json + route map) then v2 (Split_Scenario_N.json).
            route_idx = route_idx_from_filename(json_path)
            if route_idx is not None and route_idx in route_map:
                seen_route_ids.add(route_idx)
                split, scenario = route_map[route_idx]
            else:
                v2_match = RESULT_FILE_REGEX_V2.match(json_path.name)
                if v2_match is None:
                    if json_path.name != "merged.json":
                        unmapped_files.append(json_path.name)
                        strict_failure = True
                    continue
                split = v2_match.group(1)
                scenario = v2_match.group(2)
                scenario = SCENARIO_ALIASES.get(scenario, scenario)
                route_idx = int(v2_match.group(3))
            scores = record.get("scores", {}) or {}
            ds = scores.get("score_composed")
            rc = scores.get("score_route")
            if ds is None or rc is None:
                warnings.append(f"WARNING: missing scores in {json_path}")
                strict_failure = True
                continue

            rows.append(
                {
                    "Method": method,
                    "Seed": seed_dir.name,
                    "Split": split,
                    "Scenario": scenario,
                    "Id": f"{route_idx:04d}",
                    "DS": float(ds),
                    "RC": float(rc),
                    "Success": bool(is_success(record)),
                }
            )

        missing_ids = sorted(expected_route_ids.difference(seen_route_ids))
        if missing_ids:
            strict_failure = True
            preview = ", ".join(f"{route_id:04d}" for route_id in missing_ids[:10])
            suffix = " ..." if len(missing_ids) > 10 else ""
            warnings.append(
                f"WARNING: seed {seed_dir.name} is missing {len(missing_ids)} results: {preview}{suffix}"
            )
        if unmapped_files:
            preview = ", ".join(unmapped_files[:10])
            suffix = " ..." if len(unmapped_files) > 10 else ""
            warnings.append(
                f"WARNING: seed {seed_dir.name} has {len(unmapped_files)} unmapped result files: {preview}{suffix}"
            )

    return pd.DataFrame(rows), warnings, strict_failure


def build_main_table(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["Method", "Split", "DS", "Success"])

    table = rows.groupby(["Method", "Split"], as_index=False)[["DS", "Success"]].mean()
    table["Success"] *= 100.0
    return table.round({"DS": 1, "Success": 1})


def harmonic_mean(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b):
        return float("nan")
    if a < 0 or b < 0:
        return float("nan")
    if a == 0 or b == 0:
        return 0.0
    return 2.0 / ((1.0 / a) + (1.0 / b))


def pct_change(new: float, old: float) -> float:
    if old == 0 or math.isnan(new) or math.isnan(old):
        return float("nan")
    return 100.0 * new / old - 100.0


def build_summary_rows(table: pd.DataFrame, b2d_score: Optional[float]) -> list[dict]:
    rows: list[dict] = []
    for method in sorted(table["Method"].unique().tolist()):
        method_table = table[table["Method"] == method].set_index("Split")
        if "Base" not in method_table.index or "Generalization" not in method_table.index:
            continue

        base_ds = float(method_table.loc["Base", "DS"])
        base_success = float(method_table.loc["Base", "Success"])
        gen_ds = float(method_table.loc["Generalization", "DS"])
        gen_success = float(method_table.loc["Generalization", "Success"])
        base_hm = harmonic_mean(base_ds, base_success)
        gen_hm = harmonic_mean(gen_ds, gen_success)

        rows.append(
            {
                "Method": method,
                "B2D": "TODO" if b2d_score is None else f"{b2d_score:.1f}",
                "Base DS": f"{base_ds:.1f}",
                "Base Success": f"{base_success:.1f}",
                "Base HM": f"{base_hm:.1f}",
                "Gen DS": f"{gen_ds:.1f} ({pct_change(gen_ds, base_ds):.1f}%)",
                "Gen Success": f"{gen_success:.1f} ({pct_change(gen_success, base_success):.1f}%)",
                "Gen HM": f"{gen_hm:.1f} ({pct_change(gen_hm, base_hm):.1f}%)",
            }
        )
    return rows


def build_scenario_hm_overview(rows: pd.DataFrame) -> list[dict]:
    if rows.empty:
        return []

    scenario_scores = (
        rows.groupby(["Method", "Split", "Scenario"], as_index=False)[["DS", "Success"]].mean()
    )
    ordered_scenarios = SCENARIO_PAPER_ORDER

    overview: list[dict] = []
    for method in sorted(scenario_scores["Method"].unique().tolist()):
        method_scores = scenario_scores[scenario_scores["Method"] == method]
        split_scores = method_scores.set_index(["Split", "Scenario"])
        for scenario in ordered_scenarios:
            if ("Base", scenario) not in split_scores.index:
                continue
            if ("Generalization", scenario) not in split_scores.index:
                continue

            base_ds = float(split_scores.loc[("Base", scenario), "DS"])
            base_success = 100.0 * float(split_scores.loc[("Base", scenario), "Success"])
            gen_ds = float(split_scores.loc[("Generalization", scenario), "DS"])
            gen_success = 100.0 * float(split_scores.loc[("Generalization", scenario), "Success"])
            base_hm = harmonic_mean(base_ds, base_success)
            gen_hm = harmonic_mean(gen_ds, gen_success)

            overview.append(
                {
                    "Method": method,
                    "Class": next(
                        class_name
                        for class_name, scenarios in SCENARIO_CLASSES.items()
                        if scenario in scenarios
                    ),
                    "Scenario": scenario,
                    "Base HM": base_hm,
                    "Gen HM": gen_hm,
                    "Drop %": pct_change(gen_hm, base_hm),
                }
            )

    return overview


def build_class_hm_overview(rows: pd.DataFrame) -> list[dict]:
    if rows.empty:
        return []

    overview: list[dict] = []
    for method in sorted(rows["Method"].unique().tolist()):
        method_rows = rows[rows["Method"] == method]
        for class_name in CLASS_PAPER_ORDER:
            scenarios = SCENARIO_CLASSES[class_name]
            class_rows = method_rows[method_rows["Scenario"].isin(scenarios)]
            if class_rows.empty:
                continue

            class_scores = class_rows.groupby("Split", as_index=True)[["DS", "Success"]].mean()
            if "Base" not in class_scores.index or "Generalization" not in class_scores.index:
                continue

            base_ds = float(class_scores.loc["Base", "DS"])
            base_success = 100.0 * float(class_scores.loc["Base", "Success"])
            gen_ds = float(class_scores.loc["Generalization", "DS"])
            gen_success = 100.0 * float(class_scores.loc["Generalization", "Success"])
            base_hm = harmonic_mean(base_ds, base_success)
            gen_hm = harmonic_mean(gen_ds, gen_success)

            overview.append(
                {
                    "Method": method,
                    "Class": class_name,
                    "Base HM": base_hm,
                    "Gen HM": gen_hm,
                    "Drop %": pct_change(gen_hm, base_hm),
                }
            )

    return overview


def print_hm_overview(
    title: str,
    rows: list[dict],
    key_fields: tuple[str, ...],
) -> None:
    print("")
    print(f"{title}:")
    if not rows:
        print("No complete Base/Generalization overview available.")
        return

    columns = [*key_fields, "Base HM", "Gen HM", "Rel %"]
    widths: dict[str, int] = {}
    for column in columns:
        widths[column] = len(column)

    for row in rows:
        for column in key_fields:
            widths[column] = max(widths[column], len(str(row[column])))
        widths["Base HM"] = max(widths["Base HM"], len(f"{row['Base HM']:.1f}"))
        widths["Gen HM"] = max(widths["Gen HM"], len(f"{row['Gen HM']:.1f}"))
        widths["Rel %"] = max(widths["Rel %"], len(f"{row['Drop %']:.1f}%"))

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    print(header)
    print(separator)

    for row in rows:
        values = [
            *(str(row[column]).ljust(widths[column]) for column in key_fields),
            f"{row['Base HM']:.1f}".rjust(widths["Base HM"]),
            f"{row['Gen HM']:.1f}".rjust(widths["Gen HM"]),
            f"{row['Drop %']:.1f}%".rjust(widths["Rel %"]),
        ]
        print(" | ".join(values))


def build_latex_row(table: pd.DataFrame, method: str, b2d_score: Optional[float]) -> str:
    method_table = table[table["Method"] == method]
    if method_table.empty:
        raise ValueError(f"Method {method!r} was not found in the parsed results.")

    split_scores = method_table.set_index("Split")
    required_splits = {"Base", "Generalization"}
    missing_splits = sorted(required_splits.difference(split_scores.index))
    if missing_splits:
        raise ValueError(
            "Cannot build full LaTeX main-table row because split data is missing: "
            + ", ".join(missing_splits)
        )

    base_ds = float(split_scores.loc["Base", "DS"])
    base_success = float(split_scores.loc["Base", "Success"])
    gen_ds = float(split_scores.loc["Generalization", "DS"])
    gen_success = float(split_scores.loc["Generalization", "Success"])
    hm_base = harmonic_mean(base_ds, base_success)
    hm_gen = harmonic_mean(gen_ds, gen_success)
    b2d_value = "B2D_TODO" if b2d_score is None else f"{b2d_score:.1f}"

    return (
        f"{method} & {b2d_value} & {base_ds:.1f} & {base_success:.1f} & {hm_base:.1f} "
        f"& {gen_ds:.1f} \\small{{({pct_change(gen_ds, base_ds):.1f}\\%)}} "
        f"& {gen_success:.1f} \\small{{({pct_change(gen_success, base_success):.1f}\\%)}} "
        f"& {hm_gen:.1f} \\small{{({pct_change(hm_gen, hm_base):.1f}\\%)}}\\\\"
    )


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    route_dir = Path(args.route_dir)
    try:
        route_map = build_route_mapping(route_dir)
    except RuntimeError:
        route_map = {}
    rows, warnings, strict_failure = load_rows(args.method, results_root, route_map)
    table = build_main_table(rows)

    if rows.empty:
        raise SystemExit("No valid rows found.")

    print_hm_overview(
        "Per-Scenario HM Overview (scenario classes)",
        build_scenario_hm_overview(rows),
        ("Class", "Scenario"),
    )
    print_hm_overview(
        "Per-Class HM Overview (scenario classes)",
        build_class_hm_overview(rows),
        ("Class",),
    )

    print("")
    print(f"Parsed rows: {len(rows)}")
    print(f"Seeds: {', '.join(sorted(rows['Seed'].astype(str).unique().tolist()))}")
    if warnings:
        print("")
        for warning in warnings:
            print(warning)
    print("")
    print("Summary:")
    summary_rows = build_summary_rows(table, args.b2d_score)
    if not summary_rows:
        print("No complete Base/Generalization summary available.")
    else:
        for summary in summary_rows:
            print(f"B2D: {summary['B2D']}")
            pairs = [
                ("DS", summary["Base DS"], summary["Gen DS"]),
                ("Success", summary["Base Success"], summary["Gen Success"]),
                ("HM", summary["Base HM"], summary["Gen HM"]),
            ]
            label_width = max(len(label) for label, _, _ in pairs)
            base_width = max(len(base) for _, base, _ in pairs)
            for label, base_value, gen_value in pairs:
                print(
                    f"{label.ljust(label_width)}: Base {base_value.ljust(base_width)}   Gen {gen_value}"
                )

    try:
        latex_row = build_latex_row(table, args.method, args.b2d_score)
        print("")
        print("LaTeX row:")
        print(latex_row)
    except ValueError as exc:
        print("")
        print(f"LaTeX row unavailable: {exc}")

    if args.strict and strict_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
