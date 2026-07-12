"""Quality checks for the route XML files in src/lead/routes/data_routes."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from lead.common import runtime
from lead.common.constants.towns import ALL_TOWNS

DATA_ROUTES_ROOT = runtime.project_root() / "src" / "lead" / "routes" / "data_routes"


@pytest.fixture(scope="module")
def parsed_routes() -> list[tuple[Path, ET.Element]]:
    """Parse every route XML under data_routes once for all tests.

    Returns:
        Pairs of file path and parsed XML root element.
    """
    xml_paths = sorted(DATA_ROUTES_ROOT.rglob("*.xml"))
    assert len(xml_paths) > 0, f"No route XML files found under {DATA_ROUTES_ROOT}"
    return [(path, ET.parse(path).getroot()) for path in xml_paths]


def test_each_xml_has_exactly_one_route(
    parsed_routes: list[tuple[Path, ET.Element]],
) -> None:
    """Unsplit multi-route bundles must never sit among the per-route files.

    collect_data.py submits one SLURM job per XML and derives the job name and
    result path from the file stem, so a bundle file (e.g. Town12.xml with 50
    routes) silently becomes one broken job.
    """
    violations = [
        f"{path}: {len(root.findall('route'))} routes"
        for path, root in parsed_routes
        if len(root.findall("route")) != 1
    ]
    assert not violations, "XML files without exactly one <route>:\n" + "\n".join(
        violations,
    )


def test_all_towns_are_covered(parsed_routes: list[tuple[Path, ET.Element]]) -> None:
    """Every CARLA town has at least one route, and no route names an unknown town."""
    towns_found = {
        route.attrib["town"]
        for _, root in parsed_routes
        for route in root.findall("route")
    }
    missing = set(ALL_TOWNS) - towns_found
    unknown = towns_found - set(ALL_TOWNS)
    assert not missing, f"Towns without any route: {sorted(missing)}"
    assert not unknown, f"Routes reference unknown towns: {sorted(unknown)}"


def test_each_route_has_town_and_id(
    parsed_routes: list[tuple[Path, ET.Element]],
) -> None:
    """collect_data.py reads route.attrib["town"]; the leaderboard needs id."""
    violations = [
        f"{path}: attributes {sorted(route.attrib)}"
        for path, root in parsed_routes
        for route in root.findall("route")
        if "town" not in route.attrib or "id" not in route.attrib
    ]
    assert not violations, "Routes missing town/id attribute:\n" + "\n".join(violations)
