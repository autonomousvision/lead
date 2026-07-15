import pytest

from lead.common import constants
from lead.common.constants import LeadSemanticSegmentationClass


class TestSemanticClass:
    """Tests that a box class maps to the expected derived semantic class."""

    @pytest.mark.parametrize(
        ("box_class", "expected"),
        [
            ("ego_car", LeadSemanticSegmentationClass.UNLABELED),
            ("car", LeadSemanticSegmentationClass.VEHICLE),
            ("walker", LeadSemanticSegmentationClass.PEDESTRIAN),
            ("traffic_light", LeadSemanticSegmentationClass.TRAFFIC_LIGHT),
            ("stop_sign", LeadSemanticSegmentationClass.UNLABELED),
            ("static_prop_car", LeadSemanticSegmentationClass.VEHICLE),
        ],
    )
    def test_class_string_determines_semantics(self, box_class, expected):
        assert constants.semantic_class({"class": box_class}) == expected

    @pytest.mark.parametrize(
        ("mesh_path", "expected"),
        [
            ("Foo/ParkedVehicles/Car01", LeadSemanticSegmentationClass.VEHICLE),
            ("Foo/Props/Cone01", LeadSemanticSegmentationClass.OBSTACLE),
            (None, LeadSemanticSegmentationClass.OBSTACLE),
        ],
    )
    def test_static_resolves_via_mesh_path(self, mesh_path, expected):
        box = {"class": "static", "mesh_path": mesh_path}
        assert constants.semantic_class(box) == expected

    def test_static_without_mesh_path_key_is_obstacle(self):
        assert (
            constants.semantic_class({"class": "static"})
            == LeadSemanticSegmentationClass.OBSTACLE
        )
