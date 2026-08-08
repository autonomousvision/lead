import pytest

from lead.common.constants import CarlaSemanticSegmentationClass
from lead.config.policy.transfuser.label_classes import PerspectiveSemanticClass
from lead.policy.transfuser.utils import semantics


class TestSemanticSegmentationConverter:
    """Tests the CARLA to TransFuser semantic segmentation class converter."""

    def test_converter_covers_every_class(self):
        converter = semantics.SEMANTIC_SEGMENTATION_CONVERTER
        keys = list(converter.keys())
        values = list(converter.values())

        # Check keys are consecutive integers
        for i in range(len(keys) - 1):
            assert keys[i] + 1 == keys[i + 1], (
                f"Keys not consecutive: {keys[i]} -> {keys[i + 1]}"
            )

        # Check all CARLA classes are mapped
        for carla_class in CarlaSemanticSegmentationClass:
            assert carla_class in keys, f"Missing key: {carla_class}"

        # Classes without a CARLA camera class are painted from boxes at load time.
        box_painted_classes = {
            PerspectiveSemanticClass.OBSTACLE,
            PerspectiveSemanticClass.SPECIAL_VEHICLE,
            PerspectiveSemanticClass.STOP_SIGN,
        }
        for semantic_class in PerspectiveSemanticClass:
            if semantic_class in box_painted_classes:
                assert semantic_class not in values, (
                    f"Unexpected value: {semantic_class}"
                )
            else:
                assert semantic_class in values, f"Missing value: {semantic_class}"


class TestSemanticClass:
    """Tests that a box class maps to the expected derived semantic class."""

    @pytest.mark.parametrize(
        ("box_class", "expected"),
        [
            ("ego_car", PerspectiveSemanticClass.UNLABELED),
            ("car", PerspectiveSemanticClass.VEHICLE),
            ("walker", PerspectiveSemanticClass.PEDESTRIAN),
            ("traffic_light", PerspectiveSemanticClass.TRAFFIC_LIGHT),
            ("stop_sign", PerspectiveSemanticClass.UNLABELED),
            ("static_prop_car", PerspectiveSemanticClass.VEHICLE),
        ],
    )
    def test_class_string_determines_semantics(self, box_class, expected):
        assert semantics.semantic_class({"class": box_class}) == expected

    @pytest.mark.parametrize(
        ("mesh_path", "expected"),
        [
            ("Foo/ParkedVehicles/Car01", PerspectiveSemanticClass.VEHICLE),
            ("Foo/Props/Cone01", PerspectiveSemanticClass.OBSTACLE),
            (None, PerspectiveSemanticClass.OBSTACLE),
        ],
    )
    def test_static_resolves_via_mesh_path(self, mesh_path, expected):
        box = {"class": "static", "mesh_path": mesh_path}
        assert semantics.semantic_class(box) == expected

    def test_static_without_mesh_path_key_is_obstacle(self):
        assert (
            semantics.semantic_class({"class": "static"})
            == PerspectiveSemanticClass.OBSTACLE
        )
