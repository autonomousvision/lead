from lead.common import constants
from lead.common.constants import (
    CarlaSemanticSegmentationClass,
    LeadSemanticSegmentationClass,
)


class TestConstantConverters:
    """Tests for constant converter dictionaries and enums."""

    def test_semantic_segmentation_class_converter(self):
        """Test CARLA to LEAD semantic segmentation class converter."""
        converter = constants.SEMANTIC_SEGMENTATION_CONVERTER
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

        # LEAD classes without a CARLA camera class are painted from boxes at load time.
        box_painted_classes = {
            LeadSemanticSegmentationClass.OBSTACLE,
            LeadSemanticSegmentationClass.SPECIAL_VEHICLE,
            LeadSemanticSegmentationClass.STOP_SIGN,
        }
        for lead_class in LeadSemanticSegmentationClass:
            if lead_class in box_painted_classes:
                assert lead_class not in values, f"Unexpected value: {lead_class}"
            else:
                assert lead_class in values, f"Missing value: {lead_class}"
