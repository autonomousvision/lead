from lead.common.constants import LeadSemanticSegmentationClass
from lead.policy.transfuser.labels import BEVSemanticClass, BoundingBoxClass
from lead.policy.transfuser.visualization import colors


class TestColorConverters:
    """Tests for the visualization color mappings."""

    def test_bev_semantic_color_converter(self):
        """Test TransFuser BEV semantic class to color mapping."""
        converter = colors.CARLA_TRANSFUSER_BEV_SEMANTIC_COLOR_CONVERTER
        keys = list(converter.keys())

        # Check keys are in ascending order
        for i in range(len(keys) - 1):
            assert keys[i] < keys[i + 1], (
                f"Keys not in ascending order: {keys[i]} >= {keys[i + 1]}"
            )

        # Check all TransFuser BEV semantic classes have colors
        for bev_class in BEVSemanticClass:
            assert bev_class in keys, f"Missing key: {bev_class}"

        # Verify colors are RGB tuples
        for color in converter.values():
            assert isinstance(color, tuple), f"Color is not a tuple: {color}"
            assert len(color) == 3, f"Color does not have 3 components: {color}"
            assert all(0 <= c <= 255 for c in color), (
                f"Color values out of range [0, 255]: {color}"
            )

    def test_semantic_color_converter(self):
        """Test TransFuser semantic segmentation class to color mapping."""
        converter = colors.TRANSFUSER_SEMANTIC_COLORS
        keys = list(converter.keys())

        # Check keys are in ascending order
        for i in range(len(keys) - 1):
            assert keys[i] < keys[i + 1], (
                f"Keys not in ascending order: {keys[i]} >= {keys[i + 1]}"
            )

        # Check all LEAD semantic classes have colors
        for semantic_class in LeadSemanticSegmentationClass:
            assert semantic_class in keys, f"Missing key: {semantic_class}"

        # Verify colors are RGB tuples
        for color in converter.values():
            assert isinstance(color, tuple), f"Color is not a tuple: {color}"
            assert len(color) == 3, f"Color does not have 3 components: {color}"
            assert all(0 <= c <= 255 for c in color), (
                f"Color values out of range [0, 255]: {color}"
            )

    def test_bounding_box_color_converter(self):
        """Test TransFuser bounding box class to color mapping."""
        converter = colors.TRANSFUSER_BOUNDING_BOX_COLORS
        keys = list(converter.keys())

        # Check keys are in ascending order
        for i in range(len(keys) - 1):
            assert keys[i] < keys[i + 1], (
                f"Keys not in ascending order: {keys[i]} >= {keys[i + 1]}"
            )

        # Check all TransFuser bounding box classes have colors
        for bbox_class in BoundingBoxClass:
            assert bbox_class in keys, f"Missing key: {bbox_class}"

        # Verify colors are RGB tuples
        for color in converter.values():
            assert isinstance(color, tuple), f"Color is not a tuple: {color}"
            assert len(color) == 3, f"Color does not have 3 components: {color}"
            assert all(0 <= c <= 255 for c in color), (
                f"Color values out of range [0, 255]: {color}"
            )
