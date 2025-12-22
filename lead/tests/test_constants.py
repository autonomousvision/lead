import unittest

from lead.common import constants
from lead.common.constants import (
    CarlaSemanticSegmentationClass,
    ChaffeurNetBEVSemanticClass,
    TransfuserBEVSemanticClass,
    TransfuserBoundingBoxClass,
    TransfuserSemanticSegmentationClass,
)


class TestConverters(unittest.TestCase):
    def test_semantic_segmentation_class_converter(self):
        converter = constants.SEMANTIC_SEGMENTATION_CONVERTER
        keys = list(converter.keys())
        values = list(converter.values())

        for i in range(len(keys) - 1):
            self.assertEqual(keys[i] + 1, keys[i + 1])

        for i in CarlaSemanticSegmentationClass:
            self.assertIn(i, keys, f"Missing key: {i}")

        for i in TransfuserSemanticSegmentationClass:
            self.assertIn(i, values, f"Missing value: {i}")

    def test_bev_semantic_class_converter(self):
        converter = constants.CHAFFEURNET_TO_TRANSFUSER_BEV_SEMANTIC_CONVERTER
        keys = list(converter.keys())

        for i in range(len(keys) - 1):
            self.assertEqual(keys[i] + 1, keys[i + 1])

        for i in ChaffeurNetBEVSemanticClass:
            self.assertIn(i, keys, f"Missing key: {i}")

    def test_bev_semantic_color_converter(self):
        converter = constants.CARLA_TRANSFUSER_BEV_SEMANTIC_COLOR_CONVERTER
        keys = list(converter.keys())

        for i in range(len(keys) - 1):
            self.assertLess(keys[i], keys[i + 1], "Keys are not in ascending order")

        for i in TransfuserBEVSemanticClass:
            self.assertIn(i, keys, f"Missing key: {i}")

    def test_semantic_color_converter(self):
        converter = constants.TRANSFUSER_SEMANTIC_COLORS
        keys = list(converter.keys())

        for i in range(len(keys) - 1):
            self.assertLess(keys[i], keys[i + 1], "Keys are not in ascending order")

        for i in TransfuserSemanticSegmentationClass:
            self.assertIn(i, keys, f"Missing key: {i}")

    def test_bounding_box_color_converter(self):
        converter = constants.TRANSFUSER_BOUNDING_BOX_COLORS
        keys = list(converter.keys())

        for i in range(len(keys) - 1):
            self.assertLess(keys[i], keys[i + 1], "Keys are not in ascending order")

        for i in TransfuserBoundingBoxClass:
            self.assertIn(i, keys, f"Missing key: {i}")


if __name__ == "__main__":
    unittest.main()
