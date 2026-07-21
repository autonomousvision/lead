"""Resolver target for recorded logs: py123d resolves each log's label classes
by the module path stored in its metadata, which is this module for the
published dataset."""

from lead.api.py123d_log_api import (
    CarlaBoxDetectionLabel,
    CarlaCameraSegmentationLabel,
)

__all__ = ["CarlaBoxDetectionLabel", "CarlaCameraSegmentationLabel"]
