"""Convenience re-exports of the label classes stored in LEAD logs.

py123d resolves each log's label classes by the module path stored in its
metadata, and that path is the defining module ``lead.api.py123d_log_api`` —
which therefore must never move; this module only re-exports."""

from lead.api.py123d_log_api import (
    CarlaBoxDetectionLabel,
    CarlaCameraSegmentationLabel,
)

__all__ = ["CarlaBoxDetectionLabel", "CarlaCameraSegmentationLabel"]
