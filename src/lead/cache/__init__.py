"""The cache store: per-tensor codecs and the per-log LMDB backend.

Model-agnostic. A policy's dataset declares which sample-part outputs are
cacheable and with which codec; ``lead.training.build_cache`` stores them once
per ``(sample, sensor view)``, and training reads them back instead of
recomputing the builders every epoch.
"""
