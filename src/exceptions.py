"""Pipeline-specific exception hierarchy."""

from __future__ import annotations


class PipelineError(Exception):
	"""Base class for expected pipeline failures."""


class DataIntegrityError(PipelineError):
	"""Raised when input data violates a pipeline integrity requirement."""


class SchemaError(PipelineError):
	"""Raised when input data does not conform to its required schema."""


class ConfigError(PipelineError):
	"""Raised when a pipeline configuration is invalid for an operation."""
