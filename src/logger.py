"""Shared logging helpers for pipeline stages."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
	"""Return the standard logger for a pipeline module.

	Args:
		name: Logger name, typically ``__name__``.

	Returns:
		The named logger.
	"""
	return logging.getLogger(name)
