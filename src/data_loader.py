"""Shared validation helpers for tidy pipeline inputs."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from exceptions import SchemaError
from logger import get_logger


logger = get_logger(__name__)


def validate_schema(
	df: pd.DataFrame, required_columns: Sequence[str]
) -> None:
	"""Ensure a dataframe contains all required columns.

	Args:
		df: Dataframe to validate.
		required_columns: Column names that must be present.

	Raises:
		SchemaError: If required columns are absent.
	"""
	missing = [column for column in required_columns if column not in df.columns]
	if missing:
		message = f"Missing required columns: {', '.join(missing)}"
		logger.error(message)
		raise SchemaError(message)
