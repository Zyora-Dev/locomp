"""Shared pytest fixtures and marks."""

import platform
import pytest

macos_only = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Test requires macOS (Metal GPU)",
)
