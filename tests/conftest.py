"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_video_path(tmp_path):
    """Create a dummy video file path for tests."""
    p = tmp_path / "test_shot.mp4"
    p.write_bytes(b"")  # empty file — real video tests handled separately
    return str(p)
