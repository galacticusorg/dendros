"""Tests for completion-status validation."""
from __future__ import annotations

import pytest

from dendros import open_outputs


def test_complete_file_does_not_raise(single_file):
    with open_outputs(single_file) as c:
        c.validate_completion(mode="error")  # should not raise


def test_incomplete_file_raises_by_default(incomplete_file):
    with open_outputs(incomplete_file) as c:
        with pytest.raises(RuntimeError, match="incomplete"):
            c.validate_completion()


def test_incomplete_file_raises_explicit(incomplete_file):
    with open_outputs(incomplete_file) as c:
        with pytest.raises(RuntimeError):
            c.validate_completion(mode="error")


def test_incomplete_file_warns(incomplete_file):
    with open_outputs(incomplete_file) as c:
        with pytest.warns(UserWarning):
            c.validate_completion(mode="warn")


def test_incomplete_file_ignored(incomplete_file):
    with open_outputs(incomplete_file) as c:
        c.validate_completion(mode="ignore")  # should not raise or warn


def test_invalid_mode(single_file):
    with open_outputs(single_file) as c:
        with pytest.raises(ValueError, match="mode"):
            c.validate_completion(mode="silent")
