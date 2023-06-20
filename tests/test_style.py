"""Test that the src and tests directories pass ruff checks."""
import subprocess

__all__ = ["test_ruff_src_dir", "test_ruff_tests_dir"]


def test_ruff_src_dir() -> None:
    """Test that the src directory passes ruff checks."""
    completed = subprocess.run(["ruff", "check", "src"], capture_output=True, text=True)
    assert completed.returncode == 0, (
        "Ruff Linting failed on src directory \n" + completed.stdout
    )


def test_ruff_tests_dir() -> None:
    """Test that the tests directory passes ruff checks."""
    completed = subprocess.run(
        ["ruff", "check", "tests"], capture_output=True, text=True
    )
    assert completed.returncode == 0, (
        "Ruff Linting Failed on tests directory \n" + completed.stdout
    )
