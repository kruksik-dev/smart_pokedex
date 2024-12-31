import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def mock_output_path() -> Generator[Path, None, None]:
    """
    Fixture that creates a temporary directory named 'pokemon_images'
    and automatically cleans it up after the test.

    Returns:
        Path: The path to the 'pokemon_images' directory.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "pokemon_images"
        output_path.mkdir(parents=True, exist_ok=True)
        yield output_path
