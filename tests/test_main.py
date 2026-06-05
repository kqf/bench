from unittest.mock import patch

import pytest

from bench.main import main


@pytest.fixture
def dataset(tmp_path):
    return tmp_path / "data" / "tests" / "annotations.json"


@patch("bench.main.cv2.imshow")
def test_main(imshow, dataset):
    main(dataset, n_samples=100)
