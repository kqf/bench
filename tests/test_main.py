import pytest

from bench.main import main


@pytest.fixture
def dataset(tmp_path):
    return tmp_path / "data" / "tests" / "annotations.json"


def test_main(dataset):
    main(dataset, n_samples=100)
