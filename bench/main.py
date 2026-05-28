from pathlib import Path

from dadinhos.generate import make_detection_task
from modelinhos.sample import read_samples


def main(path: Path = Path("data/blobs/annotations.json")):
    if not path.exists():
        make_detection_task(
            path,
            resolution=(480, 640),
            n_samples=1000,
            n_classes=3,
        )

    print(len(read_samples(path)))


if __name__ == "__main__":
    main()
