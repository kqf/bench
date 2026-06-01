from pathlib import Path

import cv2
from dadinhos.generate import make_detection_task
from modelinhos.evaluation import (
    mean_average_precision,
    per_sample_metrics,
    visualize_fp_fn,
)
from modelinhos.processing import LabelEncoder, Sample
from modelinhos.sample import read_samples
from modelinhos.ssd.inference import TorchvisionDetector
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)


class Clamp:
    def fit(self, X):
        return self

    def transform(self, X: list[Sample]) -> list[Sample]:
        for sample in X:
            for ann in sample.annotations:
                ann.label = min(int(ann.label), 2)
        return X


def main(
    path: Path = Path("data/blobs/annotations.json"),
    resolution: tuple[int, int] = (480, 640),  # height, width
    n_samples: int = 1000,
):
    if not path.exists():
        make_detection_task(
            path,
            resolution=resolution,
            n_samples=n_samples,
            n_classes=3,
        )

    model = TorchvisionDetector(
        resolution=resolution,
        build_model=ssdlite320_mobilenet_v3_large,
        weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
    )

    train = read_samples(path)
    le = LabelEncoder().fit(train)
    # model.fit(le.transform(train)) ~
    y_pred = model.transform(
        [cv2.imread(path.parent / sample.file_name) for sample in train]
    )
    # Fix the indexing issues
    y_pred = Clamp().transform(y_pred)

    y_pred = le.inverse_transform(y_pred)
    m_ap = mean_average_precision(train, y_pred, l2i=le.l2i)
    print(m_ap["mAP"].iloc[0])

    aps = per_sample_metrics(train, y_pred, l2i=le.l2i)
    visualize_fp_fn(aps, i2l=le.i2l)


if __name__ == "__main__":
    main()
