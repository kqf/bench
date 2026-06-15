from pathlib import Path

import cv2
from dadinhos.generate import make_detection_task
from joblib import Memory
from modelinhos.evaluation import (
    mean_average_precision,
    per_sample_metrics,
    visualize_fp_fn,
)
from modelinhos.plot import plot
from modelinhos.processing import LabelEncoder, Sample
from modelinhos.sample import read_samples
from modelinhos.ssd.inference import TorchvisionDetector
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

memory = Memory(location=".cache", verbose=0)


@memory.cache
def infer(
    resolution: tuple[int, int],
    samples: list[Sample],
) -> tuple[list[Sample], LabelEncoder]:
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    model = TorchvisionDetector(
        resolution=resolution,
        build_model=ssdlite320_mobilenet_v3_large,
        weights=weights,
        lencoder=LabelEncoder(
            l2i={
                label: i
                for i, label in enumerate(
                    weights.meta["categories"],
                )
            },
        ),
    )

    return model.transform(samples), model.label_encoder


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

    train = read_samples(path)
    # model.fit(le.transform(train)) ~
    y_pred, le = infer(resolution, train)

    for i, (true, pred) in enumerate(zip(train, y_pred)):
        if i > 10:
            continue
        frame = cv2.imread(str(true.file_name))
        cv2.imshow("frame", plot(frame, pred))
        cv2.waitKey()

    l2i = {
        "0": 0,
        "1": 0,
        "2": 0,
        **{k: 0 for k, _ in le.l2i.items() if "background" not in k},
    }
    m_ap = mean_average_precision(train, y_pred, l2i=l2i)
    print(m_ap["mAP"].iloc[0])

    aps = per_sample_metrics(train, y_pred, l2i=l2i)
    visualize_fp_fn(aps, i2l={0: "any"})


if __name__ == "__main__":
    main()
