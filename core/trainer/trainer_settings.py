from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass
class TrainingSettings:
    model: str
    data: str
    epochs: int
    batch: float
    imgsz: int
    device: str
    optimizer: str
    project: str
    name: str
    exist_ok: bool = True
    time: float = None
    patience: int = 100
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    # Augmentation parameters
    weight_decay: float = 0.0005
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    bgr: float = 0.0
    mosaic: float = 1.0
    mixup: float = 0.0
    cutmix: float = 0.0
    copy_paste: float = 0.0
    copy_paste_mode: str = "flip"
    auto_augment: str = "none"
    erasing: float = 0.4
    output_dir: str = None

    def to_train_overrides(self) -> dict:
        """Keyword arguments for ultralytics' YOLO.train() (plus "model",
        the weights to start from), matching what the old `yolo` command
        line passed: `project` is the parent of the run's output folder
        and `name` the run folder itself.
        """
        overrides = asdict(self)
        output_dir = overrides.pop("output_dir")
        overrides["project"] = str(Path(output_dir).parent)
        overrides["task"] = "segment"
        if not overrides.get("time"):
            overrides["time"] = None
        return overrides
